"""Cliente HTTP para os endpoints OpenAI-compatíveis + backend mock para testar o pipeline."""
import hashlib
import json
import random
import re
import time

import requests

from config import EMBEDDING_MODEL, LLM_MODELS

TIMEOUT = 120
MAX_RETRIES = 3


class LLMError(RuntimeError):
    pass


def _post(url: str, payload: dict) -> dict:
    """POST com retry exponencial. Erros 4xx não são retentados (é bug de payload, não flake)."""
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(url, json=payload, timeout=TIMEOUT)
            if 400 <= r.status_code < 500:
                raise LLMError(f"{r.status_code} {r.text[:500]}")
            r.raise_for_status()
            return r.json()
        except LLMError:
            raise
        except Exception as e:  # rede, 5xx, timeout
            last = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    raise LLMError(f"falhou após {MAX_RETRIES} tentativas: {last}")


def chat(model_key: str, messages: list[dict], tools: list[dict] | None = None,
         max_tokens: int = 512, temperature: float = 0.0) -> dict:
    """Uma chamada de chat. Retorna {message, usage, latency_s}."""
    cfg = LLM_MODELS[model_key]
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    t0 = time.perf_counter()
    data = _post(f"{cfg['base_url']}/chat/completions", payload)
    latency = time.perf_counter() - t0
    return {
        "message": data["choices"][0]["message"],
        "usage": data.get("usage", {}),
        "latency_s": latency,
    }


def embed(texts: list[str]) -> list[list[float]]:
    data = _post(f"{EMBEDDING_MODEL['base_url']}/embeddings",
                 {"model": EMBEDDING_MODEL["model"], "input": texts})
    return [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]


# ---------------------------------------------------------------- mock backend
# Determinístico por (query, ferramentas): permite rodar o pipeline inteiro sem
# tocar nos servidores, para validar plano/parsing/CSV antes de gastar GPU.

def _seed(*parts) -> random.Random:
    return random.Random(int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16], 16))


class MockClient:
    """Simula respostas: 70% acerta a 1ª ferramenta plausível, senão erra ou não chama nada."""

    def chat(self, model_key, messages, tools=None, max_tokens=512, temperature=0.0):
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

        # o modo de invocação não vem como parâmetro: é inferido do formato pedido,
        # que é exatamente o que o modelo real também tem que fazer
        if tools:
            mode, names = "native", [t["function"]["name"] for t in tools]
        else:
            mode = "code_action" if "Funções Python" in system else "json_prompt"
            names = re.findall(r"^- (\w+)\(", system, re.M)

        rng = _seed(model_key, user_text, mode, len(names))
        picked = rng.choice(names) if names and rng.random() < 0.7 else None

        if mode == "native":
            message = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "mock", "type": "function",
                "function": {"name": picked, "arguments": json.dumps({})},
            }]} if picked else {"role": "assistant", "content": "Sem ferramenta aplicável.", "tool_calls": None}
        elif mode == "json_prompt":
            body = ({"tool": picked, "arguments": {}} if picked
                    else {"tool": None, "answer": "Sem ferramenta aplicável."})
            message = {"role": "assistant", "content": json.dumps(body), "tool_calls": None}
        else:
            message = {"role": "assistant", "content": f"{picked}()" if picked else "pass", "tool_calls": None}

        n_in = sum(len(str(m.get("content") or "")) for m in messages) // 4 + len(json.dumps(tools or [])) // 4
        return {
            "message": message,
            "usage": {"prompt_tokens": n_in, "completion_tokens": 30, "total_tokens": n_in + 30},
            "latency_s": 0.001,
        }

    def embed(self, texts):
        return [[_seed(t, i).gauss(0, 1) for i in range(64)] for t in texts]


class RealClient:
    chat = staticmethod(chat)
    embed = staticmethod(embed)


def get_client(backend: str):
    return MockClient() if backend == "mock" else RealClient()


if __name__ == "__main__":
    # o mock precisa responder no formato de cada modo, senão o pipeline "passa"
    # sem nunca exercitar os parsers de json_prompt/code_action
    from invocation import INVOCATION_MODES, build_request, parse_response

    c = MockClient()
    tools = [{"name": "send_message", "description": "envia", "parameters": {"to": "string"}},
             {"name": "get_user", "description": "busca", "parameters": {"user_id": "string"}}]

    for mode in INVOCATION_MODES:
        seen = set()
        for q in ("manda oi pro joao", "quem e o u_1", "bom dia", "qual a capital"):
            msgs, tp = build_request(mode, q, tools)
            r = c.chat("deepseek-v4-flash", msgs, tools=tp)
            predicted, parse_error = parse_response(mode, r["message"])
            assert not parse_error, f"{mode}: mock produziu saída que o parser não lê"
            assert predicted in (None, "send_message", "get_user"), (mode, predicted)
            assert r["usage"]["total_tokens"] > 0
            seen.add(predicted)
        assert seen != {None}, f"{mode}: mock nunca escolheu ferramenta"

        # determinismo: mesma entrada, mesma saída
        msgs, tp = build_request(mode, "manda oi pro joao", tools)
        assert c.chat("x", msgs, tools=tp)["message"] == c.chat("x", msgs, tools=tp)["message"]

    assert len(c.embed(["a", "b"])) == 2
    print("client.py: ok (mock cobre os 3 modos de invocação)")
