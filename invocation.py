"""Eixo 3 — modos de invocação: como a ferramenta é exposta e como a resposta é lida.

native      → API de tool calling do servidor (campo `tools`), resposta em `tool_calls`.
json_prompt → schemas no system prompt, modelo devolve um objeto JSON.
code_action → schemas no system prompt, modelo devolve uma chamada Python.

Cada modo tem build_* (prompt) e parse_* (resposta → nome da ferramenta ou None).
Falha de parsing é um resultado válido do experimento, não uma exceção: vira
tool=None + parse_error=True, e a taxa de parse_error é uma das métricas.
"""
import ast
import json
import re

INVOCATION_MODES = ("native", "json_prompt", "code_action")

_BASE = (
    "Você é um assistente que opera um sistema de mensageria corporativa. "
    "Se nenhuma ferramenta for adequada para o pedido do usuário, responda em texto normal, "
    "sem chamar ferramenta nenhuma."
)


def to_openai_schema(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": {
                "type": "object",
                "properties": {k: {"type": v} for k, v in tool["parameters"].items()},
                "required": list(tool["parameters"]),
            },
        },
    }


def _render_signatures(tools: list[dict]) -> str:
    return "\n".join(
        f"- {t['name']}({', '.join(f'{k}: {v}' for k, v in t['parameters'].items())}) — {t['description']}"
        for t in tools
    )


def build_request(mode: str, query: str, tools: list[dict]) -> tuple[list[dict], list[dict] | None]:
    """Retorna (messages, tools_param). tools_param só é usado no modo native."""
    if mode == "native":
        return [{"role": "system", "content": _BASE},
                {"role": "user", "content": query}], [to_openai_schema(t) for t in tools]

    if mode == "json_prompt":
        system = (
            f"{_BASE}\n\nFerramentas disponíveis:\n{_render_signatures(tools)}\n\n"
            'Responda APENAS com um objeto JSON, sem texto em volta e sem cercas de código.\n'
            'Para usar uma ferramenta: {"tool": "<nome>", "arguments": {<parâmetros>}}\n'
            'Se nenhuma servir: {"tool": null, "answer": "<sua resposta em texto>"}'
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": query}], None

    if mode == "code_action":
        system = (
            f"{_BASE}\n\nFunções Python disponíveis:\n{_render_signatures(tools)}\n\n"
            "Responda APENAS com uma única linha de código Python chamando a função adequada, "
            "com argumentos nomeados, sem cercas de código e sem comentários.\n"
            "Se nenhuma função servir, responda apenas com: pass"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": query}], None

    raise ValueError(f"modo de invocação desconhecido: {mode}")


_FENCE = re.compile(r"```(?:json|python)?\s*(.*?)```", re.S)


def _strip_fence(text: str) -> str:
    m = _FENCE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


def parse_response(mode: str, message: dict) -> tuple[str | None, bool]:
    """Retorna (nome_da_ferramenta | None, parse_error)."""
    if mode == "native":
        calls = message.get("tool_calls") or []
        return (calls[0]["function"]["name"], False) if calls else (None, False)

    text = _strip_fence(message.get("content") or "")
    if not text:
        return None, True

    if mode == "json_prompt":
        try:
            # modelos às vezes prefaciam o JSON; pega o primeiro objeto balanceado
            start = text.index("{")
            obj = json.JSONDecoder().raw_decode(text[start:])[0]
        except (ValueError, json.JSONDecodeError):
            return None, True
        if not isinstance(obj, dict) or "tool" not in obj:
            return None, True
        tool = obj["tool"]
        if tool is None:
            return None, False
        return (tool, False) if isinstance(tool, str) else (None, True)

    if mode == "code_action":
        line = text.splitlines()[0].strip() if text.splitlines() else ""
        try:
            node = ast.parse(line, mode="exec").body[0]
        except SyntaxError:
            return None, True
        if isinstance(node, ast.Pass):
            return None, False
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            return node.value.func.id, False
        return None, True

    raise ValueError(f"modo de invocação desconhecido: {mode}")


if __name__ == "__main__":
    tools = [{"name": "send_message", "description": "envia", "parameters": {"to": "string"}}]

    for mode in INVOCATION_MODES:
        msgs, tp = build_request(mode, "manda oi", tools)
        assert msgs[-1]["content"] == "manda oi"
        assert (tp is not None) == (mode == "native")
        if mode != "native":
            assert "send_message" in msgs[0]["content"], f"{mode}: schema faltando no prompt"

    # native
    assert parse_response("native", {"tool_calls": [{"function": {"name": "send_message"}}]}) == ("send_message", False)
    assert parse_response("native", {"tool_calls": None, "content": "oi"}) == (None, False)

    # json_prompt: limpo, com cerca, com prefácio, recusa, e lixo
    assert parse_response("json_prompt", {"content": '{"tool": "send_message", "arguments": {}}'}) == ("send_message", False)
    assert parse_response("json_prompt", {"content": '```json\n{"tool": "send_message"}\n```'}) == ("send_message", False)
    assert parse_response("json_prompt", {"content": 'Claro! {"tool": "send_message"}'}) == ("send_message", False)
    assert parse_response("json_prompt", {"content": '{"tool": null, "answer": "oi"}'}) == (None, False)
    assert parse_response("json_prompt", {"content": 'não sei'}) == (None, True)

    # code_action
    assert parse_response("code_action", {"content": 'send_message(to="joao")'}) == ("send_message", False)
    assert parse_response("code_action", {"content": '```python\nsend_message(to="joao")\n```'}) == ("send_message", False)
    assert parse_response("code_action", {"content": 'pass'}) == (None, False)
    assert parse_response("code_action", {"content": 'send_message(to='}) == (None, True)
    assert parse_response("code_action", {"content": '42'}) == (None, True)

    print("invocation.py: ok")
