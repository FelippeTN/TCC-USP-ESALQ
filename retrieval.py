"""Eixo 2 — recuperação: quais ferramentas do toolset chegam ao modelo.

full      → todas (baseline; custo de prompt cresce com o toolset)
embedding → top-k por similaridade de cosseno query↔descrição (qwen3-embedding-8B)
hybrid    → fusão RRF entre embedding e busca léxica (sobrepesa nomes/termos exatos)
two_stage → 1ª chamada ao LLM escolhe o domínio, 2ª recebe só as ferramentas dele

two_stage é a única que gasta uma chamada extra por query — é a linha a cortar
primeiro se o orçamento apertar.
"""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

from config import CORE_DOMAINS, RETRIEVAL_K

RETRIEVAL_MODES = ("full", "embedding", "hybrid", "two_stage")

CACHE_DIR = Path(__file__).parent / "results"
_RRF_K = 60  # constante padrão do Reciprocal Rank Fusion (Cormack et al., 2009)


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", _norm(text)))


def tool_document(tool: dict) -> str:
    """Texto indexado de uma ferramenta: nome + descrição + parâmetros."""
    params = " ".join(tool["parameters"])
    return f"{tool['name'].replace('_', ' ')}. {tool['description']}. Parâmetros: {params}"


class EmbeddingIndex:
    """Cache de embeddings em disco, chaveado pelo texto. Evita re-embedar 52
    descrições a cada uma das milhares de chamadas do plano.

    O cache é separado por backend: rodar com --backend mock e depois com real
    compartilhando o mesmo arquivo envenena o índice com vetores falsos, e a
    recuperação passa a errar sem levantar erro nenhum.
    """

    def __init__(self, client, namespace: str = "real"):
        self.client = client
        self.path = CACHE_DIR / f"embedding_cache_{namespace}.json"
        self.cache: dict[str, list[float]] = {}
        if self.path.exists():
            self.cache = json.loads(self.path.read_text(encoding="utf-8"))

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            for vec, text in zip(self.client.embed(missing), missing):
                self.cache[text] = vec
            self.save()
        if len(dims := {len(self.cache[t]) for t in texts}) > 1:
            raise ValueError(f"cache {self.path.name} tem dimensões misturadas {dims} — apague o arquivo")
        arr = np.array([self.cache[t] for t in texts], dtype=np.float64)
        return arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12, None)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache), encoding="utf-8")


def _embedding_ranking(index: EmbeddingIndex, query: str, tools: list[dict]) -> list[int]:
    docs = index.encode([tool_document(t) for t in tools])
    q = index.encode([query])[0]
    return list(np.argsort(-(docs @ q)))


def _lexical_ranking(query: str, tools: list[dict]) -> list[int]:
    """Sobreposição de tokens normalizada pelo tamanho do documento — BM25 pobre,
    mas suficiente: o papel do braço léxico aqui é pegar nome exato de ferramenta.
    ponytail: se a busca léxica virar objeto de estudo, trocar por BM25 de verdade."""
    q = _tokens(query)
    scores = []
    for t in tools:
        doc = _tokens(tool_document(t))
        scores.append(len(q & doc) / (len(doc) ** 0.5 or 1))
    return list(np.argsort(-np.array(scores)))


def _rrf(*rankings: list[int]) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (_RRF_K + rank + 1)
    return sorted(scores, key=lambda i: -scores[i])


_DOMAIN_PROMPT = (
    "Classifique o pedido do usuário em exatamente um domínio desta lista: {domains}.\n"
    "Responda apenas com o nome do domínio, sem mais nada."
)


def _pick_domain(client, model_key: str, query: str, tools: list[dict]) -> tuple[str | None, dict]:
    domains = sorted({t["domain"] for t in tools})
    r = client.chat(model_key, [
        {"role": "system", "content": _DOMAIN_PROMPT.format(domains=", ".join(domains))},
        {"role": "user", "content": query},
    ], max_tokens=32)
    answer = _norm(r["message"].get("content") or "")
    chosen = next((d for d in domains if _norm(d) in answer), None)
    return chosen, r


def select_tools(mode: str, query: str, tools: list[dict], *, index=None, client=None,
                 model_key=None, k: int = RETRIEVAL_K) -> tuple[list[dict], dict]:
    """Retorna (ferramentas expostas, info). `info` carrega o custo da chamada extra
    do two_stage (usage/latência), que precisa entrar na contabilidade de custo."""
    if mode == "full":
        return tools, {}

    if mode in ("embedding", "hybrid"):
        emb = _embedding_ranking(index, query, tools)
        order = emb if mode == "embedding" else _rrf(emb, _lexical_ranking(query, tools))
        return [tools[i] for i in order[:k]], {}

    if mode == "two_stage":
        domain, r = _pick_domain(client, model_key, query, tools)
        info = {"stage1_usage": r["usage"], "stage1_latency_s": r["latency_s"], "stage1_domain": domain}
        subset = [t for t in tools if t["domain"] == domain]
        # domínio inválido/alucinado: cai para o toolset inteiro em vez de expor nada
        return (subset or tools), info

    raise ValueError(f"modo de recuperação desconhecido: {mode}")


if __name__ == "__main__":
    from client import MockClient
    from corpus import build_toolset

    tools = build_toolset(50)
    assert len(select_tools("full", "oi", tools)[0]) == 50

    # o braço léxico tem que achar a ferramenta pelo nome exato
    lex = _lexical_ranking("preciso usar export conversation", tools)
    assert tools[lex[0]]["name"] == "export_conversation", tools[lex[0]]["name"]

    # RRF: item bem colocado nos dois rankings vence um que só vai bem em um
    assert _rrf([0, 1, 2], [0, 2, 1])[0] == 0

    class _StubIndex:
        """Embeddings determinísticos e degenerados só para exercitar o caminho de código."""
        def encode(self, texts):
            a = np.array([[hash((t, i)) % 97 for i in range(8)] for t in texts], dtype=np.float64)
            return a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)

    for mode in ("embedding", "hybrid"):
        sel, info = select_tools(mode, "manda mensagem", tools, index=_StubIndex(), k=5)
        assert len(sel) == 5 and info == {}, mode

    sel, info = select_tools("two_stage", "manda mensagem", tools, client=MockClient(), model_key="gemma-4-e4b")
    assert sel and "stage1_usage" in info
    assert {t["domain"] for t in sel} <= set(CORE_DOMAINS)

    # cache com dimensões misturadas (mock + real no mesmo arquivo) tem que estourar,
    # não degradar a recuperação em silêncio
    idx = EmbeddingIndex(MockClient(), namespace="_selftest")
    idx.cache = {"a": [0.1] * 64, "b": [0.1] * 4096}
    try:
        idx.encode(["a", "b"])
        raise AssertionError("cache envenenado passou sem erro")
    except ValueError as e:
        assert "dimensões misturadas" in str(e)
    idx.path.unlink(missing_ok=True)

    print("retrieval.py: ok")
