"""Configuração central do experimento. Edite aqui, não espalhe constantes pelo código.

O endereço do servidor de inferência vem de TCC_SERVER_HOST (arquivo .env, fora do
git). Sem valor padrão de propósito: este repositório é público, e um default com o
IP real derrota o motivo de ele estar no .env.
"""
import os
from pathlib import Path

ENV_PATH = Path(__file__).parent / ".env"


def _load_env(path: Path = ENV_PATH) -> None:
    """Lê KEY=value do .env sem sobrescrever o que já veio do ambiente.

    ponytail: parser mínimo (comentários, aspas, espaços). Trocar por python-dotenv
    se o .env virar algo mais complicado que três variáveis.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_env()

SERVER_HOST = os.environ.get("TCC_SERVER_HOST")
if not SERVER_HOST:
    raise RuntimeError(
        "TCC_SERVER_HOST não definido. Copie .env.example para .env e preencha "
        "com o endereço do servidor de inferência (o .env não vai para o git)."
    )

# portas e ids confirmados via GET /v1/models em 2026-08-30
LLM_MODELS = {
    "deepseek-v4-flash": {
        "base_url": f"http://{SERVER_HOST}:8000/v1",
        "model": "DeepSeek-V4-Flash-0731",
    },
    "gemma-4-e4b": {
        "base_url": f"http://{SERVER_HOST}:8001/v1",
        "model": "/srv/models/gemma-4-E4B-it",
    },
}

EMBEDDING_MODEL = {
    "base_url": f"http://{SERVER_HOST}:8002/v1",
    "model": "/srv/models/qwen3-embedding-8B",
}

# Eixo 1 — exposição: tamanho do toolset
TOOLSET_SIZES = [10, 30, 50]

# Eixo 2 — recuperação de ferramentas
# `random` é PISO de comparação, não uma técnica candidata: expõe k ferramentas
# sorteadas. Sem ele não se distingue "a recuperação achou o que importa" de
# "reduzir o toolset já ajuda por si só" — embedding/hybrid precisam superá-lo
# para que o ganho seja atribuído à relevância, e não ao tamanho do prompt.
RETRIEVAL_MODES = ["full", "random", "embedding", "hybrid", "two_stage"]
RETRIEVAL_K = 5

# Eixo 3 — modo de invocação
INVOCATION_MODES = ["native", "json_prompt", "code_action"]

# Corpus sintético — domínios núcleo (4 × ~13 ferramentas = pool de 50)
CORE_DOMAINS = ["messages", "users", "groups", "notifications"]

REPETITIONS = 5

# Estatística — decididos antes de rodar (pré-registro)
ALPHA = 0.05
MULTIPLE_COMPARISON_CORRECTION = "holm"
MIN_RELEVANT_EFFECT = 0.05  # 5 p.p. de acurácia: diferença menor que isso é significativa mas irrelevante

# preço aproximado por 1M tokens (input, output) — ajustar conforme tabela real do provedor local
TOKEN_PRICE_USD = {
    "deepseek-v4-flash": (0.0, 0.0),  # self-hosted: custo é de GPU/tempo, não por token — usar como 0 e medir latência à parte
    "gemma-4-e4b": (0.0, 0.0),
}

AVG_PROMPT_TOKENS_PER_TOOL = 60  # estimativa grosseira p/ schema de 1 ferramenta no prompt (nome+params+desc)
AVG_QUERY_TOKENS = 40
AVG_COMPLETION_TOKENS = 120
