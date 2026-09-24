"""Executor do experimento — desenho OFAT (one-factor-at-a-time) em torno de um baseline.

Baseline: toolset=50, retrieval=full, invocation=native. Cada eixo varia sozinho a
partir dele, então toda comparação tem exatamente um fator diferente.

Uso:
    py src/run_experiment.py --plan-only              # só o plano e a estimativa de custo
    py src/run_experiment.py --backend mock           # saída isolada em mock_results_v2.csv
    py src/run_experiment.py --backend real --repetitions 1 --limit 8   # piloto barato
    py src/run_experiment.py --backend real           # plano completo (retoma de onde parou)
"""
import argparse
import csv
import hashlib
import itertools
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

from client import get_client
from config import (EMBEDDING_MODEL, INVOCATION_MODES, LLM_MODELS, REPETITIONS, RETRIEVAL_K,
                    RETRIEVAL_MODES, TOOLSET_SIZES)
from corpus import build_toolset_for_query
from invocation import build_request, parse_response
from queries import QUERIES
from retrieval import EmbeddingIndex, select_tools

BASELINE = {"toolset_size": 50, "retrieval": "full", "invocation": "native"}

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RAW_CSV = RESULTS_DIR / "raw_results.csv"
PROTOCOL_VERSION = 2

LEGACY_FIELDS = [
    "run_id", "model", "toolset_size", "retrieval", "invocation", "repetition",
    "query_id", "query_type", "axis",
    "expected", "lure", "predicted", "n_exposed", "retrieval_hit",
    "correct", "hallucinated", "invented_tool", "lure_hit", "parse_error",
    "prompt_tokens", "completion_tokens", "total_tokens", "latency_s",
    "stage1_prompt_tokens", "stage1_completion_tokens", "stage1_latency_s",
    "error", "timestamp",
]
FIELDS = LEGACY_FIELDS + ["backend", "protocol_version", "correct_valid_parse", "response_message"]


def build_conditions(models: list[str]) -> list[dict]:
    """Condições do desenho OFAT, sem duplicar o baseline entre os eixos."""
    conditions, seen = [], set()
    variants = [
        ("exposicao", [{**BASELINE, "toolset_size": s} for s in TOOLSET_SIZES]),
        ("recuperacao", [{**BASELINE, "retrieval": r} for r in RETRIEVAL_MODES]),
        ("invocacao", [{**BASELINE, "invocation": i} for i in INVOCATION_MODES]),
    ]
    for model, (axis, cells) in itertools.product(models, variants):
        for cell in cells:
            cond = {"model": model, "axis": axis, **cell}
            key = (model, cond["toolset_size"], cond["retrieval"], cond["invocation"])
            if key in seen:
                continue  # baseline aparece nos 3 eixos; roda uma vez, é comparado nos 3
            seen.add(key)
            conditions.append(cond)
    return conditions


def run_id(cond: dict, query_id: str, rep: int) -> str:
    return (f"{cond['model']}|{cond['toolset_size']}|{cond['retrieval']}"
            f"|{cond['invocation']}|{query_id}|r{rep}")


def load_done(path: Path, backend: str) -> set[str]:
    """Valida a origem e retoma somente execuções válidas do mesmo protocolo."""
    if path.resolve() == RAW_CSV.resolve() or (path.exists() and RAW_CSV.exists() and path.samefile(RAW_CSV)):
        raise ValueError("raw_results.csv é a coleta histórica; escolha outro --out para o protocolo 2.")
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != FIELDS:
            raise ValueError("CSV legado ou cabeçalho incompatível; escolha um novo --out.")
        done = set()
        for row in reader:
            if row.get("backend") != backend or row.get("protocol_version") != str(PROTOCOL_VERSION):
                raise ValueError("O arquivo contém outro backend ou protocolo; use um --out separado.")
            if not row.get("error"):
                if not row.get("run_id") or row["run_id"] in done:
                    raise ValueError("CSV com run_id ausente ou execução válida duplicada.")
                done.add(row["run_id"])
        return done


def record_metadata(path: Path, backend: str) -> None:
    """Registra o ambiente local; não infere a revisão dos pesos ou o hardware remoto."""
    root = Path(__file__).parent
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "backend": backend,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "requests")},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))},
        "model_ids": {name: cfg["model"] for name, cfg in LLM_MODELS.items()},
        "embedding_model_id": EMBEDDING_MODEL["model"],
        "remote_weights_revision": None,
        "remote_hardware": None,
    }
    manifest = path.with_suffix(path.suffix + ".meta.json")
    if manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in metadata.items()):
            raise ValueError("Ambiente ou código mudou desde a coleta; escolha outro --out.")
    elif path.exists():
        raise ValueError("CSV sem manifesto de origem; escolha outro --out.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata["created_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with manifest.open("x", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            f.write("\n")


def execute_one(client, index, cond: dict, query: dict, rep: int, *, backend: str) -> dict:
    tools = build_toolset_for_query(cond["toolset_size"], query)
    row = {
        "run_id": run_id(cond, query["id"], rep), "model": cond["model"],
        "toolset_size": cond["toolset_size"], "retrieval": cond["retrieval"],
        "invocation": cond["invocation"], "repetition": rep,
        "query_id": query["id"], "query_type": query["type"], "axis": cond["axis"],
        "expected": "|".join(query["expected"]), "lure": query["lure"] or "",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "error": "",
        "backend": backend, "protocol_version": PROTOCOL_VERSION,
    }
    try:
        exposed, info = select_tools(
            cond["retrieval"], query["text"], tools,
            index=index, client=client, model_key=cond["model"], k=RETRIEVAL_K)

        messages, tools_param = build_request(cond["invocation"], query["text"], exposed)
        r = client.chat(cond["model"], messages, tools=tools_param)
        row["response_message"] = json.dumps(r["message"], ensure_ascii=False)
        predicted, parse_error = parse_response(cond["invocation"], r["message"])

        exposed_names = {t["name"] for t in exposed}
        expected = set(query["expected"])
        row.update({
            "predicted": predicted or "",
            "n_exposed": len(exposed),
            # o gabarito sobreviveu à recuperação? separa erro de retrieval de erro do modelo
            "retrieval_hit": int(bool(expected & exposed_names)) if expected else 1,
            "correct": int(predicted in expected if expected else predicted is None),
            "correct_valid_parse": int(not parse_error and (predicted in expected if expected else predicted is None)),
            # chamou ferramenta quando não devia, ou inventou nome inexistente
            "hallucinated": int(bool(predicted) and (not expected or predicted not in exposed_names)),
            "invented_tool": int(bool(predicted) and predicted not in exposed_names),
            "lure_hit": int(predicted == query["lure"] if query["lure"] else 0),
            "parse_error": int(parse_error),
            "prompt_tokens": r["usage"].get("prompt_tokens", 0),
            "completion_tokens": r["usage"].get("completion_tokens", 0),
            "total_tokens": r["usage"].get("total_tokens", 0),
            "latency_s": round(r["latency_s"], 4),
            "stage1_prompt_tokens": info.get("stage1_usage", {}).get("prompt_tokens", 0),
            "stage1_completion_tokens": info.get("stage1_usage", {}).get("completion_tokens", 0),
            "stage1_latency_s": round(info.get("stage1_latency_s", 0.0), 4),
        })
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:300]
    return row


def print_plan(conditions, queries, reps):
    print(f"\n{'='*72}\nPLANO DO EXPERIMENTO\n{'='*72}")
    print(f"{'modelo':<20} {'eixo':<12} {'tools':>5} {'retrieval':<11} {'invocation':<12} {'chamadas':>9}")
    print("-" * 72)
    total = extra = 0
    for c in conditions:
        n = len(queries) * reps
        e = n if c["retrieval"] == "two_stage" else 0
        total, extra = total + n, extra + e
        print(f"{c['model']:<20} {c['axis']:<12} {c['toolset_size']:>5} "
              f"{c['retrieval']:<11} {c['invocation']:<12} {n + e:>9}")
    print("-" * 72)
    print(f"condições: {len(conditions)} | queries: {len(queries)} | repetições: {reps}")
    print(f"chamadas ao LLM: {total + extra}  (sendo {extra} da 2ª etapa do two_stage)")
    # 0.67 s/linha medido no piloto de 2026-08-30 (192 linhas em 129 s, os dois modelos)
    print(f"Estimativa histórica do piloto: ~{total * 0.67 / 3600:.1f} h em série. "
          "Latência HTTP não mede diretamente tempo de GPU ou custo financeiro.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "real"], default="mock")
    ap.add_argument("--models", nargs="+", choices=list(LLM_MODELS), default=list(LLM_MODELS))
    ap.add_argument("--repetitions", type=int, default=REPETITIONS)
    ap.add_argument("--limit", type=int, help="usa só as N primeiras queries (piloto)")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--out", type=Path, help="CSV separado por backend e protocolo; retoma se compatível")
    args = ap.parse_args()
    if args.repetitions < 1 or (args.limit is not None and not 1 <= args.limit <= len(QUERIES)):
        ap.error("Use repetições positivas e --limit entre 1 e o total de consultas.")
    args.out = (args.out or RESULTS_DIR / f"{args.backend}_results_v{PROTOCOL_VERSION}.csv").resolve()

    queries = QUERIES[:args.limit] if args.limit else QUERIES
    conditions = build_conditions(args.models)

    print_plan(conditions, queries, args.repetitions)
    if args.plan_only:
        return

    try:
        done = load_done(args.out, args.backend)
        record_metadata(args.out, args.backend)
    except (ValueError, OSError) as e:
        ap.error(str(e))
    if done:
        print(f"retomando: {len(done)} execuções já gravadas serão puladas\n")

    client = get_client(args.backend)
    index = EmbeddingIndex(client, namespace=args.backend)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.out.exists()
    with args.out.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        jobs = [(c, q, rep) for c in conditions for q in queries
                for rep in range(1, args.repetitions + 1)
                if run_id(c, q["id"], rep) not in done]
        errors = 0
        for n, (cond, query, rep) in enumerate(jobs, 1):
            row = execute_one(client, index, cond, query, rep, backend=args.backend)
            writer.writerow(row)
            f.flush()  # dado primário: grava linha a linha, queda não perde o lote
            errors += bool(row["error"])
            if n % 25 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} — {errors} erros", flush=True)

    print(f"\nresultados em {args.out}")
    if errors:
        print(f"{errors} linhas com erro — rode de novo para reprocessar só elas")
        sys.exit(1)


if __name__ == "__main__":
    main()
