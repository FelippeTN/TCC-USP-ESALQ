"""Executor do experimento — desenho OFAT (one-factor-at-a-time) em torno de um baseline.

Baseline: toolset=50, retrieval=full, invocation=native. Cada eixo varia sozinho a
partir dele, então toda comparação tem exatamente um fator diferente.

Uso:
    py run_experiment.py --plan-only              # só o plano e a estimativa de custo
    py run_experiment.py --backend mock           # pipeline inteiro sem tocar na GPU
    py run_experiment.py --backend real --repetitions 1 --limit 8   # piloto barato
    py run_experiment.py --backend real           # plano completo (retoma de onde parou)
"""
import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

from client import get_client
from config import (INVOCATION_MODES, LLM_MODELS, REPETITIONS, RETRIEVAL_K,
                    RETRIEVAL_MODES, TOOLSET_SIZES)
from corpus import build_toolset_for_query
from invocation import build_request, parse_response
from queries import QUERIES
from retrieval import EmbeddingIndex, select_tools

BASELINE = {"toolset_size": 50, "retrieval": "full", "invocation": "native"}

RESULTS_DIR = Path(__file__).parent / "results"
RAW_CSV = RESULTS_DIR / "raw_results.csv"

FIELDS = [
    "run_id", "model", "toolset_size", "retrieval", "invocation", "repetition",
    "query_id", "query_type", "axis",
    "expected", "lure", "predicted", "n_exposed", "retrieval_hit",
    "correct", "hallucinated", "invented_tool", "lure_hit", "parse_error",
    "prompt_tokens", "completion_tokens", "total_tokens", "latency_s",
    "stage1_prompt_tokens", "stage1_completion_tokens", "stage1_latency_s",
    "error", "timestamp",
]


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


def load_done() -> set[str]:
    """run_ids já gravados — permite retomar depois de queda/rate limit sem repetir."""
    if not RAW_CSV.exists():
        return set()
    with RAW_CSV.open(encoding="utf-8", newline="") as f:
        return {row["run_id"] for row in csv.DictReader(f) if not row.get("error")}


def execute_one(client, index, cond: dict, query: dict, rep: int) -> dict:
    tools = build_toolset_for_query(cond["toolset_size"], query)
    row = {
        "run_id": run_id(cond, query["id"], rep), "model": cond["model"],
        "toolset_size": cond["toolset_size"], "retrieval": cond["retrieval"],
        "invocation": cond["invocation"], "repetition": rep,
        "query_id": query["id"], "query_type": query["type"], "axis": cond["axis"],
        "expected": "|".join(query["expected"]), "lure": query["lure"] or "",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "error": "",
    }
    try:
        exposed, info = select_tools(
            cond["retrieval"], query["text"], tools,
            index=index, client=client, model_key=cond["model"], k=RETRIEVAL_K)

        messages, tools_param = build_request(cond["invocation"], query["text"], exposed)
        r = client.chat(cond["model"], messages, tools=tools_param)
        predicted, parse_error = parse_response(cond["invocation"], r["message"])

        exposed_names = {t["name"] for t in exposed}
        expected = set(query["expected"])
        row.update({
            "predicted": predicted or "",
            "n_exposed": len(exposed),
            # o gabarito sobreviveu à recuperação? separa erro de retrieval de erro do modelo
            "retrieval_hit": int(bool(expected & exposed_names)) if expected else 1,
            "correct": int(predicted in expected if expected else predicted is None),
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
    print("Servidores locais: o custo é tempo de GPU, não token. A ~0,67 s/linha medido "
          f"no piloto, são ~{total * 0.67 / 3600:.1f} h em série.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mock", "real"], default="mock")
    ap.add_argument("--models", nargs="+", default=list(LLM_MODELS))
    ap.add_argument("--repetitions", type=int, default=REPETITIONS)
    ap.add_argument("--limit", type=int, help="usa só as N primeiras queries (piloto)")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--out", type=Path, default=RAW_CSV)
    args = ap.parse_args()

    queries = QUERIES[:args.limit] if args.limit else QUERIES
    conditions = build_conditions(args.models)

    print_plan(conditions, queries, args.repetitions)
    if args.plan_only:
        return

    done = load_done() if args.out == RAW_CSV else set()
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
            row = execute_one(client, index, cond, query, rep)
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
