"""Análise: métricas por condição + testes OFAT contra o baseline, com Holm-Bonferroni.

Unidade de análise = query (as repetições viram a média por query). Isso respeita o
pareamento: a mesma query é vista por todas as condições, então o teste é pareado e
não trata repetições da mesma query como observações independentes.

Teste: permutação pareada (troca de sinal), sem suposição de normalidade e sem
depender de scipy. Reporta também o tamanho de efeito bruto em pontos percentuais,
que é o critério de decisão prática (MIN_RELEVANT_EFFECT).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import ALPHA, MIN_RELEVANT_EFFECT
from run_experiment import BASELINE, RAW_CSV

METRICS = ["correct", "hallucinated", "invented_tool", "lure_hit", "parse_error", "retrieval_hit"]
N_PERM = 10_000
RNG = np.random.default_rng(20260830)


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    if (bad := (df["error"] != "").sum()):
        print(f"AVISO: {bad} linhas com erro descartadas da análise")
        df = df[df["error"] == ""]
    for col in METRICS + ["prompt_tokens", "total_tokens", "latency_s", "stage1_prompt_tokens", "stage1_latency_s"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # custo total da linha inclui a 1ª etapa do two_stage
    df["cost_tokens"] = df["prompt_tokens"] + df["stage1_prompt_tokens"]
    df["cost_latency_s"] = df["latency_s"] + df["stage1_latency_s"]
    return df


def condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "toolset_size", "retrieval", "invocation"]
    out = df.groupby(keys).agg(
        n=("correct", "size"),
        acuracia=("correct", "mean"),
        alucinacao=("hallucinated", "mean"),
        ferramenta_inventada=("invented_tool", "mean"),
        atraiu_lure=("lure_hit", "mean"),
        erro_parsing=("parse_error", "mean"),
        recall_retrieval=("retrieval_hit", "mean"),
        tokens_prompt_medio=("cost_tokens", "mean"),
        latencia_media_s=("cost_latency_s", "mean"),
    ).round(4)
    return out.reset_index()


def by_query_type(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "toolset_size", "retrieval", "invocation", "query_type"]
    return df.groupby(keys)["correct"].mean().round(4).unstack("query_type").reset_index()


def _paired_permutation(diff: np.ndarray) -> float:
    """p bilateral por troca de sinal. diff = por-query (variante - baseline)."""
    diff = diff[~np.isnan(diff)]
    if len(diff) == 0 or np.allclose(diff, 0):
        return 1.0
    observed = abs(diff.mean())
    signs = RNG.choice([-1.0, 1.0], size=(N_PERM, len(diff)))
    null = np.abs((signs * diff).mean(axis=1))
    return float((np.sum(null >= observed - 1e-12) + 1) / (N_PERM + 1))


def holm(pvals: list[float], alpha: float = ALPHA) -> list[bool]:
    """Holm-Bonferroni: controla FWER sem supor independência favorável."""
    order = np.argsort(pvals)
    m = len(pvals)
    rejected = [False] * m
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            rejected[i] = True
        else:
            break  # step-down: a partir da primeira falha, nada mais é rejeitado
    return rejected


def _cell(df, model, cell, metric):
    """Média por query de uma célula do desenho — série indexada por query_id."""
    sub = df[(df.model == model) & (df.toolset_size == cell["toolset_size"]) &
             (df.retrieval == cell["retrieval"]) & (df.invocation == cell["invocation"])]
    return sub.groupby("query_id")[metric].mean()


def ofat_tests(df: pd.DataFrame, metric: str = "correct") -> pd.DataFrame:
    from config import INVOCATION_MODES, RETRIEVAL_MODES, TOOLSET_SIZES

    axes = {
        "exposicao": [{**BASELINE, "toolset_size": s} for s in TOOLSET_SIZES],
        "recuperacao": [{**BASELINE, "retrieval": r} for r in RETRIEVAL_MODES],
        "invocacao": [{**BASELINE, "invocation": i} for i in INVOCATION_MODES],
    }
    rows = []
    for model in sorted(df.model.unique()):
        base = _cell(df, model, BASELINE, metric)
        for axis, cells in axes.items():
            for cell in cells:
                label = {"exposicao": str(cell["toolset_size"]),
                         "recuperacao": cell["retrieval"],
                         "invocacao": cell["invocation"]}[axis]
                if cell == BASELINE:
                    continue  # o baseline não é comparado consigo mesmo
                variant = _cell(df, model, cell, metric)
                if variant.empty:
                    continue
                paired = pd.concat([variant, base], axis=1, join="inner")
                if paired.empty:
                    continue
                diff = (paired.iloc[:, 0] - paired.iloc[:, 1]).to_numpy()
                rows.append({
                    "model": model, "eixo": axis, "variante": label,
                    "n_queries": len(diff),
                    "media_variante": round(paired.iloc[:, 0].mean(), 4),
                    "media_baseline": round(paired.iloc[:, 1].mean(), 4),
                    "delta_pp": round(diff.mean() * 100, 2),
                    "p": _paired_permutation(diff),
                })

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["significativo_holm"] = holm(out["p"].tolist())
    out["relevante"] = out["delta_pp"].abs() >= MIN_RELEVANT_EFFECT * 100
    # o que importa para a conclusão: significativo E acima do efeito mínimo relevante
    out["acionavel"] = out["significativo_holm"] & out["relevante"]
    out["p"] = out["p"].round(4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=RAW_CSV)
    ap.add_argument("--metric", default="correct", choices=METRICS)
    args = ap.parse_args()

    df = load(args.input)
    outdir = args.input.parent
    pd.set_option("display.width", 200, "display.max_columns", 50)

    summary = condition_summary(df)
    print(f"\n{'='*100}\nRESUMO POR CONDIÇÃO\n{'='*100}")
    print(summary.to_string(index=False))
    summary.to_csv(outdir / "summary_by_condition.csv", index=False)

    print(f"\n{'='*100}\nACURÁCIA POR TIPO DE QUERY\n{'='*100}")
    print(by_query_type(df).to_string(index=False))
    by_query_type(df).to_csv(outdir / "accuracy_by_query_type.csv", index=False)

    tests = ofat_tests(df, args.metric)
    print(f"\n{'='*100}\nTESTES OFAT vs BASELINE ({BASELINE}) — métrica: {args.metric}"
          f"\nHolm-Bonferroni a {ALPHA:.0%}; efeito mínimo relevante: {MIN_RELEVANT_EFFECT:.0%}\n{'='*100}")
    print(tests.to_string(index=False) if not tests.empty else "(sem comparações — dados insuficientes)")
    if not tests.empty:
        tests.to_csv(outdir / f"ofat_tests_{args.metric}.csv", index=False)

    print(f"\nArquivos gravados em {outdir}")


if __name__ == "__main__":
    # sanidade do Holm: passo-a-passo conhecido
    assert holm([0.001, 0.04, 0.9], 0.05) == [True, False, False]
    assert holm([0.001, 0.01, 0.02], 0.05) == [True, True, True]
    # permutação: diferença nula não pode ser significativa
    assert _paired_permutation(np.zeros(20)) == 1.0
    assert _paired_permutation(np.ones(20)) < 0.01
    main()
