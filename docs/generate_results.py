"""Gera figuras e tabelas do README sem executar ou modificar o experimento.

Uso, na raiz: py docs/generate_results.py
Dependências: numpy, pandas, requests (do projeto) e matplotlib (só para figuras).
"""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import analyze as analysis
from config import REPETITIONS
from queries import QUERIES
from run_experiment import build_conditions, run_id

OUT = ROOT / "docs"
MODELS = ["deepseek-v4-flash", "gemma-4-e4b"]
NAMES = ["DeepSeek-V4-Flash", "Gemma-4-E4B"]
COLORS = ["#156082", "#D46A33"]
CELLS = [(50, "full", "native"), (10, "full", "native"),
         (30, "full", "native"), (50, "random", "native"),
         (50, "embedding", "native"), (50, "hybrid", "native"),
         (50, "two_stage", "native"), (50, "full", "json_prompt"),
         (50, "full", "code_action")]
LABELS = ["Baseline · 50 / full / native", "10 ferramentas", "30 ferramentas",
          "random · 5 ferramentas", "embedding · 5 ferramentas",
          "hybrid · 5 ferramentas", "two_stage · domínio", "json_prompt", "code_action"]
KEYS = ["model", "toolset_size", "retrieval", "invocation"]


def cell(df, model, condition, metric="correct"):
    return analysis._cell(df, model, dict(zip(KEYS[1:], condition)), metric)


def paired_tests(df, variants, baseline=CELLS[0], metric="correct"):
    """Uma família Holm por chamada; mesmas queries e teste do analyze.py."""
    analysis.RNG = np.random.default_rng(20260830)
    rows = []
    for model in MODELS:
        base = cell(df, model, baseline, metric)
        for condition in variants:
            pair = pd.concat([cell(df, model, condition, metric), base], axis=1).dropna()
            assert len(pair) == 64, "Pareamento incompleto"
            diff = (pair.iloc[:, 0] - pair.iloc[:, 1]).to_numpy()
            rows.append(dict(model=model, toolset_size=condition[0], retrieval=condition[1],
                             invocation=condition[2], n_queries=len(pair),
                             media_variante=pair.iloc[:, 0].mean(), media_baseline=base.mean(),
                             delta_pp=diff.mean() * 100, p=analysis._paired_permutation(diff)))
    table = pd.DataFrame(rows)
    order = np.argsort(table.p.to_numpy())
    adjusted = np.minimum(1, np.maximum.accumulate(table.p.to_numpy()[order] *
                                                   np.arange(len(table), 0, -1)))
    table["p_holm"] = 0.0
    table.loc[order, "p_holm"] = adjusted
    table["significativo_holm"] = analysis.holm(table.p.tolist())
    assert np.array_equal(table.p_holm <= analysis.ALPHA, table.significativo_holm)
    table["relevante"] = table.delta_pp.abs() >= analysis.MIN_RELEVANT_EFFECT * 100 - 1e-10
    return table


def save(fig, name, note):
    fig.text(0.01, 0.01, note, fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, 0.13 if fig.legends else 0.055, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"{name}.{ext}", dpi=220, facecolor="white")
    plt.close(fig)


def main():
    for folder in ("figures", "data"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    protected = list(ROOT.glob("*.py")) + list((ROOT / "results").glob("*.csv"))
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    raw = pd.read_csv(analysis.RAW_CSV, keep_default_na=False)
    df = analysis.load(analysis.RAW_CSV)
    expected = {run_id(c, q["id"], r) for c in build_conditions(MODELS)
                for q in QUERIES for r in range(1, REPETITIONS + 1)}
    assert not df.run_id.duplicated().any(), "Há execuções válidas duplicadas"
    assert set(df.run_id) == expected, "Plano incompleto ou condições inesperadas"
    assert len(df) == 5760 and df.groupby(KEYS).size().eq(320).all()
    assert df.groupby(KEYS + ["query_id"]).size().eq(5).all()
    # Sensibilidade separada: nenhuma falha de parsing conta como acerto.
    df["correct_valid_parse"] = df.correct * (1 - df.parse_error)
    summary = analysis.condition_summary(df)
    types = analysis.by_query_type(df)
    tests = paired_tests(df, CELLS[1:])
    random_tests = paired_tests(df, CELLS[4:6], baseline=CELLS[3])
    sensitivity = paired_tests(df, CELLS[1:], metric="correct_valid_parse")
    reliability = df.groupby(KEYS).agg(acuracia_original=("correct", "mean"),
                                      acuracia_sem_falha_parsing=("correct_valid_parse", "mean"),
                                      erro_parsing=("parse_error", "mean"))
    availability = df[df.expected != ""].groupby(KEYS).agg(
        n=("retrieval_hit", "size"), disponibilidade_gabarito=("retrieval_hit", "mean"))
    for name, table in [("summary_by_condition", summary), ("accuracy_by_query_type", types),
                        ("ofat_tests_correct", tests), ("exploratory_vs_random", random_tests),
                        ("sensitivity_tests", sensitivity), ("sensitivity_summary", reliability.reset_index()),
                        ("retrieval_with_tool", availability.reset_index())]:
        table.to_csv(OUT / "data" / f"{name}.csv", index=False)
    audit = dict(raw_sha256=hashes[analysis.RAW_CSV], rows_raw=len(raw), rows_valid=len(df),
                 rows_error=int(raw.error.ne("").sum()), valid_duplicates=int(df.run_id.duplicated().sum()),
                 queries=int(df.query_id.nunique()), conditions=len(summary), repetitions=REPETITIONS,
                 timestamp_min=df.timestamp.min(), timestamp_max=df.timestamp.max(),
                 parsing_errors=int(df.parse_error.sum()),
                 parsing_errors_counted_correct=int(((df.parse_error == 1) & (df.correct == 1)).sum()),
                 permutation_seed=20260830, permutations=analysis.N_PERM,
                 python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                 matplotlib=matplotlib.__version__,
                 generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 source_hashes={str(p.relative_to(ROOT)): h for p, h in hashes.items()})
    (OUT / "data" / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#334155",
                         "text.color": "#172B3A", "pdf.fonttype": 42})
    indexed = summary.set_index(KEYS)
    y = np.arange(len(CELLS))
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for i, model in enumerate(MODELS):
        values = [indexed.loc[(model, *c), "acuracia"] * 100 for c in CELLS]
        bars = ax.barh(y + (i - .5) * .35, values, .32, color=COLORS[i], label=NAMES[i])
        ax.bar_label(bars, labels=[f"{v:.1f}".replace(".", ",") for v in values], padding=4, fontsize=9)
    ax.set(yticks=y, yticklabels=LABELS, xlim=(0, 109), xlabel="Acurácia original (%)",
           title="Acurácia nas 18 condições do experimento")
    ax.invert_yaxis()
    fig.legend(loc="lower center", bbox_to_anchor=(.60, .05), ncol=2, frameon=False)
    save(fig, "01_acuracia", "Fonte: raw_results.csv · 320 execuções por condição (64 consultas × 5 repetições). Médias descritivas.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for i, (ax, model) in enumerate(zip(axes, MODELS)):
        t = tests[tests.model == model]
        ax.axvspan(-5, 5, color="#E2E8F0", alpha=.65)
        ax.axvline(0, color="#64748B", lw=1)
        for j, row in enumerate(t.itertuples()):
            ax.plot([0, row.delta_pp], [j, j], color=COLORS[i], lw=2)
            ax.scatter(row.delta_pp, j, s=75, edgecolors=COLORS[i],
                       facecolors=COLORS[i] if row.significativo_holm else "white", zorder=3)
            ax.text(row.delta_pp + (1.6 if row.delta_pp >= 0 else -1.6), j,
                    f"{row.delta_pp:+.2f}".replace(".", ","), va="center",
                    ha="left" if row.delta_pp >= 0 else "right", fontsize=9)
        ax.set(title=NAMES[i], yticks=np.arange(8), yticklabels=LABELS[1:],
               xlim=(-60, 32), xlabel="Diferença ante o baseline (p.p.)")
    axes[0].invert_yaxis()
    fig.suptitle("Efeito na acurácia: comparação pareada por consulta", fontweight="bold")
    save(fig, "02_efeitos", "Ponto cheio: significativo após Holm (16 testes, α = 5%). Vazio: não significativo. Faixa cinza: ±5 p.p.; não é IC.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for i, model in enumerate(MODELS):
        for ax, metric, limit in zip(axes, ["tokens_prompt_medio", "latencia_media_s"], [3700, 1.3]):
            values = [indexed.loc[(model, *c), metric] for c in CELLS]
            bars = ax.barh(y + (i - .5) * .35, values, .32, color=COLORS[i], label=NAMES[i])
            labels = [f"{v:.0f}" if metric == "tokens_prompt_medio" else f"{v:.2f}".replace(".", ",") for v in values]
            ax.bar_label(bars, labels=labels, padding=3, fontsize=8)
            ax.set(yticks=y, yticklabels=LABELS, xlim=(0, limit))
    axes[0].invert_yaxis()
    axes[0].set(title="Tokens de entrada por execução", xlabel="Média de tokens de entrada")
    axes[1].set(title="Latência HTTP das chamadas ao LLM", xlabel="Média em segundos")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.60, .05), ncol=2, frameon=False)
    save(fig, "03_custo", "Médias descritivas; two_stage inclui as duas chamadas. Exclui embeddings, busca local e overhead. Tokens não são custo monetário.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    categories = ["direct", "ambiguous", "distractor", "no_tool"]
    for ax, model, title in zip(axes, MODELS, NAMES):
        matrix = np.array([types.set_index(KEYS).loc[(model, *c), categories].to_numpy(dtype=float) * 100 for c in CELLS])
        ax.imshow(matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        for row in range(9):
            for col in range(4):
                ax.text(col, row, f"{matrix[row, col]:.1f}".replace(".", ","), ha="center", va="center",
                        color="white" if matrix[row, col] >= 65 else "#172B3A", fontsize=9)
        ax.set(title=title, xticks=range(4), xticklabels=["Direta", "Ambígua", "Distrator", "Sem ferramenta"],
               yticks=range(9), yticklabels=LABELS)
        ax.tick_params(axis="x", labelrotation=20)
    fig.suptitle("Acurácia original por tipo de consulta (%)", fontweight="bold")
    save(fig, "04_tipos_consulta", "80 execuções por célula (16 consultas × 5). Em no_tool, a métrica original pode contar falhas de parsing como acertos.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    modes = ["native", "json_prompt", "code_action"]
    for ax, model, title in zip(axes, MODELS, NAMES):
        for j, (metric, label, color) in enumerate([
            ("acuracia_original", "Original", "#156082"),
            ("acuracia_sem_falha_parsing", "Acerto sem falha de parsing", "#D46A33")]):
            values = [reliability.loc[(model, 50, "full", m), metric] * 100 for m in modes]
            bars = ax.bar(np.arange(3) + (j - .5) * .35, values, .32, color=color, label=label)
            ax.bar_label(bars, labels=[f"{v:.2f}".replace(".", ",") for v in values], padding=3, fontsize=9)
        ax.set(title=title, xticks=range(3), xticklabels=modes, ylim=(0, 112), ylabel="Acurácia (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.55, .05), ncol=2, frameon=False)
    fig.suptitle("Sensibilidade ao tratamento das falhas de parsing", fontweight="bold")
    save(fig, "05_sensibilidade", "Análise adicional: correct × (1 − parse_error). Dados e métrica primária preservados; não avalia os valores dos argumentos.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True)
    retrieval_modes = ["full", "random", "embedding", "hybrid", "two_stage"]
    for ax, model, title, color in zip(axes, MODELS, NAMES, COLORS):
        values = [availability.loc[(model, 50, r, "native"), "disponibilidade_gabarito"] * 100 for r in retrieval_modes]
        bars = ax.bar(range(5), values, color=color, width=.65)
        ax.bar_label(bars, labels=[f"{v:.2f}".replace(".", ",") for v in values], padding=3, fontsize=9)
        ax.set(title=title, xticks=range(5), xticklabels=retrieval_modes, ylim=(0, 112),
               ylabel="Consultas com gabarito disponível (%)")
        ax.tick_params(axis="x", labelrotation=20)
    fig.suptitle("Disponibilidade de ao menos uma ferramenta correta", fontweight="bold")
    save(fig, "06_recuperacao", "Apenas consultas que exigem ferramenta: 48 consultas × 5 repetições por condição. two_stage usa domínio, não top-5.")
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in hashes.items())
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    print(tests.to_string(index=False))
    print("SENSITIVITY\n", sensitivity[sensitivity.invocation != "native"].to_string(index=False))
    print("VS RANDOM\n", random_tests.to_string(index=False))


if __name__ == "__main__":
    main()
