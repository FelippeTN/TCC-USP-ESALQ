"""Gera figuras e tabelas do README sem executar ou modificar o experimento.

Uso, na raiz:
    py scripts/generate_results.py                                  # coleta histórica → docs/
    py scripts/generate_results.py --input results/real_results_v3.csv --out docs
Dependências: as de requirements.txt.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import analyze as analysis
from corpus import TOOLS_BY_NAME
from queries import ABSTENTION_TYPES, QUERIES, QUERY_TYPES
from run_experiment import build_conditions, run_id

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
TYPE_LABELS = {"direct": "Direta", "ambiguous": "Ambígua", "distractor": "Distrator",
               "no_tool": "Sem ferramenta", "near_miss": "Fora do catálogo"}
POSITION_BINS = [-1, 4, 14, np.inf]
POSITION_LABELS = ["0-4", "5-14", "15+"]
TOOL_NAME_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, TOOLS_BY_NAME), key=len, reverse=True)) + r")\b")


def cell(df, model, condition, metric="correct"):
    return analysis._cell(df, model, dict(zip(KEYS[1:], condition)), metric)


def paired_tests(df, variants, baseline=CELLS[0], metric="correct"):
    """Uma família Holm por chamada; mesmas queries e teste do analyze.py."""
    analysis.RNG = np.random.default_rng(20260830)
    n_queries = df.query_id.nunique()
    rows = []
    for model in MODELS:
        base = cell(df, model, baseline, metric)
        for condition in variants:
            pair = pd.concat([cell(df, model, condition, metric), base], axis=1).dropna()
            assert len(pair) == n_queries, "Pareamento incompleto"
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


def check_coverage(df):
    query_ids = sorted(df.query_id.unique())
    unknown = set(query_ids) - {q["id"] for q in QUERIES}
    assert not unknown, f"Consultas fora do dataset atual: {sorted(unknown)}"
    assert set(df.query_type) <= set(QUERY_TYPES), "Tipo de consulta desconhecido"
    reps = sorted(int(r) for r in df.repetition.unique())
    assert reps == list(range(1, len(reps) + 1)), f"Repetições não contíguas: {reps}"
    expected = {run_id(c, qid, r) for c in build_conditions(MODELS) for qid in query_ids for r in reps}
    assert not df.run_id.duplicated().any(), "Há execuções válidas duplicadas"
    assert set(df.run_id) == expected, "Plano incompleto ou condições inesperadas"
    assert len(df) == len(expected) and df.groupby(KEYS).size().eq(len(query_ids) * len(reps)).all()
    assert df.groupby(KEYS + ["query_id"]).size().eq(len(reps)).all()
    return query_ids, reps


def position_effect(df):
    if "expected_position" not in df:
        return None
    sub = df[df.expected != ""].copy()
    pos = pd.to_numeric(sub.expected_position, errors="coerce")
    assert pos.notna().astype(int).eq(sub.retrieval_hit).all(), "expected_position diverge de retrieval_hit"
    assert (pos.dropna() < sub.n_exposed[pos.notna()]).all(), "expected_position fora da lista exposta"
    band = pd.cut(pos, POSITION_BINS, labels=POSITION_LABELS).astype(object).where(pos.notna(), "nao_exposto")
    sub["faixa_posicao"] = pd.Categorical(band, categories=POSITION_LABELS + ["nao_exposto"], ordered=True)
    return (sub.groupby(KEYS + ["faixa_posicao"], observed=True)
            .agg(n=("correct", "size"), consultas=("query_id", "nunique"), acuracia=("correct", "mean"))
            .round(4).reset_index())


def native_text_calls(df):
    if "response_message" not in df:
        return None
    columns = KEYS + ["run_id", "query_id", "query_type", "expected", "predicted",
                      "correct", "parse_error", "ferramentas_citadas", "content"]
    rows = []
    for row in df[df.invocation == "native"].to_dict("records"):
        message = json.loads(row["response_message"]) if row["response_message"] else {}
        if message.get("tool_calls"):
            continue
        content = message.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        names = sorted(set(TOOL_NAME_RE.findall(content)))
        if names:
            rows.append({**{k: row[k] for k in columns if k in row},
                         "ferramentas_citadas": "|".join(names), "content": content})
    return pd.DataFrame(rows, columns=columns)


def save(fig, out, name, note):
    fig.text(0.01, 0.01, note, fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, 0.13 if fig.legends else 0.055, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(out / "figures" / f"{name}.{ext}", dpi=220, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=analysis.RAW_CSV, help="CSV da coleta (padrão: raw_results.csv)")
    ap.add_argument("--out", type=Path, default=ROOT / "docs", help="pasta que recebe data/ e figures/ (padrão: docs/)")
    args = ap.parse_args()
    source, out = args.input.resolve(), args.out.resolve()

    def rel(p):
        return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.as_posix()

    for folder in ("figures", "data"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    protected = {p.resolve() for p in [*(ROOT / "src").glob("*.py"), *(ROOT / "results").glob("*.csv"), source]}
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(protected)}
    raw = pd.read_csv(source, keep_default_na=False)
    df = analysis.load(source)
    query_ids, reps = check_coverage(df)
    n_queries, n_reps = len(query_ids), len(reps)
    per_type = df.drop_duplicates("query_id").query_type.value_counts()
    categories = [t for t in QUERY_TYPES if t in per_type.index]
    with_tool = int(df.loc[df.expected != "", "query_id"].nunique())
    # Sensibilidade separada: nenhuma falha de parsing conta como acerto.
    df["correct_valid_parse"] = df.correct * (1 - df.parse_error)
    summary = analysis.condition_summary(df)
    types = analysis.by_query_type(df)
    tests = paired_tests(df, CELLS[1:])
    random_tests = paired_tests(df, CELLS[4:6], baseline=CELLS[3])
    sensitivity = paired_tests(df, CELLS[1:], metric="correct_valid_parse")
    hallucination = paired_tests(df, CELLS[1:], metric="hallucinated")
    reliability = df.groupby(KEYS).agg(acuracia_original=("correct", "mean"),
                                      acuracia_sem_falha_parsing=("correct_valid_parse", "mean"),
                                      erro_parsing=("parse_error", "mean"))
    availability = df[df.expected != ""].groupby(KEYS).agg(
        n=("retrieval_hit", "size"), disponibilidade_gabarito=("retrieval_hit", "mean"))
    positions, text_calls = position_effect(df), native_text_calls(df)
    tables = [("summary_by_condition", summary), ("accuracy_by_query_type", types),
              ("ofat_tests_correct", tests), ("exploratory_vs_random", random_tests),
              ("sensitivity_tests", sensitivity), ("sensitivity_summary", reliability.reset_index()),
              ("retrieval_with_tool", availability.reset_index()),
              ("ofat_tests_hallucinated", hallucination),
              ("position_effect", positions), ("native_text_calls", text_calls)]
    for name, table in tables:
        if table is None:
            print(f"{name}.csv não gerado: coluna ausente em {source.name}")
            continue
        table.to_csv(out / "data" / f"{name}.csv", index=False)
    protocol = sorted(set(df.protocol_version)) if "protocol_version" in df else []
    audit = dict(input=rel(source), raw_sha256=hashes[source], protocol_version=int(protocol[0]) if protocol else None,
                 rows_raw=len(raw), rows_valid=len(df),
                 rows_error=int(raw.error.ne("").sum()), valid_duplicates=int(df.run_id.duplicated().sum()),
                 queries=n_queries, queries_by_type={t: int(per_type[t]) for t in categories},
                 conditions=len(summary), repetitions=n_reps,
                 timestamp_min=df.timestamp.min(), timestamp_max=df.timestamp.max(),
                 parsing_errors=int(df.parse_error.sum()),
                 parsing_errors_counted_correct=int(((df.parse_error == 1) & (df.correct == 1)).sum()),
                 native_text_calls=None if text_calls is None else len(text_calls),
                 permutation_seed=20260830, permutations=analysis.N_PERM,
                 python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                 matplotlib=matplotlib.__version__,
                 generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 source_hashes={rel(p): h for p, h in hashes.items()})
    (out / "data" / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
           title=f"Acurácia nas {len(summary)} condições do experimento")
    ax.invert_yaxis()
    fig.legend(loc="lower center", bbox_to_anchor=(.60, .05), ncol=2, frameon=False)
    save(fig, out, "01_acuracia", f"Fonte: {source.name} · {n_queries * n_reps} execuções por condição "
         f"({n_queries} consultas × {n_reps} repetições). Médias descritivas.")

    lo, hi = min(-60, tests.delta_pp.min() - 10), max(32, tests.delta_pp.max() + 10)
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
        ax.set(title=NAMES[i], yticks=np.arange(len(CELLS) - 1), yticklabels=LABELS[1:],
               xlim=(lo, hi), xlabel="Diferença ante o baseline (p.p.)")
    axes[0].invert_yaxis()
    fig.suptitle("Efeito na acurácia: comparação pareada por consulta", fontweight="bold")
    save(fig, out, "02_efeitos", f"Ponto cheio: significativo após Holm ({len(tests)} testes, α = 5%). "
         "Vazio: não significativo. Faixa cinza: ±5 p.p.; não é IC.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    limits = [max(3700, indexed.tokens_prompt_medio.max() * 1.15), max(1.3, indexed.latencia_media_s.max() * 1.15)]
    for i, model in enumerate(MODELS):
        for ax, metric, limit in zip(axes, ["tokens_prompt_medio", "latencia_media_s"], limits):
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
    save(fig, out, "03_custo", "Médias descritivas; two_stage inclui as duas chamadas. Exclui embeddings, busca local e overhead. Tokens não são custo monetário.")

    fig, axes = plt.subplots(1, 2, figsize=(11 if len(categories) <= 4 else 12.5, 6), sharey=True)
    for ax, model, title in zip(axes, MODELS, NAMES):
        matrix = np.array([types.set_index(KEYS).loc[(model, *c), categories].to_numpy(dtype=float) * 100 for c in CELLS])
        ax.imshow(matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, f"{matrix[row, col]:.1f}".replace(".", ","), ha="center", va="center",
                        color="white" if matrix[row, col] >= 65 else "#172B3A", fontsize=9)
        ax.set(title=title, xticks=range(len(categories)), xticklabels=[TYPE_LABELS.get(c, c) for c in categories],
               yticks=range(len(CELLS)), yticklabels=LABELS)
        ax.tick_params(axis="x", labelrotation=20)
    fig.suptitle("Acurácia original por tipo de consulta (%)", fontweight="bold")
    per_cell = (f"{int(per_type.iloc[0]) * n_reps} execuções por célula ({int(per_type.iloc[0])} consultas × {n_reps})"
                if per_type.nunique() == 1 else "Execuções por célula variam com o tipo")
    abstention = " e ".join(t for t in ABSTENTION_TYPES if t in categories)
    save(fig, out, "04_tipos_consulta", f"{per_cell}. Em {abstention}, a métrica original pode contar falhas de parsing como acertos.")

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
    save(fig, out, "05_sensibilidade", "Análise adicional: correct × (1 − parse_error). Dados e métrica primária preservados; não avalia os valores dos argumentos.")

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
    save(fig, out, "06_recuperacao", f"Apenas consultas que exigem ferramenta: {with_tool} consultas × {n_reps} repetições "
         "por condição. two_stage usa domínio, não top-5.")
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in hashes.items())
    print(json.dumps(audit, ensure_ascii=True, indent=2))
    print(tests.to_string(index=False))
    print("SENSITIVITY\n", sensitivity[sensitivity.invocation != "native"].to_string(index=False))
    print("VS RANDOM\n", random_tests.to_string(index=False))
    print("HALLUCINATED\n", hallucination.to_string(index=False))


if __name__ == "__main__":
    main()
