# TCC-USP-ESALQ

**A forma de expor ferramentas condiciona acurácia, alucinação e custo em agentes LLM**

Benchmark controlado sobre três eixos de exposição de ferramentas, com dois LLMs
servidos localmente. Este README registra as decisões de desenho — o que ficou
fixo, o que varia, e por quê.

## Decisões de escopo (fechadas em 2026-08-30)

| Questão | Decisão |
|---|---|
| Modelos | **Os dois entram como dimensão comparativa**: DeepSeek-V4-Flash-0731 e gemma-4-E4B-it |
| Corpus | **Sintético** — 52 ferramentas, 4 domínios, 8 pares confusáveis. Sem questão de confidencialidade; validade externa é limitação declarada |
| Eixos | **Os 3 completos**, incluindo `code_action` |
| Queries | **64**, balanceadas 16 por tipo |
| Repetições | 5 |
| Significância | Holm-Bonferroni a 5%, teste de permutação pareada |
| Efeito mínimo relevante | 5 pontos percentuais de acurácia |

## Infraestrutura

Endpoints OpenAI-compatíveis, verificados via `GET /v1/models`:

| Serviço | Porta | Model id |
|---|---|---|
| LLM A | `8000` | `DeepSeek-V4-Flash-0731` |
| LLM B | `8001` | `/srv/models/gemma-4-E4B-it` |
| Embeddings | `8002` | `/srv/models/qwen3-embedding-8B` (4096 dim) |

O host fica em `TCC_SERVER_HOST`, no `.env` (fora do git — este repositório é
público). Copie `.env.example` para `.env` antes de rodar:

```bash
cp .env.example .env    # e preencha o endereço do servidor
```

`config.py` lê o `.env` e **falha na hora** se a variável não existir, em vez de cair
num default silencioso.

> Os dois LLMs suportam *native tool calling* (campo `tools` + `tool_calls`), o que
> viabiliza o eixo 3 sem gambiarra.

**Custo:** os servidores são locais, então o custo é tempo de GPU, não token.
Medido no piloto: **0,67 s por linha**; o plano completo (5.120 linhas / 5.760
chamadas) roda em **~1 h em série**. Não há orçamento monetário a estourar, o que
torna as 5 repetições e os 3 eixos completos viáveis sem corte.

## Desenho experimental (OFAT)

Baseline: `toolset=50, retrieval=full, invocation=native`. Cada eixo varia sozinho
a partir dele, então toda comparação difere do baseline em exatamente um fator.

- **Eixo 1 — exposição:** toolset de 10 / 30 / 50 ferramentas
- **Eixo 2 — recuperação:** `full` / `embedding` / `hybrid` (RRF) / `two_stage`, k=5
- **Eixo 3 — invocação:** `native` / `json_prompt` / `code_action`

16 condições (8 por modelo; o baseline roda uma vez e é comparado nos três eixos).

### Duas decisões de desenho que não são óbvias

**1. O toolset é montado por query.** O gabarito da query (mais o `lure` e o par
confusável correspondente) está sempre presente, em qualquer tamanho; o resto é
preenchido com distratores. Sem isso, num toolset de 10 sorteado de um pool de 52
a ferramenta correta quase nunca estaria lá, e o eixo de exposição mediria
*ausência*, não efeito do tamanho. Os toolsets são aninhados (10 ⊂ 30 ⊂ 50) e
determinísticos por query, então a comparação é pareada de verdade.

**2. A unidade de análise é a query, não a chamada.** As 5 repetições viram a média
por query antes do teste. Tratar repetições da mesma query como observações
independentes infla o N e o poder estatístico artificialmente.

## Dataset de queries

64 queries em pt-BR, 16 por tipo:

- `direct` — uma ferramenta claramente correta
- `ambiguous` — cabe em mais de uma ferramenta de um par confusável; acertar qualquer uma conta
- `distractor` — a correta existe, mas o texto evoca lexicalmente outra (o `lure`)
- `no_tool` — nenhuma ferramenta se aplica; chamar qualquer uma é alucinação

## Métricas

| Métrica | Definição |
|---|---|
| `correct` | ferramenta prevista ∈ gabarito (ou nenhuma chamada, em `no_tool`) |
| `hallucinated` | chamou ferramenta quando não devia, **ou** chamou nome fora do toolset exposto |
| `invented_tool` | nome previsto não existe no toolset exposto |
| `lure_hit` | caiu exatamente no distrator lexical |
| `parse_error` | resposta não pôde ser lida no formato pedido |
| `retrieval_hit` | o gabarito sobreviveu à etapa de recuperação — separa erro de retrieval de erro do modelo |
| custo | `prompt_tokens` e latência, somando a 1ª etapa do `two_stage` |

## Como rodar

```bash
py run_experiment.py --plan-only                  # plano + estimativa, não chama nada
py run_experiment.py --backend mock               # pipeline inteiro sem tocar na GPU
py run_experiment.py --backend real --repetitions 1 --limit 12 --out results/pilot.csv
py run_experiment.py --backend real               # plano completo, retoma de onde parou
py analyze.py                                     # métricas + testes OFAT
```

Cada módulo tem autoteste embutido: `py corpus.py`, `py queries.py`, `py client.py`,
`py invocation.py`, `py retrieval.py`.

### Docker (Python 3.14)

```bash
docker compose build
docker compose run --rm tcc            # abre o bash dentro do container, em /app
```

Dentro do container os comandos são os mesmos, com `python` no lugar de `py`:

```bash
python corpus.py                       # autotestes
python run_experiment.py --plan-only
python run_experiment.py --backend real
python analyze.py
```

O projeto inteiro é volume (`.:/app`): editar no host reflete na hora, e `results/`
fica no host — o dado primário não morre junto com o container.

> **Um executor por vez.** Host e container escrevem no mesmo `results/raw_results.csv`.
> O resume evita repetir trabalho, mas dois processos anexando ao mesmo CSV
> intercalam linhas e corrompem o dado primário. Use `--out` separado se precisar
> rodar os dois.

O runner grava linha a linha com `flush` e retoma pelo `run_id`: queda, timeout ou
rate limit não perdem o lote nem duplicam trabalho. Linhas com erro são regravadas
na próxima execução.

> **Cache de embeddings é separado por backend.** Rodar `mock` e depois `real`
> compartilhando o mesmo arquivo envenena o índice com vetores falsos e a recuperação
> passa a errar em silêncio — foi o que aconteceu no primeiro piloto (recall@5 caiu
> para 0,12). `EmbeddingIndex` agora estoura se as dimensões se misturarem.

## Dados

`results/raw_results.csv` é o **dado primário do TCC** e é versionado no git. Não
regenerar por cima sem necessidade: o runner já retoma sem duplicar.

## Limitações declaradas

- Corpus sintético: validade externa limitada frente a APIs reais de produção.
- `k=5` fixo no retrieval; variar `k` seria uma quarta dimensão, fica para trabalhos futuros.
- Busca léxica do `hybrid` é sobreposição de tokens, não BM25.
- Um único avaliador no gabarito das queries (campo `reviewed_by` está pendente) —
  segunda revisão independente aumentaria a credibilidade do instrumento.
