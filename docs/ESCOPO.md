# Escopo fechado

Documento de referência do TCC. Descreve o que o trabalho é, o que ele não é, e
por quê. Substitui a versão anterior do escopo, que previa um eixo de
"representação" (`full`/`compact`/`name_only`) e um único modelo — ambos alterados
durante a implementação. As mudanças estão registradas na seção *Histórico*.

---

## Título

**A forma de expor ferramentas condiciona acurácia, alucinação e custo em agentes LLM**

Título alternativo ainda não sustentado por uma comparação direta entre técnicas:

**O mecanismo de invocação supera a recuperação na acurácia de agentes LLM**

O título principal permanece como referência. Os resultados contra o baseline
não demonstram, por si só, superioridade direta da invocação sobre a recuperação.

O título será fixado depois da análise do protocolo 3. O termo "alucinação" só
permanece se a família secundária (`hallucinated`) mostrar algum contraste
significativo após Holm.

---

## Pergunta de pesquisa

Como o tamanho do toolset exposto, a estratégia de recuperação de ferramentas e o
mecanismo de invocação afetam a acurácia de seleção, a taxa de alucinação e o
custo de operação de agentes baseados em LLM?

## Objetivo geral

Quantificar, sob condições controladas, o efeito de três fatores de exposição e
invocação de ferramentas sobre o desempenho de agentes LLM, em dois modelos de
código aberto servidos localmente.

## Objetivos específicos

1. Construir um instrumento experimental reprodutível, com corpus de ferramentas,
   dataset de queries com gabarito e execução retomável.
2. Medir o efeito do tamanho do toolset exposto (10, 30, 50 ferramentas).
3. Medir o efeito da estratégia de recuperação, ancorada em um piso aleatório.
4. Medir o efeito do mecanismo de invocação, incluindo a taxa de falha de parsing.
5. Separar falhas de recuperação de falhas de decisão do modelo.
6. Verificar se os efeitos observados se mantêm entre dois modelos distintos.

---

## Desenho experimental

**OFAT** (*one-factor-at-a-time*) em torno de um baseline.

- **Baseline:** `toolset=50`, `retrieval=full`, `invocation=native`
- **Eixo 1 — exposição:** 10 / 30 / 50 ferramentas
- **Eixo 2 — recuperação:** `full` / `random` / `embedding` / `hybrid` / `two_stage`; k=5 nos três métodos top-k
- **Eixo 3 — invocação:** `native` / `json_prompt` / `code_action`

18 condições (9 por modelo), 80 queries, 5 repetições → 7.200 execuções e 8.000
chamadas ao LLM (800 delas da 1ª etapa do `two_stage`).

### Modelos

| Papel | Modelo |
|---|---|
| LLM A | DeepSeek-V4-Flash-0731 |
| LLM B | gemma-4-E4B-it |
| Embeddings | qwen3-embedding-8B (4096 dim) |

Ambos servidos localmente por endpoints OpenAI-compatíveis. São medidos tokens
de entrada e latência HTTP; não há medição direta de tempo de GPU ou custo financeiro.

### Corpus e queries

- 52 ferramentas sintéticas, 4 domínios, 8 pares confusáveis declarados.
- Toolset montado **por query**: o gabarito e o `lure` estão sempre presentes; o
  restante é preenchido com distratores. Toolsets aninhados (10 ⊂ 30 ⊂ 50) e
  determinísticos, o que torna a comparação pareada de fato.
- 80 queries em pt-BR, 16 por tipo: `direct`, `ambiguous`, `distractor`, `no_tool`,
  `near_miss`. O tipo `near_miss` (protocolo 3) reúne pedidos do domínio que nenhuma
  ferramenta atende, cada um com um `lure` vizinho. O acerto é a abstenção; chamar
  qualquer ferramenta conta como alucinação.
- Ordem de apresentação (protocolo 3): a lista exposta ao modelo é embaralhada por
  consulta e repetição, com semente que não depende do modelo. A recuperação define
  quais ferramentas entram; a ordem não informa onde está o gabarito.
  `expected_position` registra a posição da primeira ferramenta do gabarito.

### Instrumentos de medida (não são técnicas candidatas)

- **`random`** — controle que expõe cinco ferramentas sorteadas. A disponibilidade
  de ao menos uma ferramenta correta foi de 10,42% nas consultas com ferramenta.
  Mantém o número exposto comparável, mas não isola um efeito puro de tamanho,
  pois também pode remover o gabarito. O sorteio é fixo por consulta.
- **`retrieval_hit`** — calculado em toda linha, não só numa condição dedicada.
  Indica se o gabarito sobreviveu à recuperação, separando erro de retrieval de
  erro de decisão do modelo. É um indicador observado, não uma intervenção oracle.

### Métricas

`correct`, `correct_valid_parse`, `hallucinated`, `invented_tool`, `lure_hit`,
`parse_error`, `retrieval_hit`, `prompt_tokens` (somando a 1ª etapa do `two_stage`),
`latency_s`. `expected_position` é usada só na análise descritiva de posição.

### Estatística documentada no projeto

#### Plano de análise do protocolo 3 (registrado antes da coleta)

Este plano foi escrito antes da coleta `results/real_results_v3.csv`. O commit que
o introduz serve como registro datado; não equivale a um pré-registro externo.

- **Hipótese primária.** Os 16 contrastes OFAT contra o baseline (8 variantes ×
  2 modelos) na métrica `correct`, em uma família Holm-Bonferroni a 5%. Variantes:
  10 e 30 ferramentas; `random`, `embedding`, `hybrid`, `two_stage`;
  `json_prompt`, `code_action`.
- **Sensibilidade.** Os mesmos 16 contrastes em `correct_valid_parse`, em família
  Holm separada. Verifica se as conclusões dependem de falhas de parsing contadas
  como acerto.
- **Hipótese secundária.** Os mesmos 16 contrastes em `hallucinated`, em família
  Holm separada (`analyze.py --metric hallucinated`).
- **Exploratório.** `embedding` e `hybrid` contra `random` (4 testes, família
  própria), como na coleta histórica.
- **Descritivo, sem teste.** Acurácia por faixa de `expected_position` (0–4, 5–14,
  15+) e comparação entre a coleta exploratória (protocolo 1) e o protocolo 3,
  restrita às 64 consultas comuns. As duas coletas diferem também na data; a
  comparação não isola o efeito da ordem.
- O limiar de 5 p.p. vale para as três famílias. Unidade de análise, teste,
  semente e número de permutações seguem os itens abaixo, sem alteração.

- Unidade de análise: a **query**. As 5 repetições viram a média por query antes
  do teste, para não inflar o N com observações não independentes.
- Teste de permutação pareada (troca de sinal), sem suposição de normalidade.
- Holm-Bonferroni a 5%.
- Efeito mínimo relevante: **5 pontos percentuais** de acurácia. Um resultado só é
  tratado como acionável se for significativo **e** atingir esse limiar.

---

## Fora do escopo

- Agentes multi-turno, memória, horizonte longo.
- Chamadas paralelas ou encadeadas de múltiplas ferramentas.
- Correção dos **valores dos argumentos** — mede-se a seleção da ferramenta, não o
  preenchimento dos parâmetros.
- Fine-tuning ou qualquer modificação de pesos.
- Modelos proprietários de fronteira.
- Software de produção, interface ou integração com sistemas reais.
- Segurança, prompt injection, execução efetiva das ferramentas.
- Variação de `k` no retrieval (fixo em 5).

## Limitações declaradas

1. **Corpus sintético** — validade externa limitada frente a APIs de produção.
2. **OFAT não estima interações** — o efeito de `embedding` é conhecido em
   toolset=50, e o efeito do tamanho apenas sob `full`. A leitura "X piora conforme
   o toolset cresce" não é sustentada por este desenho.
3. **Dois modelos servidos localmente** — os achados não se estendem automaticamente
   a outras famílias de modelos.
4. **Busca léxica do `hybrid`** é sobreposição de tokens, não BM25.
5. **`k=5` fixo.**
6. **Um único avaliador** no gabarito das queries (`reviewed_by` pendente). Uma
   segunda revisão independente aumentaria a credibilidade do instrumento.
7. **Consultas `near_miss` redigidas pelo autor** — 16 pedidos, um `lure` por
   pedido. Medem abstenção dentro do domínio, não alucinação em geral.

---

## Entregáveis

Scripts de experimento versionados, dataset de queries com gabarito,
`results/real_results_v3.csv` como dado primário versionado,
`results/raw_results.csv` preservado sem alteração como coleta exploratória
(protocolo 1), scripts de análise, figuras e o texto do TCC.

**Critério de sucesso:** o experimento roda de ponta a ponta de forma reprodutível
e produz comparação estatisticamente fundamentada entre as 18 condições, com
separação entre falha de recuperação e falha de decisão.

---

## Histórico de mudanças de escopo

| Data | Mudança | Justificativa |
|---|---|---|
| 2026-08-30 | Corpus fixado como sintético | Sem questão de confidencialidade; validade externa vira limitação declarada |
| 2026-08-30 | Dois modelos passam a compor dimensão comparativa (antes: um modelo, comparação entre modelos fora do escopo) | Servidores locais tornam o custo marginal desprezível; permite verificar se os efeitos se sustentam entre modelos |
| 2026-08-30 | Eixo "representação" (`full`/`compact`/`name_only`) substituído por eixo "recuperação" (`full`/`embedding`/`hybrid`/`two_stage`) | Recuperação responde mais diretamente à pergunta de pesquisa; representação fica para trabalhos futuros |
| 2026-09-07 | Braço `random` adicionado ao eixo de recuperação | Sem piso, o ganho de `embedding`/`hybrid` sobre `full` é ambíguo entre "menos ruído" e "mais relevância" |
| 2026-09-07 | Protocolo 2 do executor e validação da análise | Parsing validado, dados novos separados por origem e métrica de sensibilidade explícita; coleta histórica preservada |
| 2026-09-26 | Protocolo 3: ordem das ferramentas sorteada por consulta e repetição, coluna `expected_position`, 16 consultas `near_miss`, plano de análise registrado antes da coleta e nova coleta completa em `real_results_v3.csv` | Na coleta histórica o gabarito ocupava a primeira posição da lista em `full`, `two_stage` e nos três modos de invocação, e a ordem de `embedding`/`hybrid` seguia o ranking; a alucinação dependia só de consultas `no_tool` triviais. `raw_results.csv` passa a ser a coleta exploratória (protocolo 1) e não é alterado |

### Pendências antes de fechar a redação

- [x] Incorporar as 640 execuções de `random`; a base histórica contém 5.760 execuções válidas.
- [x] Recalcular as 18 condições e apresentar os contrastes exploratórios contra
      `random`, com família Holm separada em `docs/data/`.
- [x] Documentar a sensibilidade às falhas de parsing, mantendo `correct` original.
- [x] Proteger novas coletas por backend, protocolo e manifesto de origem.
- [ ] Validar com o orientador a inclusão dos dois modelos como dimensão.
- [ ] Revisar as 16 consultas `near_miss` (rascunho em `src/queries.py`).
- [ ] Preencher `reviewed_by` no dataset de queries (segundo avaliador), antes da
      coleta do protocolo 3. Mudar o texto de uma consulta depois exige nova coleta.
- [ ] Registrar versão do vLLM, argumentos de inicialização (`--tool-call-parser`,
      chat template) e hash dos pesos na tabela de `REVISAO_TECNICA.md`.
- [ ] Piloto do protocolo 3 (`--repetitions 1`) e coleta completa.
- [ ] Fixar o título depois dos resultados finais (regra em *Título*).
