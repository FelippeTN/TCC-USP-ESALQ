# Escopo fechado

Documento de referência do TCC. Descreve o que o trabalho é, o que ele não é, e
por quê. Substitui a versão anterior do escopo, que previa um eixo de
"representação" (`full`/`compact`/`name_only`) e um único modelo — ambos alterados
durante a implementação. As mudanças estão registradas na seção *Histórico*.

---

## Título

**A forma de expor ferramentas condiciona acurácia, alucinação e custo em agentes LLM**

Título alternativo, a adotar se o achado de invocação se confirmar como o principal:

**O mecanismo de invocação supera a recuperação na acurácia de agentes LLM**

A escolha entre os dois depende dos resultados finais. O primeiro é neutro quanto
à direção; o segundo compromete com um achado e só pode ser adotado depois da
rodada completa com o braço `random`.

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
- **Eixo 2 — recuperação:** `full` / `random` / `embedding` / `hybrid` / `two_stage`, k=5
- **Eixo 3 — invocação:** `native` / `json_prompt` / `code_action`

18 condições (9 por modelo), 64 queries, 5 repetições → 6.400 chamadas ao LLM.

### Modelos

| Papel | Modelo |
|---|---|
| LLM A | DeepSeek-V4-Flash-0731 |
| LLM B | gemma-4-E4B-it |
| Embeddings | qwen3-embedding-8B (4096 dim) |

Ambos self-hosted, endpoints OpenAI-compatíveis. Custo é tempo de GPU, não token.

### Corpus e queries

- 52 ferramentas sintéticas, 4 domínios, 8 pares confusáveis declarados.
- Toolset montado **por query**: o gabarito e o `lure` estão sempre presentes; o
  restante é preenchido com distratores. Toolsets aninhados (10 ⊂ 30 ⊂ 50) e
  determinísticos, o que torna a comparação pareada de fato.
- 64 queries em pt-BR, 16 por tipo: `direct`, `ambiguous`, `distractor`, `no_tool`.

### Instrumentos de medida (não são técnicas candidatas)

- **`random`** — piso. Expõe 5 ferramentas sorteadas. Isola o ganho que vem apenas
  de encurtar o prompt do ganho que vem da relevância da recuperação. Recall@5
  medido: 0,10, coerente com 5/50 por acaso.
- **`retrieval_hit`** — calculado em toda linha, não só numa condição dedicada.
  Indica se o gabarito sobreviveu à recuperação, separando erro de retrieval de
  erro de decisão do modelo. Cumpre o papel que um braço `oracle` cumpriria, sem
  gastar uma condição inteira do plano.

### Métricas

`correct`, `hallucinated`, `invented_tool`, `lure_hit`, `parse_error`,
`retrieval_hit`, `prompt_tokens` (somando a 1ª etapa do `two_stage`), `latency_s`.

### Estatística (pré-registrada)

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
3. **Dois modelos self-hosted de porte semelhante** — os achados não se estendem a
   modelos de fronteira.
4. **Busca léxica do `hybrid`** é sobreposição de tokens, não BM25.
5. **`k=5` fixo.**
6. **Um único avaliador** no gabarito das queries (`reviewed_by` pendente). Uma
   segunda revisão independente aumentaria a credibilidade do instrumento.

---

## Entregáveis

Scripts de experimento versionados, dataset de queries com gabarito,
`results/raw_results.csv` como dado primário versionado, scripts de análise,
figuras e o texto do TCC.

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

### Pendências antes de fechar a redação

- [ ] Rodar as 640 linhas novas do braço `random` (o resume pula as 5.120 já feitas).
- [ ] Reexecutar `analyze.py` e reavaliar se os achados de `embedding`/`hybrid`
      sobrevivem à comparação contra o piso.
- [ ] Validar com o orientador a inclusão dos dois modelos como dimensão.
- [ ] Preencher `reviewed_by` no dataset de queries (segundo avaliador).
- [ ] Fixar o título depois dos resultados finais.
