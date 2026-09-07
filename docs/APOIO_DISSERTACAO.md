# Apoio à redação: resultados e discussão

Texto-base elaborado a partir do recorte de `results/raw_results.csv` identificado
em [audit.json](data/audit.json). Os números se referem a 5.760 execuções válidas,
18 condições, 64 consultas e cinco repetições. A fonte de cada quadro é a análise
local dos dados do próprio estudo. Este material apoia a redação empírica; a revisão
de literatura e a formatação institucional devem ser integradas pelo autor.

## Proposta de organização do capítulo

| Seção | Conteúdo | Evidência |
|---|---|---|
| Delineamento e integridade | OFAT, baseline, cobertura e unidade de análise | Auditoria e metodologia do README |
| Desempenho global | Acurácia por configuração | Figura 1 e resumo por condição |
| Comparações planejadas | Efeito pareado e decisão após Holm | Figura 2 e tabela dos 16 testes |
| Papel da recuperação | Disponibilidade do gabarito e controle aleatório | Figura 3 e contrastes exploratórios |
| Eficiência observada | Tokens de entrada e latência HTTP | Figura 4 |
| Decomposição dos erros | Ambiguidade, ausência de ferramenta e alucinação | Figura 5 |
| Robustez da mensuração | Sensibilidade às falhas de parsing | Figura 6 |
| Discussão e limites | Interpretação por modelo e validade externa | Limitações e texto abaixo |

## Texto-base para Resultados

O experimento compreendeu 5.760 execuções válidas distribuídas em 18 condições,
com 320 execuções por condição. Foram utilizadas 64 consultas em português,
balanceadas em quatro categorias, com cinco repetições por consulta. A cobertura
do plano foi integral, sem duplicação de execuções válidas no recorte analisado.
Para a inferência estatística, as repetições foram agregadas por consulta, de modo
que cada comparação utilizou 64 pares, e não 320 observações independentes.

O baseline, definido por 50 ferramentas expostas integralmente e invocação nativa,
apresentou acurácia de 77,50% no DeepSeek e 71,56% no Gemma. No DeepSeek, a invocação
por `json_prompt` alcançou 95,00%, enquanto `code_action` atingiu 98,44%. Os aumentos
foram de 17,50 e 20,94 pontos percentuais, respectivamente, com valores de p ajustados
por Holm de 0,0039 e 0,0016. Ambos excederam o limiar de relevância prática de cinco
pontos percentuais. No Gemma, `code_action` alcançou 82,19% pela métrica original,
com aumento de 10,62 pontos percentuais e p ajustado de 0,0228.

As estratégias `embedding` e `hybrid` apresentaram acurácias superiores às do
baseline em termos descritivos, mas os contrastes não foram significativos após
Holm-Bonferroni na família de 16 comparações. Para `embedding`, os aumentos foram
de 9,38 pontos percentuais no DeepSeek e 5,00 no Gemma. Para `hybrid`, foram de
10,63 e 6,56 pontos percentuais. A ausência de significância não demonstra igualdade
entre as configurações. Já `two_stage` reduziu a acurácia do Gemma em 13,75 pontos
percentuais, com p ajustado de 0,0451, resultado próximo ao limiar adotado.

Entre as consultas que exigiam ferramenta, a recuperação por embeddings manteve
ao menos uma alternativa correta em todas as execuções, enquanto `hybrid` obteve
95,83% e o controle aleatório, 10,42%. Em quatro comparações exploratórias corrigidas
por Holm em família separada, `embedding` e `hybrid` superaram o controle aleatório
nos dois modelos, com p ajustado de aproximadamente 0,0004. Os ganhos de acurácia
variaram de 48,44 a 57,81 pontos percentuais. A comparação mantém cinco ferramentas
expostas, mas o controle aleatório foi executado em data distinta e com sorteio
fixo por consulta, fatores que limitam uma interpretação causal abrangente.

Quanto aos recursos observados, `embedding` reduziu a média de tokens de entrada
em 82,04% no DeepSeek e 87,64% no Gemma, em relação ao baseline de cada modelo.
Entretanto, a latência não acompanhou uniformemente a redução de tokens. Com
`code_action`, a latência média do DeepSeek passou de 1,0003 para 0,2105 segundo,
enquanto a do Gemma passou de 0,7756 para 1,0223 segundo. Esses valores descrevem
as chamadas HTTP de chat e não incluem geração de embeddings, busca local ou
medição direta do uso de GPU.

## Texto-base para a análise de sensibilidade

A inspeção da definição operacional de acerto mostrou que, nas consultas sem
ferramenta aplicável, a ausência de chamada era contabilizada como acerto mesmo
quando decorria de falha de parsing. Foram encontradas 117 falhas de parsing no
Gemma, das quais 60 também receberam `correct=1`: 34 em `code_action` e 26 em
`json_prompt`. Para avaliar a dependência dos resultados em relação a essa
definição, foi realizada análise de sensibilidade com a métrica
`correct × (1 − parse_error)`, mantendo-se inalterados os registros e a métrica
primária do experimento.

Sob essa exigência adicional, a acurácia do Gemma em `code_action` passou de 82,19%
para 71,56%, igual à média do baseline, com diferença de 0,00 ponto percentual
e p ajustado de 1,0000. Em `json_prompt`, passou de 72,19% para 64,06%, sem
significância após Holm. Os resultados do DeepSeek permaneceram inalterados.
Assim, a superioridade de `code_action` no Gemma pela métrica original não se
manteve quando o critério passou a exigir também uma resposta sem falha de parsing.
Essa análise não examina a correção dos valores dos argumentos nem a execução
efetiva das ferramentas.

## Texto-base para Discussão

Os resultados indicam que a configuração de exposição e invocação afeta o
desempenho no instrumento avaliado, mas a direção e a interpretação dos efeitos
dependem do modelo e do critério de avaliação. No DeepSeek, a invocação textual
estruturada apresentou ganhos de acurácia sustentados pelos testes contra o
baseline e pela análise de sensibilidade. No Gemma, a distinção entre ausência
de chamada e resposta interpretável alterou a conclusão sobre `code_action`.
Portanto, a avaliação de agentes deve explicitar como trata falhas de formato
em consultas que requerem abstenção de uso de ferramentas.

A recuperação relevante mostrou utilidade em duas dimensões distintas: preservou
ferramentas corretas em conjuntos pequenos e reduziu os tokens de entrada das
chamadas ao LLM. A disponibilidade de uma alternativa correta, entretanto, não
garantiu a seleção correta pelo modelo, como mostram as acurácias inferiores a
100% mesmo com disponibilidade integral em `embedding`. A queda de disponibilidade
em `two_stage`, especialmente no Gemma, é compatível com perda de alternativas
corretas após a escolha de domínio. O desenho não permite atribuir toda a queda
de acurácia a essa etapa nem tratar disponibilidade como uma intervenção oracle.

As consultas ambíguas exibiram as menores acurácias no baseline, sugerindo que a
discriminação entre ferramentas semanticamente próximas constitui uma dificuldade
relevante neste corpus. Essa interpretação é descritiva: não foram realizados
testes adicionais por categoria. As cinco ocorrências de alucinação operacional,
concentradas no DeepSeek com `two_stage`, são insuficientes para afirmar uma
vantagem geral de uma configuração na prevenção de alucinações. Ademais, o indicador
não abrange toda forma de erro semântico ou de argumento.

## Proposta de conclusão

Neste benchmark sintético, os resultados sustentam que decisões de exposição e
invocação influenciam a seleção de ferramentas e os recursos observados das chamadas
ao LLM. No DeepSeek, `json_prompt` e `code_action` apresentaram ganhos de acurácia
em relação à invocação nativa que persistiram sob a análise de sensibilidade.
No Gemma, o ganho original de `code_action` mostrou-se dependente do tratamento
de falhas de parsing. A recuperação por embeddings e a estratégia híbrida reduziram
os tokens de entrada e superaram o controle aleatório, embora seus ganhos de
acurácia sobre a exposição integral não tenham sido conclusivos após a correção
das múltiplas comparações. Os achados recomendam avaliar conjuntamente seleção,
abstenção, validade da resposta e recursos, respeitando os limites do corpus,
dos modelos e do desenho OFAT.

## Afirmações a evitar

| Afirmação | Formulação sustentada pelos dados |
|---|---|
| “code_action é melhor para qualquer LLM” | O ganho robusto foi observado no DeepSeek; no Gemma depende da métrica. |
| “A recuperação não faz diferença” | Não houve ganho significativo de acurácia contra full após Holm; houve redução descritiva de tokens e vantagem ante random. |
| “Invocação supera recuperação” | Há ganhos de invocação contra o baseline no DeepSeek; não houve teste direto de superioridade contra hybrid. |
| “Reduzir tokens reduz o custo de GPU em 88%” | Houve redução de até 87,83% nos tokens de entrada com hybrid no Gemma; GPU e custo financeiro não foram medidos. |
| “São 5.760 amostras independentes” | São 5.760 execuções, com 64 consultas pareadas por contraste. |
| “Não houve alucinação” | Houve cinco ocorrências pela definição operacional; ausência de nome inventado não significa ausência de erro. |
| “Os testes provam a hipótese” | Os testes fornecem evidência dentro do instrumento e dos pressupostos descritos. |

## Figuras para inserir no texto

As legendas completas estão no [README](../README.md). Use a numeração abaixo;
o prefixo do nome do arquivo é apenas um identificador. Fonte sugerida:
“Elaboração própria com base nos dados do experimento (2026)”.

| Figura | Título | PDF vetorial | PNG |
|---|---|---|---|
| 1 | Acurácia original por condição e modelo | [PDF](figures/01_acuracia.pdf) | [PNG](figures/01_acuracia.png) |
| 2 | Diferenças pareadas em relação ao baseline | [PDF](figures/02_efeitos.pdf) | [PNG](figures/02_efeitos.png) |
| 3 | Disponibilidade de ferramenta correta | [PDF](figures/06_recuperacao.pdf) | [PNG](figures/06_recuperacao.png) |
| 4 | Tokens de entrada e latência HTTP | [PDF](figures/03_custo.pdf) | [PNG](figures/03_custo.png) |
| 5 | Acurácia por tipo de consulta | [PDF](figures/04_tipos_consulta.pdf) | [PNG](figures/04_tipos_consulta.png) |
| 6 | Sensibilidade ao tratamento das falhas de parsing | [PDF](figures/05_sensibilidade.pdf) | [PNG](figures/05_sensibilidade.png) |

## Antes de fechar a dissertação

- Registrar a revisão do código e dos pesos, configuração do servidor e hardware
  efetivamente usados, sem inferir esses detalhes apenas do nome do endpoint.
- Corroborar o backend das execuções com logs disponíveis; o CSV não contém esse campo.
- Documentar a data de inclusão de `random` e distinguir os contrastes originais
  das comparações exploratórias e da sensibilidade adicionadas nesta apresentação.
- Tratar o alegado pré-registro como decisão documentada no projeto; só afirmar
  registro público ou anterior à coleta se houver comprovação datada.
- Completar a revisão independente dos gabaritos e discutir a definição de acerto
  em `no_tool` com o orientador, sem substituir resultados seletivamente.
- Explicitar que as condições não foram intercaladas aleatoriamente, que `random`
  foi coletado depois e que o sorteio é fixo por consulta.
- Reconhecer que 64 consultas sintéticas e dois modelos não sustentam generalização
  para produção, outros idiomas, agentes multi-turno ou todas as famílias de LLMs.
- Avaliar intervalos de confiança pareados e replicações em novas consultas como
  complementos futuros; os gráficos atuais não contêm intervalos de confiança.
- Integrar referências acadêmicas verificadas à Discussão e aplicar o manual do
  curso na versão final; este texto não declara conformidade com uma norma específica.
