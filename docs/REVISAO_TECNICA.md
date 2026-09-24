# Revisão técnica e reprodução

Revisão de 07/09/2026. O protocolo 2 altera o tratamento de respostas inválidas em
novas execuções. Não altera os prompts, gabaritos, ferramentas, estratégias de
recuperação ou a definição histórica de `correct`. O CSV da coleta original e os
gráficos já publicados permanecem preservados.

## O que mudou

| Item | Coleta histórica | Protocolo 2 |
|---|---|---|
| Python com apenas comentário | Podia gerar `IndexError` e virar erro excluído da análise | Gera `parse_error=1`, mantendo a execução na análise |
| Vários comandos Python | O parser examinava somente a primeira instrução | Exige exatamente uma instrução AST: chamada simples ou `pass` |
| Chamadas nativas malformadas ou múltiplas | Podiam ser aceitas parcialmente ou causar exceção | São registradas como falha de parsing |
| Acerto | `correct` mede seleção/ausência de ferramenta | `correct` mantido e `correct_valid_parse = correct × (1 − parse_error)` registrado separadamente |
| Origem | CSV sem identificação de backend ou versão | Cada linha informa `backend` e `protocol_version` |
| Resposta para auditoria | Resposta bruta não armazenada | `response_message` armazena a mensagem da chamada final em JSON |
| Retomada | Identificadores sem verificação de origem | Exige mesmo backend, protocolo, código e ambiente local registrados |
| Saída padrão | Mesmo arquivo para simulação e coleta real | `mock_results_v2.csv` e `real_results_v2.csv` |

O parser continua tolerando cercas de código e, no modo JSON, texto antes do objeto,
conforme a implementação anterior. Ele não valida os valores dos argumentos nem
executa ferramentas. O campo `response_message` não contém a resposta intermediária
da classificação de domínio de `two_stage`.

## Proteção dos resultados existentes

O executor recusa escrever em `results/raw_results.csv` e em CSVs legados.
Para uma nova coleta, use um nome novo. Uma execução válida é identificada por
`run_id` dentro de um CSV cuja origem foi verificada; identificadores iguais em
arquivos de backends distintos não significam a mesma observação experimental.

O analisador continua lendo a coleta histórica. Ele rejeita duplicatas válidas,
colunas obrigatórias ausentes, valores não numéricos ou não finitos, indicadores
binários fora de 0/1, custos negativos e backends/protocolos misturados quando
essas colunas estão presentes. Os contrastes exigem as mesmas consultas e
repetições que o baseline. Pilotos balanceados continuam permitidos.

`correct` permanece a métrica padrão. Para a sensibilidade, use
`--metric correct_valid_parse`; os derivados têm nomes próprios. A correção dos
parsers não é aplicada retrospectivamente: as respostas históricas não foram
armazenadas, portanto não se pode reproduzir sua interpretação com o parser novo.

O gerador pseudoaleatório dos testes OFAT é reiniciado em cada chamada de análise,
com a semente 20260830. Isso elimina dependência de autotestes ou análises executadas
antes no mesmo processo. Teste, número de permutações e correção Holm permanecem;
pequenas diferenças de p frente a saídas antigas podem refletir a sequência de
números aleatórios. A conferência usa as tabelas documentadas em `docs/data/`.

## Ambiente e manifesto

As versões em [requirements.txt](../requirements.txt) correspondem ao ambiente
local inspecionado nesta revisão. Não certificam retroativamente o ambiente dos
servidores ou da coleta de agosto. O Docker usa Python 3.14.0 e essas versões.
[requirements-figures.txt](../requirements-figures.txt) acrescenta o Matplotlib.
A imagem-base usa uma versão explícita, mas não um digest imutável de sistema.

Cada CSV novo recebe `<arquivo>.csv.meta.json` com versão do protocolo, backend,
Python, plataforma local, versões de NumPy/Pandas/Requests, hashes SHA-256 dos
módulos Python e identificadores de modelos configurados. O manifesto não registra
o endereço privado do servidor. Alterações no ambiente registrado ou nos módulos
exigem um novo arquivo; a retomada não mistura implementações.

As revisões dos pesos e o hardware remoto são registrados como desconhecidos.
Para cada coleta real, mantenha um registro complementar, com evidências do servidor:

| Informação | Como completar |
|---|---|
| CSV e seu manifesto | Caminho e hash do arquivo após encerrar a coleta |
| Backend real | Comando executado e log da execução |
| Pesos dos modelos e embeddings | Revisão/commit/hash efetivo dos artefatos no servidor |
| Servidor de inferência | Nome, versão e argumentos de inicialização |
| Hardware | GPUs, memória e demais recursos efetivamente usados |
| Configuração experimental | Temperatura, limites de tokens, datas e condições de carga |
| Revisão dos gabaritos | Identificação do segundo avaliador e data da conferência |

Não preencha os campos desconhecidos por inferência a partir do nome do endpoint.
Use um registro complementar, preservando o manifesto automático.

## Verificação local

```bash
py -m unittest discover -s tests -v
py corpus.py
py queries.py
py client.py
py invocation.py
py retrieval.py
docker compose config --quiet
```

Os testes de regressão usam arquivos temporários e backend simulado. Eles verificam
falhas de parsing, separação das métricas, rejeição de dados inválidos, retomada sem
duplicação e estabilidade das decisões estatísticas. Não validam a qualidade dos
modelos remotos nem substituem a revisão independente das consultas.

## Pendências acadêmicas

As 64 consultas continuam com `reviewed_by=None`, até que um segundo avaliador
efetue a revisão. A identidade/revisão efetiva dos pesos e o hardware remoto também
dependem de registros do responsável pela infraestrutura. Essas pendências estão
explícitas no escopo e não foram preenchidas automaticamente.
