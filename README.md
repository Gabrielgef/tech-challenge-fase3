# Tech Challenge Fase 3: previsão do cumprimento da meta de alfabetização

Projeto da Pós Tech. Usa a camada Gold da Fase 2 (Indicador Criança Alfabetizada) para prever quais municípios cumprem a meta de alfabetização e identificar onde está o risco.

## Contexto do problema

A alfabetização no 2º ano do Ensino Fundamental é um dos principais indicadores educacionais do país, e cada município tem uma meta anual. Gestores precisam saber com antecedência quais municípios podem ficar abaixo da meta e o que pesa nisso. O projeto transforma os dados públicos da Fase 2 em um modelo preditivo e em análises que apoiem essa decisão.

## Objetivo analítico

Prever se um município atinge a meta de alfabetização de 2024 usando apenas informações de 2023 (classificação binária) e usar o modelo para responder às perguntas de negócio do enunciado. O enunciado fala em prever se um aluno é alfabetizado, mas o indicador só existe agregado, então a unidade de análise é o município.

## Base utilizada

Seis tabelas da camada Gold (três fatos e três dimensões) e dados do IBGE (população e PIB municipal, via API SIDRA). A união está em `src/preprocessing/build_dataset.py`, que gera `data/processed/abt_alfabetizacao_municipios.csv`: um município por linha, rede municipal, 5.232 municípios.

* Variáveis de 2023 (ano-base): taxa de alfabetização, média de português, participação na avaliação.
* IBGE: população e PIB per capita (PIB total dividido pela população).
* Target: 1 se a taxa de 2024 ficou igual ou acima da meta de 2024.

Os achados sobre a qualidade da Gold (metas só na rede municipal, sem distribuição por níveis em 2023, tabela de cobertura zerada e sem SP) estão em `notebooks/01_eda_hipoteses.ipynb`.

## Estrutura do repositório

```
data/            tabelas Gold (data/gold), cache do IBGE e ABT (data/processed)
notebooks/       01_eda_hipoteses, 02_modelagem, 03_aplicacao_estrategica
src/
  preprocessing/ build_dataset.py (ABT) e features.py (variáveis do modelo)
  modeling/      pipeline.py (pré-processamento + modelos + busca de hiperparâmetros)
  evaluation/    metrics.py (validação cruzada e métricas de teste)
  visualization/ plots.py
reports/         modelo treinado, métricas, importâncias e listas de risco
images/          gráficos gerados pelos notebooks
```

## Como reproduzir

1. `pip install -r requirements.txt`
2. Copiar as 6 tabelas Gold para `data/gold/`.
3. `python src/preprocessing/build_dataset.py` (a primeira execução consulta o IBGE e guarda o cache em `data/external/`).
4. Executar os notebooks 01, 02 e 03, nessa ordem.

Versões usadas nos testes: Python 3.12, pandas 3.0, numpy 2.4, scikit-learn 1.8, scipy 1.17, matplotlib 3.10, seaborn 0.13. As sementes aleatórias são fixas (42).

## Etapas de modelagem

1. **EDA e hipóteses (notebook 01).** A meta de 2024 é função quase exata da taxa de 2023, com teto de 80%. O esforço exigido não explica o cumprimento da meta. O RS caiu 20 p.p. em 2024. A participação é o sinal mais limpo, e a renda explica o nível, não o cumprimento.
2. **Separação e vazamento (notebook 02).** 20% para teste, estratificado. Imputação, escala e codificação dentro do `Pipeline`, ajustadas só no treino de cada fold. Só entram variáveis de 2023. Como demonstração, incluir a taxa de 2024 leva o AUC de 0,70 a 0,997.
3. **Escopo.** O RS ficou fora do treino (choque de 2024) e foi analisado à parte. Com o RS dentro, a macrorregião "Sul" aprende a ser sinônimo de risco.
4. **Comparação e ajuste.** Baseline, regressão logística, random forest e gradient boosting, com validação cruzada 5x2. A random forest foi ajustada com busca aleatória (12 combinações).
5. **Generalização.** Cinco sementes, validação por UF (GroupKFold), modelo sem geografia e teste no RS.
6. **Interpretação.** Importância por permutação e SHAP.
7. **Aplicação estratégica (notebook 03).** Fatores, ranking de risco, agrupamento de UFs e metas futuras.

Variáveis do modelo: taxa, média de português e participação de 2023, log da população, log do PIB per capita e macrorregião. O esforço exigido (meta menos taxa) ficou de fora por ser redundante com a taxa.

## Escolha do algoritmo

Random forest (300 árvores, profundidade máxima 8, folhas de pelo menos 5 municípios). Teve o melhor AUC de validação (0,708), mas a diferença para o gradient boosting (0,697) está dentro do desvio entre folds (0,014). A regressão logística (0,680) ficou atrás.

## Métricas de avaliação

Foco no AUC, e também precisão e recall da classe "não cumpre a meta", que é a de interesse para a política pública.

| | Validação cruzada | Teste |
|---|---|---|
| AUC | 0,708 | 0,708 |
| Acurácia | | 0,673 (baseline 0,571) |
| Precisão, não cumpre | 0,635 | 0,664 |
| Recall, não cumpre | 0,493 | 0,479 |

Generalização: com cinco sementes diferentes o AUC fica em 0,705 (desvio de 0,003). Sem a macrorregião cai para 0,658, e com estados inteiros fora do treino cai para cerca de 0,60.

## Interpretação dos resultados

Importância por permutação (queda do AUC): macrorregião 0,110, participação 0,066, taxa de 2023 0,053, média de português 0,043, porte 0,026, PIB per capita praticamente zero. O SHAP confirma a leitura: por coluna, a participação tem o maior impacto médio (0,067), seguida pela taxa de 2023 (0,052), pelo Nordeste (0,047) e pelo porte (0,038). O Sul quase não pesa (0,002), o que mostra o efeito de deixar o RS fora do treino. A probabilidade de cumprir a meta sobe com a participação e cai com a taxa de partida (regressão à média) e com o porte.

## Insights encontrados

* A participação na avaliação é a variável mais associada ao cumprimento da meta. O efeito aparece dentro de cada faixa de taxa (diferença de 16 a 18 p.p.).
* Renda não prevê o cumprimento da meta, apesar de se relacionar com o nível da taxa.
* Municípios grandes oscilam menos e cumprem menos a meta (46,9% contra 56,7% nos menores).
* O risco previsto se confirma nos dados: no decil de maior risco, 75,5% não cumpriram a meta.
* O maior risco está em municípios de porte médio do Norte e do Nordeste (AM, PA, BA, RN, PB, MA).
* Quatro grupos de UFs com perfil semelhante. O de pior resultado (AM, PA e RJ) combina participação baixa, municípios maiores e quase nenhum avanço.

## Limitações do projeto

* Só há um par de anos para treinar (2023 e 2024), e não há como testar o modelo em outro ano.
* A geografia é instável: parte do efeito da macrorregião vem de choques de 2024 (por exemplo, RS e MG).
* O RS ficou fora do treino, e o modelo não antecipa o choque: previu 52% de cumprimento onde o observado foi 9,5%. A hipótese da enchente de 2024 não pôde ser testada com os dados disponíveis.
* O recall da classe "não cumpre" é de 48% no limiar de 0,5.
* PIB e população usam o último ano disponível na API do IBGE, que pode não coincidir com 2023.
* A meta de 2025 está ancorada na taxa de 2023. Em 1.881 municípios a distância até ela passa do máximo visto no treino, e o risco deles vem da frequência histórica de avanços, uma aproximação e não uma previsão do modelo.

## Aplicação prática para políticas públicas

* Priorizar apoio técnico nos municípios de maior risco (`reports/risco_metas_2025.csv`), usando o ranking e as faixas, e não as probabilidades como valores exatos.
* Monitorar e ampliar a participação na avaliação, que é a variável mais acionável do modelo (associação, não causa comprovada).
* Tratar choques regionais, como o do RS, com ações específicas e não pela média nacional.
* Olhar municípios pequenos com cautela, porque a taxa deles oscila muito de um ano para o outro.

## Possíveis evoluções futuras

* Retreinar com 2024 e 2025 quando houver resultado (`monta_features(abt, 2024, 2025)` já aceita outros anos).
* Incorporar Censo Escolar, FUNDEB e outros indicadores além do PIB.
* Escolher o limiar de decisão por custo e calibrar as probabilidades.
* Testar a hipótese da enchente com dados de municípios em situação de emergência.
