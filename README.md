# Felicidade por idade e país

Laboratório de Inteligência Artificial da PUC-SP sobre a relação entre idade,
país e avaliação de vida, usando dados do World Happiness Report republicados
pela Our World in Data, indicadores do Banco Mundial e modelos do scikit-learn.

## Como executar

Requer Python 3.10 ou superior. No terminal, na pasta do projeto:

```bash
python -m pip install -r requirements.txt
```

Depois abra `PUCSP_CS_AI_10_AI_ML_Happines.ipynb` no VS Code ou no Google Colab
e execute o notebook de cima para baixo. Os dados por idade são referentes a
2021-2023; os indicadores do Banco Mundial usam o último valor disponível entre
2019 e 2023. A data exata do download deve ser registrada no notebook antes da
entrega.

## Conteúdo

- `PUCSP_CS_AI_10_AI_ML_Happines.ipynb`: análise, gráficos, regressão linear e logística.
- `lab_helpers.py`: download, transformação e enriquecimento dos dados.
- `regressao_linear.py`: modelos lineares, validação agrupada e análise da curva de idade.
- `regressao_logistica.py`: classificação, avaliação agrupada e consulta de probabilidade.
- `requirements.txt`: dependências para reproduzir a análise.

## Divisão sugerida da dupla

Uma pessoa pode revisar `regressao_linear.py` e as visualizações da Parte 3;
a outra pode revisar `regressao_logistica.py` e as visualizações da Parte 4.
As duas pessoas devem revisar conjuntamente os dados, a discussão e os limites
da análise.

## Limitações

As observações são médias de grupos por país e faixa etária, não indivíduos.
Uma fotografia transversal mistura efeitos de idade, país e coorte de nascimento;
portanto, os modelos mostram associações e não relações causais. O ponto médio
das faixas também é uma aproximação, especialmente para a faixa `60+`.

## Fontes

- World Happiness Report, capítulo “Happiness and age”, edição 2024.
- Our World in Data, “Self-reported life satisfaction by age”, CC BY.
- Banco Mundial, indicadores `NY.GDP.PCAP.PP.KD`, `SP.POP.TOTL` e `AG.LND.TOTL.K2`.
- Gallup, World Happiness Report; World Economic Forum, “At what age does happiness peak?”.
