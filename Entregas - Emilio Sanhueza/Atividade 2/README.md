# 2 - Classificação de Texto com Keras/TensorFlow

Projeto **ponta a ponta** de classificação de texto usando o dataset **Reuters** (`keras.datasets.reuters`): dados → pré-processamento → modelo → treino → validação → teste → inferência.

## Visão geral

- **Dataset**: Reuters-21578 (11.228 notícias, 46 tópicos, vocabulário das décadas de 1980-90).
- **Modelo**: `Embedding` + `GlobalMaxPooling1D` + `Dense(64, relu)` + `Dropout` + `Dense(46, softmax)` — ~647k parâmetros.
- **Treino**: 15 épocas, Adam, `categorical_crossentropy`, `validation_split=0.2`.
- **Extras**: comparação de arquiteturas (MLP × Conv1D × LSTM), matriz de confusão, inferência com texto novo e salvamento/recarga do modelo (`.keras`).

## Como usar

1. Abra o notebook no **Google Colab** (é obrigatório — o ambiente check falha fora dele).
[2](2). Execute as células em ordem. O TensorFlow já vem instalado; as demais dependências (`matplotlib`, `seaborn`, `scikit-learn`) são instaladas automaticamente.

Clique [neste link](https://colab.research.google.com/drive/1jLGTYVkP5c3dlq-O3nUGhmIV8VAZhd3S?usp=sharing) para acessar diretamente no Google Colab:
`https://colab.research.google.com/drive/1jLGTYVkP5c3dlq-O3nUGhmIV8VAZhd3S?usp=sharing`.

## Dependências

`tensorflow`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

> Não rode `pip install -U numpy` — o notebook cuida da versão compatível com o Colab automaticamente.
