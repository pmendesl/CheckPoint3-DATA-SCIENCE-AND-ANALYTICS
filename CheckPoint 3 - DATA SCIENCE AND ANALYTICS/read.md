Checkpoint 03 - Machine Learning

Este projeto explora dois cenários clássicos de classificação: um usando dados criados artificialmente para testar a lógica de redes neurais e outro com dados reais para detecção de fraudes.

Parte 1: Classificação em Espiral

Aqui o desafio foi lidar com dados que não podem ser separados por uma linha reta. Usei uma rede neural do tipo MLP (Perceptron Multicamadas) para aprender o formato de uma espiral.

•
O que foi feito: Testei diferentes camadas e neurônios para ver como a rede se comportava.

•
Conclusão: Percebi que aumentar o tamanho da rede ajuda até certo ponto, mas o ajuste fino da taxa de aprendizado e a normalização dos dados foram os grandes diferenciais para chegar a quase 100% de acerto.

Parte 2: Detecção de Fraude (Cartão de Crédito)

Nesta etapa, usei um dataset real do Kaggle. O maior problema aqui não é a complexidade dos dados, mas o fato de que existem pouquíssimas fraudes comparadas às transações normais (dados desbalanceados).

•
O que foi feito: Comparei modelos como Regressão Logística, SVM e Random Forest.

•
Conclusão: O Random Forest foi o melhor disparado. Ele conseguiu identificar as fraudes com precisão, evitando dar muitos "alarmes falsos", o que é essencial em um sistema bancário real.

