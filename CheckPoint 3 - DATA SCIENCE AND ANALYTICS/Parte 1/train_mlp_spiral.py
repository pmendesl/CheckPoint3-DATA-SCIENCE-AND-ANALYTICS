import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Carregar os dados
data = np.load('spiral_data.npz')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']
X_train_scaled = data['X_train_scaled']
X_test_scaled = data['X_test_scaled']

# Função para plotar as fronteiras de decisão
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap='coolwarm')
    plt.title(title)
    plt.savefig(f'decision_boundary_config_{i+1}.png')
    plt.close()

# Configurações a serem testadas
configurations = [
    {'hidden_layer_sizes': (10,), 'activation': 'relu', 'learning_rate_init': 0.01, 'normalize': False, 'max_iter': 1000},
    {'hidden_layer_sizes': (10, 10), 'activation': 'relu', 'learning_rate_init': 0.01, 'normalize': False, 'max_iter': 1000},
    {'hidden_layer_sizes': (20, 20), 'activation': 'tanh', 'learning_rate_init': 0.001, 'normalize': False, 'max_iter': 1000},
    {'hidden_layer_sizes': (10,), 'activation': 'relu', 'learning_rate_init': 0.01, 'normalize': True, 'max_iter': 1000},
    {'hidden_layer_sizes': (10, 10), 'activation': 'relu', 'learning_rate_init': 0.01, 'normalize': True, 'max_iter': 1000}
]

results = []

for i, config in enumerate(configurations):
    print(f"\nTreinando modelo com configuração: {config}")
    if config['normalize']:
        X_train_data = X_train_scaled
        X_test_data = X_test_scaled
    else:
        X_train_data = X_train
        X_test_data = X_test

    mlp = MLPClassifier(
        hidden_layer_sizes=config['hidden_layer_sizes'],
        activation=config['activation'],
        learning_rate_init=config['learning_rate_init'],
        max_iter=config['max_iter'],
        random_state=42
    )
    mlp.fit(X_train_data, y_train)
    y_pred = mlp.predict(X_test_data)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Acurácia do modelo: {accuracy:.4f}")
    results.append({'config': config, 'accuracy': accuracy})

    # Plotar fronteira de decisão
    plt.figure(figsize=(8, 6))
    plot_decision_boundary(mlp, X_test_data, y_test, f"Fronteira de Decisão - Config {i+1} (Acc: {accuracy:.4f})")
    plt.tight_layout()

# Salvar resultados em um arquivo de texto
with open('mlp_spiral_results.txt', 'w') as f:
    for res in results:
        f.write(f"Configuração: {res['config']}, Acurácia: {res['accuracy']:.4f}\n")

print("\nResultados salvos em mlp_spiral_results.txt")
