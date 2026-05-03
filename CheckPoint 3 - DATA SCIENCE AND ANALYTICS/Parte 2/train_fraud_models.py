
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Carregar o dataset
df = pd.read_csv('creditcard.csv')

# Separar features (X) e target (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Dividir em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalizar as features (exceto 'Time' e 'Amount' que já são transformadas no dataset original)
# As features V1-V28 já são o resultado de uma PCA, então não precisam de normalização adicional.
# No entanto, 'Time' e 'Amount' precisam ser escaladas.
scaler = StandardScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

# Modelos a serem testados
models = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'Linear SVM': LinearSVC(random_state=42, class_weight='balanced', dual=False, max_iter=2000) # dual=False para n_samples > n_features
}

results = []

for name, model in models.items():
    print(f"\nTreinando {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Resultados para {name}:")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\n" + classification_report(y_test, y_pred))

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

# Salvar resultados em um arquivo de texto
with open('fraud_detection_results.txt', 'w') as f:
    f.write("Resultados da Classificação de Fraudes em Cartão de Crédito\n\n")
    for res in results:
        f.write(f"Modelo: {res['Model']}\n")
        f.write(f"Acurácia: {res['Accuracy']:.4f}\n")
        f.write(f"Precisão: {res['Precision']:.4f}\n")
        f.write(f"Recall: {res['Recall']:.4f}\n")
        f.write(f"F1-Score: {res['F1-Score']:.4f}\n\n")

print("\nResultados salvos em fraud_detection_results.txt")
