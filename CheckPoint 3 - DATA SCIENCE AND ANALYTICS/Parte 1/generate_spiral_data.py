import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_spiral(n_samples=1000, noise=0.2):
    np.random.seed(42)

    n = n_samples // 2

    theta = np.sqrt(np.random.rand(n)) * 2.3 * np.pi
    r = 0.5 * theta + np.pi

    x1 = r * np.cos(theta) + np.random.randn(n) * noise
    y1 = r * np.sin(theta) + np.random.randn(n) * noise

    x2 = -r * np.cos(theta) + np.random.randn(n) * noise
    y2 = -r * np.sin(theta) + np.random.randn(n) * noise

    X = np.vstack([
        np.column_stack([x1, y1]),
        np.column_stack([x2, y2])
    ])

    y = np.hstack([
        np.zeros(n),
        np.ones(n)
    ])

    return X, y

X, y = generate_spiral(n_samples=1000)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

np.savez('spiral_data.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled)
