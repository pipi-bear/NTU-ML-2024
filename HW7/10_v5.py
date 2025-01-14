import numpy as np
import matplotlib.pyplot as plt

def load_data(file):
    labels = []
    features = []
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            labels.append(int(parts[0]))
            feature_dict = {int(p.split(':')[0]): float(p.split(':')[1]) for p in parts[1:]}
            features.append(feature_dict)
    max_index = max(max(d.keys()) for d in features)
    X = np.zeros((len(features), max_index))
    for i, feature_dict in enumerate(features):
        for index, value in feature_dict.items():
            X[i, index - 1] = value
    return np.array(X), np.array(labels)

class DecisionStump:
    def __init__(self):
        self.j = None  
        self.threshold = None  
        self.s = None  

    def train(self, X, y, weights):
        n, d = X.shape
        best_error = float('inf')
        # Loop through all d features
        # note: the definition of extended multi-dimensional decision stump model is in HW2.pdf 13.
        for j in range(d):  
            thresholds = np.unique(X[:, j])
            for threshold in thresholds:
                for s in [-1, 1]:
                    predictions = s * np.sign(X[:, j] - threshold)
                    error = np.sum(weights[predictions != y])
                    if error < best_error:
                        best_error = error
                        self.j = j
                        self.threshold = threshold
                        self.s = s
        return best_error

    def predict(self, X):
        predictions = self.s * np.sign(X[:, self.j] - self.threshold)
        return predictions

class AdaBoost:
    def __init__(self, T):
        self.T = T
        self.stumps = []
        self.alphas = []
        self.errors = []

    def fit(self, X, y):
        n = X.shape[0]
        # Initialize the weights as 1/n
        weights = np.ones(n) / n            

        # note: the definition can be found in ppt 208, p.17
        for t in range(self.T):
            stump = DecisionStump()
            error = stump.train(X, y, weights)

            epsilon_t = max(error, 1e-10)  
            alpha = 0.5 * np.log((1 - epsilon_t) / epsilon_t)

            self.stumps.append(stump)
            self.alphas.append(alpha)
            self.errors.append(error)

            predictions = stump.predict(X)
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            print(f"Iteration {t + 1}, Weighted Error: {error:.4f}, Alpha: {alpha:.4f}")

    def predict(self, X, t=None):
        if t is None:
            t = len(self.stumps)
        final_predictions = np.zeros(X.shape[0])
        for i in range(t):
            final_predictions += self.alphas[i] * self.stumps[i].predict(X)
        return np.sign(final_predictions)

if __name__ == "__main__":
    X_train, y_train = load_data("train.txt")
    X_test, y_test = load_data("test.txt")

    y_train = np.where(y_train == 0, -1, y_train)
    y_test = np.where(y_test == 0, -1, y_test)

    T = 500
    adaboost = AdaBoost(T)
    adaboost.fit(X_train, y_train)

    E_in = []
    E_out = []
    for t in range(1, T + 1):
        predictions_train = adaboost.predict(X_train, t)
        predictions_test = adaboost.predict(X_test, t)
        E_in.append(np.mean(predictions_train != y_train))  # In-sample error
        E_out.append(np.mean(predictions_test != y_test))   # Out-of-sample error

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T + 1), E_in, label="$E_{in}$", color="blue")
    plt.plot(range(1, T + 1), E_out, label="$E_{in}^{u}$ (normalized)", color="orange")
    plt.xlabel("Iteration $t$")
    plt.ylabel("Error")
    plt.title("Original and normalized E_in vs Iterations")
    plt.legend()
    plt.grid()
    plt.show()
