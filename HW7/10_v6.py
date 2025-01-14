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
        self.polarity = None  

    def train(self, X, y, weights):
        n, d = X.shape
        best_error = float('inf')
        # Loop through all d features
        # note: the definition of extended multi-dimensional decision stump model is in HW2.pdf 13.
        for j in range(d):  
            thresholds = np.unique(X[:, j])
            for threshold in thresholds:
                for polarity in [-1, 1]:
                    predictions = polarity * np.sign(X[:, j] - threshold)
                    error = np.sum(weights[predictions != y])
                    if error < best_error:
                        best_error = error
                        self.j = j
                        self.threshold = threshold
                        self.polarity = polarity
        return best_error

    def predict(self, X):
        predictions = self.polarity * np.sign(X[:, self.j] - self.threshold)
        return predictions

class AdaBoost:
    def __init__(self, T):
        self.T = T
        self.stumps = []
        self.alphas = []
        self.true_Ein = []
        self.weighted_Ein = []

    def fit(self, X, y):
        n = X.shape[0]

        # Initialize the weights as 1/n
        weights = np.ones(n) / n  

        # note: the definition can be found in ppt 208, p.17
        for t in range(self.T):
            stump = DecisionStump()
            error = stump.train(X, y, weights)

            # Calculate alpha
            epsilon_t = max(error, 1e-10)  
            alpha = 0.5 * np.log((1 - epsilon_t) / epsilon_t)

            # Predict with the current stump
            predictions = stump.predict(X)

            # Compute True Ein
            true_error = np.mean(predictions != y)
            self.true_Ein.append(true_error)

            # Compute Weighted Ein (Normalized)
            weighted_error = np.sum(weights * (predictions != y))
            self.weighted_Ein.append(weighted_error / np.sum(weights))

            # Update weights
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            # Store stump and alpha
            self.stumps.append(stump)
            self.alphas.append(alpha)

            print(f"Iteration {t + 1}, True Ein: {true_error:.4f}, Weighted Ein: {self.weighted_Ein[-1]:.4f}, Alpha: {alpha:.4f}")

if __name__ == "__main__":
    X_train, y_train = load_data("train.txt")

    # Convert labels to -1 and +1
    y_train = np.where(y_train == 0, -1, y_train)

    # Train AdaBoost
    T = 500
    adaboost = AdaBoost(T)
    adaboost.fit(X_train, y_train)

    # Plot True Ein and Normalized Ein
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T + 1), adaboost.true_Ein, label="True $E_{in}(g_t)$", color="blue")
    plt.plot(range(1, T + 1), adaboost.weighted_Ein, label="Normalized $E_{in}^u(g_t)$", color="orange")
    plt.xlabel("Iteration $t$")
    plt.ylabel("Error")
    plt.title("True and Normalized $E_{in}$ vs Iterations")
    plt.legend()
    plt.grid()
    plt.show()