import numpy as np
import matplotlib.pyplot as plt

# Load libsvm format dataset
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

# Regularized E_in (i.e. E_{in}^u), here we're asked to use 0 / 1 error
def regularized_E_in(u, predictions, y_data):
    error_sum = 0
    for i in range(len(predictions)):
        individual_error = 1 if predictions[i] != y_data[i] else 0
        weighted_individual_error = individual_error * u[i]
        error_sum += weighted_individual_error
    return error_sum / len(predictions)

# Define decision stump weak learner
class DecisionStump:
    def __init__(self):
        self.j = None  # Feature index
        self.threshold = None
        self.s = None
        self.alpha = None

    # train the decision stump model using weighted error
    def train(self, X, y, weights):
        n, d = X.shape
        best_error = float('inf')
        for j in range(d):
            thresholds = np.unique(X[:, j])
            for threshold in thresholds:
                for s in [-1, 1]:
                    predictions = np.ones(n) * s
                    predictions[X[:, j] < threshold] = -s
                    error = np.sum(weights[predictions != y])
                    if error < best_error:
                        best_error = error
                        self.j = j
                        self.threshold = threshold
                        self.s = s
        return best_error

    def predict(self, X):
        N = X.shape[0]
        predictions = np.ones(N) * self.s
        predictions[X[:, self.j] < self.threshold] = -self.s
        return predictions

# AdaBoost implementation
class AdaBoost:
    def __init__(self, T):
        self.T = T
        self.stumps = []
        self.alphas = []
        self.weighted_Ein = []

    def fit(self, X, y):
        n = X.shape[0]
        weights = np.ones(n) / n  
        for t in range(self.T):
            stump = DecisionStump()
            error = stump.train(X, y, weights)

            if error == 0:
                alpha = 1e6  # Assign a very large weight since we're asked not to stop before T < 500
            else:
                alpha = 0.5 * np.log((1 - error) / error)

            # Update weights
            predictions = stump.predict(X)
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            # Track weighted E_in
            weighted_error = regularized_E_in(weights, predictions, y)
            self.weighted_Ein.append(weighted_error)

            # Store the stump and alpha
            stump.alpha = alpha
            self.stumps.append(stump)
            self.alphas.append(alpha)

            print(f"Iteration {t + 1}, Weighted Error: {weighted_error:.4f}, Alpha: {alpha:.4f}")

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

    # AdaBoost with Decision Stumps
    T = 500
    adaboost = AdaBoost(T)
    adaboost.fit(X_train, y_train)

    # Compute normalized Ein(Gt) and Eout(Gt)
    Ein = adaboost.weighted_Ein  # Normalized E_in
    Eout = []
    for t in range(1, len(adaboost.stumps) + 1):
        predictions_test = adaboost.predict(X_test, t)
        Eout.append(np.mean(predictions_test != y_test))

    # Plot Ein and Eout
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(Ein) + 1), Ein, label="Normalized $E_{in}^u(G_t)$ (Training Error)")
    plt.plot(range(1, len(Eout) + 1), Eout, label="$E_{out}^u(G_t)$ (Testing Error)")
    plt.xlabel("Iteration $t$")
    plt.ylabel("Error")
    plt.title("Normalized Training and Testing Errors vs Iterations for $G_t$")
    plt.legend()
    plt.grid()
    plt.show()