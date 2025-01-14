import numpy as np
import matplotlib.pyplot as plt

# Load libsvm format data
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

# Decision stump weak learner
class DecisionStump:
    def __init__(self):
        self.j = None  # Best feature index
        self.threshold = None  # Best threshold
        self.polarity = None  # Polarity of the decision

    def train(self, X, y, weights):
        """Train decision stump by minimizing weighted error."""
        n, d = X.shape
        best_error = float('inf')
        for j in range(d):  # Loop through all features
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
        """Predict using trained stump."""
        predictions = self.polarity * np.sign(X[:, self.j] - self.threshold)
        return predictions

# AdaBoost implementation
class AdaBoost:
    def __init__(self, T):
        self.T = T
        self.stumps = []
        self.alphas = []
        self.errors = []
        self.U_t = []

    def fit(self, X, y):
        """Train AdaBoost with decision stumps."""
        n = X.shape[0]
        weights = np.ones(n) / n  # Initialize uniform weights
        cumulative_weight = 1  # Initialize cumulative weight product

        for t in range(self.T):
            stump = DecisionStump()
            error = stump.train(X, y, weights)

            # Calculate alpha
            epsilon_t = max(error, 1e-10)  # Avoid division by zero
            alpha = 0.5 * np.log((1 - epsilon_t) / epsilon_t)

            # Update U_t for Problem 12
            cumulative_weight *= 2 * np.sqrt(epsilon_t * (1 - epsilon_t))
            self.U_t.append(cumulative_weight)

            # Store stump and alpha
            self.stumps.append(stump)
            self.alphas.append(alpha)
            self.errors.append(error)

            # Update weights
            predictions = stump.predict(X)
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            print(f"Iteration {t + 1}, Weighted Error: {error:.4f}, Alpha: {alpha:.4f}")

    def predict(self, X, t=None):
        """Predict using the first t weak learners (use all if t=None)."""
        if t is None:
            t = len(self.stumps)
        final_predictions = np.zeros(X.shape[0])
        for i in range(t):
            final_predictions += self.alphas[i] * self.stumps[i].predict(X)
        return np.sign(final_predictions)

# Main function
if __name__ == "__main__":
    # Load data
    X_train, y_train = load_data("train.txt")
    X_test, y_test = load_data("test.txt")

    # Convert labels to -1 and +1
    y_train = np.where(y_train == 0, -1, y_train)
    y_test = np.where(y_test == 0, -1, y_test)

    # Train AdaBoost
    T = 500
    adaboost = AdaBoost(T)
    adaboost.fit(X_train, y_train)

    # Compute E_in(G_t), E_out(G_t), and U_t
    E_in = []
    E_out = []
    for t in range(1, T + 1):
        predictions_train = adaboost.predict(X_train, t)
        predictions_test = adaboost.predict(X_test, t)
        E_in.append(np.mean(predictions_train != y_train))  # In-sample error
        E_out.append(np.mean(predictions_test != y_test))   # Out-of-sample error

    # Plot for Problem 11: E_in(G_t) and E_out(G_t)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T + 1), E_in, label="$E_{in}(G_t)$", color="blue")
    plt.plot(range(1, T + 1), E_out, label="$E_{out}(G_t)$", color="orange")
    plt.xlabel("Iteration $t$")
    plt.ylabel("Error")
    plt.title("In sample and out-of-sample Errors vs Iterations $t$")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot for Problem 12: U_t and E_in(G_t)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T + 1), adaboost.U_t, label="$U_t$", color="purple")
    plt.plot(range(1, T + 1), E_in, label="$E_{in}(G_t)$", color="blue")
    plt.xlabel("Iteration $t$")
    plt.ylabel("Value")
    plt.title("$U_t$ and $E_{in}(G_t)$ vs Iterations $t$")
    plt.legend()
    plt.grid()
    plt.show()