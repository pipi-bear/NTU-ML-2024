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

# Define a decision stump weak learner
class DecisionStump:
    def __init__(self):
        self.j = None   
        self.threshold = None  
        self.s = None  
        self.alpha = None  

    def train(self, X, y, weights):
        N, d = X.shape
        best_error = float('inf')
        for j in range(d):  # Iterate through all dimensions
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

    def fit(self, X, y):
        n = X.shape[0]
        weights = np.ones(n) / n  # Initialize uniform weights
        for t in range(self.T):
            stump = DecisionStump()
            error = stump.train(X, y, weights)

            # Avoid division by zero
            if error == 0:
                alpha = 1e6
            else:
                alpha = 0.5 * np.log((1 - error) / error)

            predictions = stump.predict(X)
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)

            # Store the stump and alpha
            stump.alpha = alpha
            self.stumps.append(stump)
            self.alphas.append(alpha)
            print(f"Iteration {t + 1}, Error: {error:.4f}, Alpha: {alpha:.4f}")

    def predict(self, X, t=None):
        if t is None:
            t = len(self.stumps)
        final_predictions = np.zeros(X.shape[0])
        for i in range(t):
            final_predictions += self.alphas[i] * self.stumps[i].predict(X)
        return np.sign(final_predictions)

# Main function
if __name__ == "__main__":
    # Load training and testing data
    X_train, y_train = load_data("train.txt")
    X_test, y_test = load_data("test.txt")

    # Convert labels to -1 and +1
    y_train = np.where(y_train == 0, -1, y_train)
    y_test = np.where(y_test == 0, -1, y_test)

    # AdaBoost with Decision Stumps
    T = 500
    adaboost = AdaBoost(T)
    adaboost.fit(X_train, y_train)

    # Compute true Ein and normalized Ein
    Ein = []
    normalized_Ein = []
    for t in range(1, len(adaboost.stumps) + 1):
        predictions_train = adaboost.predict(X_train, t)
        predictions_test = adaboost.predict(X_test, t)
        # True E_in (0/1 error)
        Ein.append(np.mean(predictions_train != y_train))
        # Normalized E_in
        weights = np.ones(len(y_train)) / len(y_train)  # Reset uniform weights
        normalized_error = np.sum(weights * (predictions_train != y_train))
        normalized_Ein.append(normalized_error)

    # Plot true Ein and normalized Ein
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, T + 1), Ein, label="$E_{in}$")
    plt.plot(range(1, T + 1), normalized_Ein, label="Normalized $E_{in}^u$")
    plt.xlabel("Iteration $t$")
    plt.ylabel("Error")
    plt.title("True and Normalized $E_{in}$ vs Iterations $t$")
    plt.legend()
    plt.grid()
    plt.show()