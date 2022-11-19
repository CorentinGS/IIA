import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


class DecisionTreeClassifierParams:
    def __init__(self, criterion, max_depth, min_samples_leaf, random_state):
        self.criterion: str = criterion
        self.max_depth: int = max_depth
        self.min_samples_leaf: int = min_samples_leaf
        self.random_state: int = random_state
        self.name: str = f'{criterion}_{max_depth}_{min_samples_leaf}_{random_state}'

    def __str__(self):
        return f"DecisionTreeClassifier(criterion='{self.criterion}', max_depth={self.max_depth}, min_samples_leaf={self.min_samples_leaf}, " \
               f"random_state={self.random_state})"

    def create(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth,
                                      min_samples_leaf=self.min_samples_leaf, random_state=self.random_state)




def predict(model: DecisionTreeClassifier, x_test: pd.DataFrame) -> np.ndarray:
    return model.predict(x_test)


def get_accuracy(y_test: pd.DataFrame, y_pred: np.ndarray) -> float:
    return accuracy_score(y_test, y_pred)


def get_precision(y_test: pd.DataFrame, y_pred: np.ndarray) -> float:
    return accuracy_score(y_test, y_pred) * 100


def get_confusion_matrix(y_test: pd.DataFrame, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_test, y_pred)


def get_root_mean_squared_error(y_test: pd.DataFrame, y_pred: np.ndarray) -> float:
    return np.sqrt(metrics.mean_squared_error(y_test, y_pred))


def get_mean_squared_error(y_test: pd.DataFrame, y_pred: np.ndarray) -> float:
    return metrics.mean_squared_error(y_test, y_pred)


def get_absolute_error(y_test: pd.DataFrame, y_pred: np.ndarray) -> float:
    return metrics.mean_absolute_error(y_test, y_pred)


def plot_tree_graph(model: DecisionTreeClassifier):
    # Plot Decision Tree
    fn = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    cn = ["setosa", "versicolor", "virginica"]
    plt.figure(figsize=(7, 5))
    plot_tree(model, feature_names=fn, class_names=cn, filled=True)
    plt.show()


def init_data(random_state: int, test_size: float) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    # Load data
    iris = load_iris()

    # Create DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Add columns to DataFrame
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # Add target to DataFrame
    target = pd.DataFrame(iris.target)

    # Create test and train data
    x_train, x_test, y_train, y_test = train_test_split(df, target, random_state=random_state, test_size=test_size)

    return x_train, x_test, y_train, y_test


class Stats:
    def __init__(self, accuracy: float, precision: float, confusion_matrix: np.ndarray, root_mean_squared_error: float,
                 mean_squared_error: float, absolute_error: float, params: DecisionTreeClassifierParams):
        self.accuracy: float = accuracy
        self.precision: float = precision
        self.confusion_matrix: np.ndarray = confusion_matrix
        self.root_mean_squared_error: float = root_mean_squared_error
        self.mean_squared_error: float = mean_squared_error
        self.absolute_error: float = absolute_error
        self.params: DecisionTreeClassifierParams = params

    def __str__(self):
        return f"Stats(accuracy={self.accuracy}, precision={self.precision}, confusion_matrix={self.confusion_matrix}, " \
               f"root_mean_squared_error={self.root_mean_squared_error}, mean_squared_error={self.mean_squared_error}, " \
               f"absolute_error={self.absolute_error}, params={self.params.__str__()})"


def run_decision_tree_classifier(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.DataFrame,
                                 y_test: pd.DataFrame, params: DecisionTreeClassifierParams) -> Stats:
    # print(f"Decision Tree Classifier with params: {params}")
    model = params.create()
    model.fit(x_train, y_train)

    y_pred = predict(model, x_test)
    # print(f"Accuracy: {get_accuracy(y_test, y_pred)}")
    # print(f"Precision: {get_precision(y_test, y_pred)}")
    # print(f"Confusion Matrix: {get_confusion_matrix(y_test, y_pred)}")
    # print(f"Root Mean Squared Error: {get_root_mean_squared_error(y_test, y_pred)}")
    # print(f"Mean Squared Error: {get_mean_squared_error(y_test, y_pred)}")
    # print(f"Absolute Error: {get_absolute_error(y_test, y_pred)}")
    # plot_tree_graph(model)
    return Stats(get_accuracy(y_test, y_pred), get_precision(y_test, y_pred), get_confusion_matrix(y_test, y_pred),
                 get_root_mean_squared_error(y_test, y_pred), get_mean_squared_error(y_test, y_pred),
                 get_absolute_error(y_test, y_pred), params)


def main():
    x_train, x_test, y_train, y_test = init_data(100, 0.2)
    stats: [Stats] = []

    criterions = ['gini', 'entropy']
    max_depths = [3, 4, 5, 6, 7, 8, 9, 10]
    min_samples_leafs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    random_states = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    decision_tree_classifier_params = []

    for criterion in criterions:
        for max_depth in max_depths:
            for min_samples_leaf in min_samples_leafs:
                for random_state in random_states:
                    decision_tree_classifier_params.append(
                        DecisionTreeClassifierParams(criterion, max_depth, min_samples_leaf, random_state))

    for params in decision_tree_classifier_params:
        stats.append(run_decision_tree_classifier(x_train, x_test, y_train, y_test, params))

    for stat in stats:
        print(stat)

    # Print best accuracy and params
    best_accuracy = max(stats, key=lambda x: x.accuracy)
    print(f"Best accuracy: {best_accuracy.accuracy} with params: {best_accuracy.params.name}")

    # Print best precision and params
    best_precision = max(stats, key=lambda x: x.precision)
    print(f"Best precision: {best_precision.precision} with params: {best_precision.params.name}")

    # Print best root mean squared error and params
    best_root_mean_squared_error = min(stats, key=lambda x: x.root_mean_squared_error)
    print(f"Best root mean squared error: {best_root_mean_squared_error.root_mean_squared_error} with params: "
          f"{best_root_mean_squared_error.params.name}")

    # Print worst root mean squared error and params
    worst_root_mean_squared_error = max(stats, key=lambda x: x.root_mean_squared_error)
    print(f"Worst root mean squared error: {worst_root_mean_squared_error.root_mean_squared_error} with params: "
          f"{worst_root_mean_squared_error.params.name}")

    # Print worst accuracy and params
    worst_accuracy = min(stats, key=lambda x: x.accuracy)
    print(f"Worst accuracy: {worst_accuracy.accuracy} with params: {worst_accuracy.params.name}")

    # Print worst precision and params
    worst_precision = min(stats, key=lambda x: x.precision)
    print(f"Worst precision: {worst_precision.precision} with params: {worst_precision.params.name}")


def ex2():
    # Load data
    iris = load_iris()

    # Create DataFrame
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Add columns to DataFrame
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # Add target to DataFrame
    target = pd.DataFrame(iris.target)

    # Create test and train data
    X_train, X_test, y_train, y_test = train_test_split(df, target, random_state=1, test_size=0.3)

    # Create Decision Tree Classifier
    model_decisiontree = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
    model_decisiontree.fit(X_train, y_train)

    # Predict
    y_pred = model_decisiontree.predict(X_test)

    # Accuracy Score
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    precision = accuracy * 100
    print("Precision:", precision)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:", cm)

    # Classification Report
    print("Classification Report:", classification_report(y_test, y_pred))

    # Absolute Error
    print("Absolute Error:", metrics.mean_absolute_error(y_test, y_pred))

    # Mean Squared Error
    print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred))

    # Root Mean Squared Error
    print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # Plot Decision Tree
    fn = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    cn = ["setosa", "versicolor", "virginica"]
    plt.figure(figsize=(7, 5))
    plot_tree(model_decisiontree, feature_names=fn, class_names=cn, filled=True)
    plt.show()


if __name__ == "__main__":
    main()
