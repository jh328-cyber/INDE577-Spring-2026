"""Supervised learning algorithms."""

from ml_toolkit.supervised.decision_tree import DecisionTreeClassifier
from ml_toolkit.supervised.knn import KNNClassifier, KNNRegressor
from ml_toolkit.supervised.linear_regression import LinearRegression
from ml_toolkit.supervised.logistic_regression import LogisticRegression
from ml_toolkit.supervised.naive_bayes import GaussianNB
from ml_toolkit.supervised.neural_network import MLPClassifier
from ml_toolkit.supervised.perceptron import Perceptron
from ml_toolkit.supervised.random_forest import RandomForestClassifier
from ml_toolkit.supervised.svm import LinearSVC

__all__ = [
    "LinearRegression",
    "LogisticRegression",
    "Perceptron",
    "KNNClassifier",
    "KNNRegressor",
    "DecisionTreeClassifier",
    "RandomForestClassifier",
    "GaussianNB",
    "MLPClassifier",
    "LinearSVC",
]
