"""Train a small scikit-learn model on the Iris dataset.

This is deliberately simple: the focus of the lab is testing and automation,
not model building.
"""
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_model():
    """Train a classifier and return (X_test, y_test, model).

    X_test and y_test are the held-out test split, which the unit tests use
    to check the model's quality.
    """
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    # random_state makes the training reproducible: same data in, same model out.
    model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    return X_test, y_test, model


if __name__ == "__main__":
    # Only runs with `python model.py`, not when another file imports train_model.
    X_test, y_test, model = train_model()
    print("Model trained. Test accuracy:", model.score(X_test, y_test))
    print("Sample prediction:", model.predict([X_test[0]]))
