"""Unit tests: check model.py on its own, with no web server involved."""
import numpy as np

from model import train_model

# The minimum accuracy we accept. If the model drops below this, the pipeline fails.
ACCURACY_THRESHOLD = 1.01


def test_train_model_returns_valid_data():
    """The training function returns sensible test data and a fitted model."""
    X_test, y_test, model = train_model()
    print(f"\nTest features shape: {X_test.shape}")
    print(f"Test classes present: {np.unique(y_test)}")
    print(f"Model type: {type(model).__name__}")

    assert X_test.shape[1] == 4, "Iris has 4 features per flower"
    assert len(X_test) == len(y_test), "Every row needs a label"
    assert len(np.unique(y_test)) > 1, "Test labels should contain more than one class"


def test_model_accuracy_gate():
    """Quality gate: fail the pipeline if the model gets noticeably worse."""
    X_test, y_test, model = train_model()
    accuracy = model.score(X_test, y_test)
    print(f"\nAccuracy on the held-out test set: {accuracy:.3f}")

    assert accuracy >= ACCURACY_THRESHOLD, (
        f"Accuracy {accuracy:.3f} is below the {ACCURACY_THRESHOLD} threshold"
    )


def test_training_is_reproducible():
    """Training twice gives identical predictions (random_state is fixed)."""
    X_test, _, model_a = train_model()
    _, _, model_b = train_model()

    assert (model_a.predict(X_test) == model_b.predict(X_test)).all()
