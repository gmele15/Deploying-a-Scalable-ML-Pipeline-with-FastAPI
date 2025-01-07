import pytest
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from ml.data import apply_label, process_data
from ml.model import train_model

def test_train_model():
    """
    Ensure that the function returns a model that is not None
    """
    X = pd.DataFrame({
        "workclass": ["Private", "Public", "Private"],
        "education": ["Bachelors", "Masters", "Bachelors"],
        "salary": [0, 1, 0]
    })

    cat_features = ["workclass", "education"]
    label = "salary"
    X_train, y_train, encoder, lb = process_data(X, cat_features, label, training=True)

    model = train_model(X_train, y_train)

    assert model is not None
    assert isinstance(model, RandomForestClassifier)

def test_process_data_training_true():
    """
    Ensure that the function returns the correct values when training is True
    """
    X = pd.DataFrame({
        "workclass": ["Private", "Public", "Private"],
        "education": ["Bachelors", "Masters", "Bachelors"],
        "salary": [0, 1, 0]
    })

    cat_features = ["workclass", "education"]
    label = "salary"
    X_train, y_train, encoder, lb = process_data(
        X,
        categorical_features=cat_features,
        label=label,
        training=True
    )

    assert X_train.shape == (3, 4)
    assert y_train.shape == (3,)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)

def test_apply_label():
    """
    Ensure that the label 0 is converted to '<=50K' and 1 is converted to '>50K'
    """
    assert apply_label([0]) == "<=50K"
    assert apply_label([1]) == ">50K"
