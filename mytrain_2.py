import sys
import os
import datetime

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import metrics
import pickle
import joblib

pathname = os.path.dirname(sys.argv[0])
path = os.path.abspath(pathname)


def train_model(data):
    train, test = train_test_split(
        data, test_size=0.4, stratify=data["species"], random_state=42
    )
    X_train = train[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_train = train.species
    X_test = test[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y_test = test.species

    mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
    mod_dt.fit(X_train, y_train)
    prediction = mod_dt.predict(X_test)
    print(
        "The accuracy of the Decision Tree is",
        "{:.3f}".format(metrics.accuracy_score(prediction, y_test)),
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    return mod_dt, timestamp


def train_top_model():
    # Load Iris dataset
    data1 = pd.read_csv("mydata/v2/v2/data.csv")
    v1_r1_model, v1_r1_t = train_model(data1)
    joblib.dump(v1_r1_model, "model.joblib")


train_top_model()

