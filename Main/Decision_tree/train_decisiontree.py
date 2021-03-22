import pickle
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def train():
    dataset = pd.read_csv('../../Dataset/PIMA Indian Diabetes/diabetes.csv')
    # X = dataset[['Insulin', 'Glucose', 'Age', 'Pregnancies', 'BMI']]
    # Y = dataset[['Outcome']]

    dataset.rename({'DiabetesPedigreeFunction': 'DPF'}, inplace=True, axis=1)

    # mengganti nilai 0 dengan nan
    data_modified = dataset.copy(deep=True)
    data_modified[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = data_modified[
        ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)

    # mengganti nilai nan dengan mean dan median
    data_modified["Glucose"].fillna(data_modified["Glucose"].mean(), inplace=True)
    data_modified["BloodPressure"].fillna(data_modified["BloodPressure"].median(), inplace=True)
    data_modified["SkinThickness"].fillna(data_modified["SkinThickness"].mean(), inplace=True)
    data_modified["Insulin"].fillna(data_modified["Insulin"].median(), inplace=True)
    data_modified["BMI"].fillna(data_modified["BMI"].mean(), inplace=True)

    X = data_modified[['Insulin', 'Glucose', 'Age', 'Pregnancies', 'BMI']]
    Y = data_modified[['Outcome']]

    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=0)

    from sklearn import tree
    # Create Decision Tree classifer object
    clf = tree.DecisionTreeClassifier(criterion="entropy", random_state=0,
                                      max_depth=3)

    # Train Decision Tree Classifer
    tree = clf.fit(X_train, Y_train)

    # model = SVC(kernel='rbf', random_state=0)
    # svc = model.fit(X_train, Y_train)
    # Save Model As Pickle File
    with open('tree.pkl', 'wb') as m:
        pickle.dump(tree, m)
    test(X_test, Y_test)


# Test accuracy of the model


def test(X_test, Y_test):
    with open('tree.pkl', 'rb') as mod:
        p = pickle.load(mod)

    pre = p.predict(X_test)
    print(accuracy_score(Y_test, pre))  # Prints the accuracy of the model


def find_data_file(filename):
    if getattr(sys, "frozen", False):
        # The application is frozen.
        datadir = os.path.dirname(sys.executable)
    else:
        # The application is not frozen.
        datadir = os.path.dirname(__file__)

    return os.path.join(datadir, filename)


def check_input(data) -> int:
    df = pd.DataFrame(data=data, index=[0])
    with open(find_data_file('tree.pkl'), 'rb') as model:
        p = pickle.load(model)
    op = p.predict(df)
    return op[0]


if __name__ == '__main__':
    train()
