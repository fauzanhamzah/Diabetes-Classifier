import pickle
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler


# fungsi untuk train


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

    # X = data_modified[['Insulin', 'Glucose', 'Age', 'Pregnancies', 'BMI']]
    # Y = data_modified[['Outcome']]

    X = data_modified.iloc[:, 0:8]
    Y = data_modified.iloc[:, 8]

    bestfeatures = SelectKBest(score_func=chi2, k=5)
    fit = bestfeatures.fit(X, Y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns

    bestfeatures = SelectKBest(score_func=chi2, k=5).fit_transform(X, Y)

    X_new = bestfeatures
    # print(X_new)

    # train test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_new, Y, test_size=0.20, random_state=0)

    # sc_X = StandardScaler()
    # X_train = sc_X.fit_transform(X_train)
    # X_test = sc_X.transform(X_test)

    from sklearn.svm import SVC
    model = SVC(kernel='rbf', random_state=0)
    svc = model.fit(X_train, Y_train)
    # simpan model hasil training menjadi pickle file
    with open('train_svc.pkl', 'wb') as m:
        pickle.dump(svc, m)
    test(X_test, Y_test)


# test akurasi model


def test(X_test, Y_test):
    with open('train_svc.pkl', 'rb') as mod:
        p = pickle.load(mod)

    pre = p.predict(X_test)
    print(accuracy_score(Y_test, pre))  # menampilkan hasil akurasi


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
    with open(find_data_file('train_svc.pkl'), 'rb') as model:
        p = pickle.load(model)
    op = p.predict(df)
    return op[0]


if __name__ == '__main__':
    train()
