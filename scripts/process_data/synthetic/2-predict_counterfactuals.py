from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as skm
from imblearn.over_sampling import SMOTE
import random
import numpy as np
import os

datasets = [1, 2, 3, 4, 5]

def v1():
    probs = [0.1, 0.25, 0.5, 0.75, 0.9]
    for prob in probs:
        for dataset in datasets:
            file_name = 'data_train_' + str(prob) + '_' + str(dataset) + '.csv'
            file_name_enc = 'data_train_enc_' + str(prob) + '_' + str(dataset) + '.csv'
            # ----- CHANGE THE FILE PATH -----
            file_path = '../../data/processed/synthetic/'
            df = pd.read_csv(os.path.join(file_path, file_name))
            df_enc = pd.read_csv(os.path.join(file_path, file_name_enc))
            t_unique = df['t'].unique()
            test_linear = {}
            test_lasso = {}

            # for 0.1, y=1 will be minority class and for 0.85, y=0 will be minority class.
            # perform smote to increase the accuracy?

            for i in t_unique:
                buffer = df[df['t'] == i]
                X = buffer.iloc[:, :2]
                y = buffer['y']

                # lr = LogisticRegression().fit(X, y)

                # lr = DecisionTreeRegressor().fit(X, y)
                lasso = Lasso(alpha=0.08).fit(X, y)
                linear = LinearRegression().fit(X, y)
                # lr = DecisionTreeClassifier().fit(X, y)
                test_linear[i] = linear
                test_lasso[i] = lasso

            for i in range(len(t_unique)):
                model_linear = test_linear[i]
                model_lasso = test_lasso[i]

                X = df.iloc[:, :2]
                prediction_linear = model_linear.predict(X)
                prediction_lasso = model_lasso.predict(X)

                real = df['y' + str(i)]

                print(skm.r2_score(real, prediction_linear))
                print(skm.r2_score(real, prediction_lasso))

                # handling the minority class

                df_enc['linear' + str(i)] = prediction_linear
                df_enc['lasso' + str(i)] = prediction_lasso

            print(df_enc)

            df_enc.to_csv(os.path.join(file_path, file_name_enc), index=False)
