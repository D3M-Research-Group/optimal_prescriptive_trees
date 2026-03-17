import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import scipy.stats
import time
import os


def run(df, df_test, model, sm):
    t_unique = df['t'].unique()
    # for 0.1, y=1 will be minority class and for 0.85, y=0 will be minority class.
    # perform smote to increase the accuracy?
    for t in t_unique:
        buffer = df[df['t'] == t]
        X = buffer.iloc[:, :17]
        y = buffer['y']
        y_values = y.value_counts()
        
        if sm:
            if y_values[0] == 1 or y_values[1] == 1:
                pass
            else:
                if y_values[1] > 5 and y_values[0] > 5:
                    smote = SMOTE(sampling_strategy=1.0, k_neighbors=5)
                elif y_values[0] <= 5:
                    smote = SMOTE(sampling_strategy=1.0, k_neighbors=y_values[0] - 1)
                elif y_values[1] <= 5:
                    smote = SMOTE(sampling_strategy=1.0, k_neighbors=y_values[1]-1)
                X, y = smote.fit_resample(X, y)

        lr = model.fit(X, y)
            
        X_test = df_test.iloc[:, :17]
        df_test['pred' + str(t)] = [i[1] for i in lr.predict_proba(X_test)]

    ## EVALUATE PERFORMANCE
    def find_highest_y(row):
        if row['pred1'] > row['pred0'] and row['pred1'] > row['pred2']:
            return 1
        elif row['pred2'] > row['pred0'] and row['pred2'] > row['pred1']:
            return 2
        else:
            return 0
        
    def find_highest_y2(row):
        if row['pred1'] > row['pred0']:
            return 1
        else:
            return 0

    def t_opt(row):
        if row['y1'] == 1:
            return 1
        elif row['y2'] == 1:
            return 2
        else:
            return 0

    if len(t_unique) == 3:
        df_test['t_pred'] = df_test.apply(lambda row: find_highest_y(row), axis=1)
    elif len(t_unique) == 2:
        df_test['t_pred'] = df_test.apply(lambda row: find_highest_y2(row), axis=1)
    df_test['t_opt'] = df_test.apply(lambda row: t_opt(row), axis=1)
    return (df_test['t_opt'] == df_test['t_pred']).sum()/len(df_test)

def driver(model, sm):
    probs = ['0.33', 'r0.11', 'r0.06']
    seeds = [1, 2, 3, 4, 5]
    datasets = [1, 2, 3, 4, 5]


    opt_policy_dic = {}
    for prob in probs:
        buffer = []
        for dataset in datasets:
            for seed in seeds:

                file_name = 'data_train_' + str(prob) + '_' + str(dataset) + '.csv'
                file_name_test = 'data_test_' + str(prob) + '_' + str(dataset) + '.csv'
                # ----- CHANGE THE FILE PATH -----
                file_path = '../../../data/processed/warfarin/seed' + str(seed) + '/'
                df = pd.read_csv(os.path.join(file_path, file_name))
                df_test = pd.read_csv(os.path.join(file_path, file_name_test))
                
                buffer.append(run(df, df_test, model, sm))
        opt_policy_dic[prob] = buffer
    return opt_policy_dic

def make_table_sub(l, method, r):
    dataset_col = []
    seed_cols = []
    for dataset in [1, 2, 3, 4, 5]:
        for seed in [1, 2, 3, 4, 5]:
            dataset_col.append(dataset)
            seed_cols.append(seed)
    df = pd.DataFrame({'method': [method]*25, 'seed': seed_cols, 'dataset': dataset_col, 'oopt': l,
                      'randomization': [r]*25})
    return df

def make_table(dic, method):
    df = pd.DataFrame(columns=['method', 'seed', 'dataset', 'oopt', 'randomization'])
    for des, vals in dic.items():
        df_buffer = make_table_sub(vals, method, des)
        df = pd.concat([df, df_buffer], ignore_index=True)
    return df

df = pd.DataFrame(columns=['method', 'seed', 'dataset', 'oopt', 'randomization'])


df = pd.concat([df, make_table(driver(LogisticRegression(max_iter=10000, class_weight='balanced'), False), 'balanced_lr')], ignore_index=True)

df = pd.concat([df, make_table(driver(RandomForestClassifier(class_weight='balanced'), False), 
                               'balanced_rf')], ignore_index=True)


probs = ['0.33', 'r0.06', 'r0.11']
seeds = [1, 2, 3, 4, 5]
datasets = [1, 2, 3, 4, 5]
opt_policy_dic = {}
for prob in probs:
    buffer = []
    for dataset in datasets:
        for seed in seeds:
            file_name = 'data_train_' + str(prob) + '_' + str(dataset) + '.csv'
            file_name_test = 'data_test_' + str(prob) + '_' + str(dataset) + '.csv'
            # ----- CHANGE THE FILE PATH -----
            file_path = '../../../data/processed/warfarin/seed' + str(seed) + '/'
            df = pd.read_csv(os.path.join(file_path, file_name))
            df_test = pd.read_csv(os.path.join(file_path, file_name_test))

            def find_highest_y(row):
                if row['lrrf1'] > row['lrrf0'] and row['lrrf1'] > row['lrrf2']:
                    return 1
                elif row['lrrf2'] > row['lrrf0'] and row['lrrf2'] > row['lrrf1']:
                    return 2
                else:
                    return 0

            def t_opt(row):
                if row['y1'] == 1:
                    return 1
                elif row['y2'] == 1:
                    return 2
                else:
                    return 0

            df_test['t_opt'] = df_test.apply(lambda row: t_opt(row), axis=1)    
            df_test['t_pred'] = df_test.apply(lambda row: find_highest_y(row), axis=1)
            buffer.append((df_test['t_opt'] == df_test['t_pred']).sum()/len(df_test))
    opt_policy_dic[prob] = buffer

df = pd.concat([df, make_table(opt_policy_dic, 'lrrf')], ignore_index=True)

df.to_csv('../../../results/warfarin/compiled/RC/rc_raw.csv', index=False)