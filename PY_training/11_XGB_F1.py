import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

path = 'D:/University/Assignment/2019 - 2020/2019 Fall Semester/Nonlinear Optimization/Project/'

classical_train  = pd.read_csv(path + "feature/classical_features_train.csv")
classical_test   = pd.read_csv(path + "feature/classical_features_test.csv")
similarity_train = pd.read_csv(path + "feature/similarity_features_train.csv")
similarity_test  = pd.read_csv(path + "feature/similarity_features_test.csv")
structural_train = pd.read_csv(path + "feature/structural_features_train.csv")
structural_test  = pd.read_csv(path + "feature/structural_features_test.csv")
tfidf_train      = pd.read_csv(path + "feature/tfidf_features_train.csv")
tfidf_test       = pd.read_csv(path + "feature/tfidf_features_test.csv")
train            = pd.read_csv(path + "feature_train.csv")
test             = pd.read_csv(path + "feature_test.csv")
b                = pd.read_csv(path + "train_label.csv")

df_train = pd.read_csv(path + 'train.csv')
df_test = pd.read_csv(path + 'test.csv')

label = df_train['is_duplicate'].values

def F1(x_train, x_test, b):
    b = df_train['is_duplicate'].values
    pos_train = x_train[b == 1]
    neg_train = x_train[b == 0]
    
    p = 0.165
    scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
    while scale > 1:
        neg_train = pd.concat([neg_train, neg_train])
        scale -= 1
    neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
    
    x_train = pd.concat([pos_train, neg_train])
    y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
    del pos_train, neg_train
    
    x_train, x_valid, b, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=4242)
    # Set our parameters for xgboost
    params = {}
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['max_depth'] = 4
    
    d_train = xgb.DMatrix(x_train, label=b)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)
    
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    
    bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
    
    d_test = xgb.DMatrix(x_valid)
    p_test = bst.predict(d_test).tolist()
    for i in range(len(p_test)):
        if p_test[i] >= 0.5:
            p_test[i] = 1
        else:
            p_test[i] = 0
    return f1_score(y_valid, p_test, average='micro')

classical_F_1_Value = F1(classical_train , classical_test , label)
similarity_F_1_Value = F1(similarity_train , similarity_test , label)
structural_F_1_Value = F1(structural_train , structural_test , label)
tfidf_F_1_Value = F1(tfidf_train , tfidf_test , label)
overall_F_1_Value = F1(train , test , label)