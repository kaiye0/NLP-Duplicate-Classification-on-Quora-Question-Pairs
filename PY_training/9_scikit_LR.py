import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

path = "D:/University/Assignment/2019 - 2020/2019 Fall Semester/Nonlinear Optimization/Project/"
classical_train  = np.loadtxt(open(path + "feature/classical_features_train.csv"  ,"rb"), delimiter=",", skiprows = 1)
classical_test   = np.loadtxt(open(path + "feature/classical_features_test.csv"   ,"rb"), delimiter=",", skiprows = 1)
similarity_train = np.loadtxt(open(path + "feature/similarity_features_train.csv" ,"rb"), delimiter=",", skiprows = 1)
similarity_test  = np.loadtxt(open(path + "feature/similarity_features_test.csv"  ,"rb"), delimiter=",", skiprows = 1)
structural_train = np.loadtxt(open(path + "feature/structural_features_train.csv" ,"rb"), delimiter=",", skiprows = 1)
structural_test  = np.loadtxt(open(path + "feature/structural_features_test.csv"  ,"rb"), delimiter=",", skiprows = 1)
tfidf_train      = np.loadtxt(open(path + "feature/tfidf_features_train.csv"      ,"rb"), delimiter=",", skiprows = 1)
tfidf_test       = np.loadtxt(open(path + "feature/tfidf_features_test.csv"       ,"rb"), delimiter=",", skiprows = 1)
train            = np.loadtxt(open(path + "feature_train.csv"                     ,"rb"), delimiter=",", skiprows = 1)
test             = np.loadtxt(open(path + "feature_test.csv"                      ,"rb"), delimiter=",", skiprows = 1)
b                = np.loadtxt(open(path + "train_label.csv"                       ,"rb"), delimiter=",", skiprows = 1).reshape(-1, 1)

classical_clf = LogisticRegression(random_state=0).fit(classical_train, b)
classical_is_duplicate = classical_clf.predict_proba(classical_test)
classical_df = pd.DataFrame()
classical_df['test_id'] = np.arange(0, len(classical_is_duplicate), 1)
classical_df['is_duplicate'] = classical_is_duplicate[:,1]
classical_df.to_csv(path + "scikit Classical Features Predicted.csv", index=False)

similarity_clf = LogisticRegression(random_state=0).fit(similarity_train, b)
similarity_is_duplicate = similarity_clf.predict_proba(similarity_test)
similarity_df = pd.DataFrame()
similarity_df['test_id'] = np.arange(0, len(similarity_is_duplicate), 1)
similarity_df['is_duplicate'] = similarity_is_duplicate[:,1]
similarity_df.to_csv(path + "scikit Similarity Features Predicted.csv", index=False)

structural_clf = LogisticRegression(random_state=0).fit(structural_train, b)
structural_is_duplicate = structural_clf.predict_proba(structural_test)
structural_df = pd.DataFrame()
structural_df['test_id'] = np.arange(0, len(structural_is_duplicate), 1)
structural_df['is_duplicate'] = structural_is_duplicate[:,1]
structural_df.to_csv(path + "scikit Structural Features Predicted.csv", index=False)

tfidf_clf = LogisticRegression(random_state=0).fit(tfidf_train, b)
tfidf_is_duplicate = tfidf_clf.predict_proba(tfidf_test)
tfidf_df = pd.DataFrame()
tfidf_df['test_id'] = np.arange(0, len(tfidf_is_duplicate), 1)
tfidf_df['is_duplicate'] = tfidf_is_duplicate[:,1]
tfidf_df.to_csv(path + "scikit TFIDF Features Predicted.csv", index=False)

clf = LogisticRegression(random_state=0).fit(train, b)
is_duplicate = clf.predict_proba(test)
df = pd.DataFrame()
df['test_id'] = np.arange(0, len(is_duplicate), 1)
df['is_duplicate'] = is_duplicate[:,1]
df.to_csv(path + "scikit Overall Features Predicted.csv", index=False)