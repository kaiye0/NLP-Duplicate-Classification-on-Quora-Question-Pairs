import re
import pandas as pd
from fuzzywuzzy import fuzz
import distance
import time

data_root = '/Users/lemon/Downloads/EE5239/Data/'

def extract_fuzzy_features(df):
    print("Extracting fuzzy features..")
    # fuzzy features
    df["token_set_ratio"] = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
    df["token_sort_ratio"] = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
    df["fuzz_ratio"] = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
    df["fuzz_partial_ratio"] = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)

    print("Extracting similarity features..")
    # similarity features
    df["dis_sor"] = df.apply(lambda x: distance.sorensen(x["question1"], x["question2"]), axis=1) # sorensen similarity coefficient
    df["dis_jac"] = df.apply(lambda x: distance.jaccard(x["question1"], x["question2"]), axis=1) # jaccard similarity coefficient
    df["dis_levs"] = df.apply(lambda x: distance.nlevenshtein(x["question1"], x["question2"], method=1), axis=1) # two methods of normalization
    df["dis_levl"] = df.apply(lambda x: distance.nlevenshtein(x["question1"], x["question2"], method=2), axis=1)
    return df

cols_fuzz = ["token_set_ratio", "token_sort_ratio", "fuzz_ratio", "fuzz_partial_ratio"]
cols_sim = ["dis_sor", "dis_jac", "dis_levs", "dis_levl"]
df_train = pd.read_csv(data_root + "train_preprocessed.csv")
df_test = pd.read_csv(data_root + "test_preprocessed.csv")

train_start = time.time()
print("Extracting features for training set:")
feature_fz_train = extract_fuzzy_features(df_train)
feature_fz_train.loc[:, cols_fuzz].to_csv(data_root + "fuzzy_features_train.csv", index=False)
feature_fz_train.loc[:, cols_sim].to_csv(data_root + "similarity_features_train.csv", index=False)
train_end = time.time()
print('\nFeatures extracted in %fs' % (train_end - train_start))

test_start = time.time()
print("Extracting features for test set:")
feature_fz_test = extract_fuzzy_features(df_test)
feature_fz_test.loc[:, cols_fuzz].to_csv(data_root + "fuzzy_features_test.csv", index=False)
feature_fz_test.loc[:, cols_sim].to_csv(data_root + "similarity_features_test.csv", index=False)
test_end = time.time()
print('\nFeatures extracted in %fs' % (test_end - test_start))




