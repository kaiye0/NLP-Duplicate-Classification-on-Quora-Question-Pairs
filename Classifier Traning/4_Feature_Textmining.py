import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import time
from collections import Counter

data_root = '/Users/lemon/Downloads/EE5239/Data/'

def get_weight(count, eps=5000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)


# Get features by text mining
SAFE_DIV = 0.0001
STOP_WORDS = stopwords.words("english")  # stopwords in English


def get_text_mining_features(q1, q2, weights):
    tm_features = [0.0] * 16

    q1 = q1.split()
    q2 = q2.split()

    if len(q1) == 0 or len(q2) == 0:  # computer-generated empty questions
        return tm_features

    q1_words = set([word for word in q1 if word not in STOP_WORDS])
    q2_words = set([word for word in q2 if word not in STOP_WORDS])

    q1_stops = set([word for word in q1 if word in STOP_WORDS])
    q2_stops = set([word for word in q2 if word in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1).intersection(set(q2)))

    # TF-IDF reweighing
    common_word_weights = np.sum([weights.get(w, 0) for w in q1_words.intersection(q2_words)])
    common_stop_weights = np.sum([weights.get(w, 0) for w in q1_stops.intersection(q2_stops)])
    common_token_weights = np.sum([weights.get(w, 0) for w in set(q1).intersection(set(q2))])

    total_word_weights_1 = np.sum([weights.get(w, 0) for w in q1_words])
    total_word_weights_2 = np.sum([weights.get(w, 0) for w in q2_words])
    total_stop_weights_1 = np.sum([weights.get(w, 0) for w in q1_stops])
    total_stop_weights_2 = np.sum([weights.get(w, 0) for w in q2_stops])
    total_token_weights_1 = np.sum([weights.get(w, 0) for w in q1])
    total_token_weights_2 = np.sum([weights.get(w, 0) for w in q2])

    # Feature 0~5: (for different word type) number of common words / number of words in each question (shorter first)
    tm_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    tm_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    tm_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    tm_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    tm_features[4] = common_token_count / (min(len(q1), len(q2)) + SAFE_DIV)
    tm_features[5] = common_token_count / (max(len(q1), len(q2)) + SAFE_DIV)

    # Feature 6~11: TF-IDF reweighted features 0~5
    tm_features[6] = common_word_weights / (min(total_word_weights_1, total_word_weights_2) + SAFE_DIV)
    tm_features[7] = common_word_weights / (max(total_word_weights_1, total_word_weights_2) + SAFE_DIV)
    tm_features[8] = common_stop_weights / (min(total_stop_weights_1, total_stop_weights_2) + SAFE_DIV)
    tm_features[9] = common_stop_weights / (max(total_stop_weights_1, total_stop_weights_2) + SAFE_DIV)
    tm_features[10] = common_token_weights / (min(total_token_weights_1, total_token_weights_2) + SAFE_DIV)
    tm_features[11] = common_token_weights / (max(total_token_weights_1, total_token_weights_2) + SAFE_DIV)

    # shallow text mining
    tm_features[12] = int(q1[0] == q2[0])  # Is the first element the same
    tm_features[13] = int(q1[-1] == q2[-1])  # Is the last element the same
    tm_features[14] = abs(len(q1) - len(q2))  # Absolute difference of length of questions
    tm_features[15] = (len(q1) + len(q2)) / 2  # Average length of questions
    return tm_features


def extract_tm_features(df, weights):
    print("Extracting text mining features...")
    tm_features = df.apply(lambda x: get_text_mining_features(x["question1"], x["question2"], weights), axis=1)
    df["cwc_min"] = list(map(lambda x: x[1], tm_features))
    df["cwc_max"] = list(map(lambda x: x[0], tm_features))
    df["csc_min"] = list(map(lambda x: x[3], tm_features))
    df["csc_max"] = list(map(lambda x: x[2], tm_features))
    df["ctc_min"] = list(map(lambda x: x[5], tm_features))
    df["ctc_max"] = list(map(lambda x: x[4], tm_features))
    df["cww_min"] = list(map(lambda x: x[7], tm_features))
    df["cww_max"] = list(map(lambda x: x[6], tm_features))
    df["csw_min"] = list(map(lambda x: x[9], tm_features))
    df["csw_max"] = list(map(lambda x: x[8], tm_features))
    df["ctw_min"] = list(map(lambda x: x[11], tm_features))
    df["ctw_max"] = list(map(lambda x: x[10], tm_features))
    df["first_eq"] = list(map(lambda x: x[12], tm_features))
    df["last_eq"] = list(map(lambda x: x[13], tm_features))
    df["diff_len"] = list(map(lambda x: x[14], tm_features))
    df["mean_len"] = list(map(lambda x: x[15], tm_features))
    return df


# TF-IDF reweighting
df_train = pd.read_csv(data_root + "train_preprocessed.csv")
df_test = pd.read_csv(data_root + "test_preprocessed.csv")
qs_train = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)
qs_test = pd.Series(df_test['question1'].tolist() + df_test['question2'].tolist()).astype(str)
words_train = (" ".join(qs_train)).lower().split()
counts_train = Counter(words_train)
weights_train = {word: get_weight(count) for word, count in counts_train.items()} #TF-DF weights
words_test = (" ".join(qs_test)).lower().split()
counts_test = Counter(words_test)
weights_test = {word: get_weight(count) for word, count in counts_test.items()} #TF-DF weights

cols_tm = ["first_eq", "last_eq", "diff_len", "mean_len", "cwc_min", "cwc_max", "csc_min", "csc_max", "ctc_min", "ctc_max"]
cols_tfidf = ["cww_min", "cww_max", "csw_min", "csw_max", "ctw_min", "ctw_max"]

# Feature extraction; save to different csv.
train_start = time.time()
print("Extracting features for training set:")
feature_tm_train = extract_tm_features(df_train, weights_train)
feature_tm_train.loc[:, cols_tm].to_csv(data_root + "text_mining_features_train.csv", index=False)
feature_tm_train.loc[:, cols_tfidf].to_csv(data_root + "tfidf_features_train.csv", index=False)
train_end = time.time()
print('\nFeatures extracted in %fs' % (train_end - train_start))

test_start = time.time()
print("Extracting features for test set:")
feature_tm_test = extract_tm_features(df_test, weights_test)
feature_tm_test.loc[:, cols_tm].to_csv(data_root + "text_mining_features_test.csv", index=False)
feature_tm_test.loc[:, cols_tfidf].to_csv(data_root + "tfidf_features_test.csv", index=False)
test_end = time.time()
print('\nFeatures extracted in %fs' % (test_end - test_start))

# plotting
plt.figure(figsize=(12,6))
sns.boxplot(x="is_duplicate", y="cwc_max", data=feature_tm_train )
plt.xlabel('Is duplicate', fontsize=12)
plt.ylabel('Ratio of common words (non-stopwords)', fontsize=12)
plt.show()