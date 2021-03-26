import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk import word_tokenize


data_root = '/Users/lemon/Downloads/EE5239/Data/'
eng_stopwords = set(stopwords.words('english'))

# Size of the given data set
train_df = pd.read_csv(data_root + "train_preprocessed.csv")
test_df = pd.read_csv(data_root + "test_preprocessed.csv")
print(train_df.shape)
print(test_df.shape)

# Ratio of duplicate and non-duplicate cases
is_dup = train_df['is_duplicate'].value_counts()
print(is_dup / is_dup.sum())

# Plot distribution of question length
allq = pd.DataFrame(pd.concat([train_df['question1'], train_df['question2']]))
allq.columns = ["questions"]
allq["num_of_words"] = allq["questions"].apply(lambda x : len(str(x).split()))
now = allq['num_of_words'].value_counts()
print('Questions has at most %i non-stop words'% max(now.index) ," and at least %i non-stop words." % min(now.index))
plt.figure(figsize=(12,6))
sns.barplot(now.index[0:50], now.values[0:50])
plt.ylabel('Number of questions', fontsize=12)
plt.xlabel('Number of words in the question', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()



