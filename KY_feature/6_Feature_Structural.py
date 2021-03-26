from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

data_root = '/Users/lemon/Downloads/EE5239/Data/'
FREQ_UPPER_BOUND = 100
NEIGHBOR_UPPER_BOUND = 5

def get_neighbors(train_df, test_df):
    neighbors = defaultdict(set)
    for df in [train_df, test_df]:
        for q1, q2 in zip(df["qid1"], df["qid2"]):
            neighbors[q1].add(q2)
            neighbors[q2].add(q1)
    return neighbors


def get_neighbor_features(df, neighbors):
    common_nc = df.apply(lambda x: len(neighbors[x.qid1].intersection(neighbors[x.qid2])), axis=1)
    min_nc = df.apply(lambda x: min(len(neighbors[x.qid1]), len(neighbors[x.qid2])), axis=1)
    df["common_neighbor_ratio"] = common_nc / min_nc
    df["common_neighbor_count"] = common_nc.apply(lambda x: min(x, NEIGHBOR_UPPER_BOUND))
    return df


def get_freq_features(df, frequency_map):
    df["freq1"] = df["qid1"].map(lambda x: min(frequency_map[x], FREQ_UPPER_BOUND))
    df["freq2"] = df["qid2"].map(lambda x: min(frequency_map[x], FREQ_UPPER_BOUND))
    return df


def get_kcore_dict(df):
    g = nx.Graph() # graph built woth networkx structure
    g.add_nodes_from(df.qid1) # node tagged by question id
    edges = list(df[["qid1", "qid2"]].to_records(index=False))
    g.add_edges_from(edges)
    g.remove_edges_from(g.selfloop_edges())
    list_kcore = pd.DataFrame(data=g.nodes(), columns=["qid"])
    list_kcore["kcore"] = 0
    for k in range(2, 10):
        ck = nx.k_core(g, k=k).nodes()
        dict_kcore.ix[list_kcore.qid.isin(ck), "kcore"] = k
    return list_kcore.to_dict()["kcore"]


def get_kcore_features(df, kcore_dict): # get the k-core degenearacy of q1 & q2 from the k_core
    df["kcore1"] = df["qid1"].apply(lambda x: kcore_dict[x])
    df["kcore2"] = df["qid2"].apply(lambda x: kcore_dict[x])
    return df


def resort(df, col): #resort each min-mac feature pair for better generalization
    sorted_features = np.sort(np.vstack([df[col + "1"], df[col + "2"]]).T) #resort in each row
    df["min_" + col] = sorted_features[:, 0]
    df["max_" + col] = sorted_features[:, 1]
    return df.drop([col + "1", col + "2"], axis=1)


df_train = pd.read_csv(data_root + "train_preprocessed.csv")
df_test = pd.read_csv(data_root + "test_preprocessed.csv")

start = time.time()
print("Extracting common neighbor features...")
neighbors = get_neighbors(df_train, df_test)
df_train = get_neighbor_features(df_train, neighbors)
df_test = get_neighbor_features(df_test, neighbors)

print("Extracting frequency features...")
frequency_map = dict(zip(*np.unique(np.vstack((all_df["qid1"], all_df["qid2"])), return_counts=True)))
df_train = get_freq_features(df_train, frequency_map)
df_test = get_freq_features(df_test, frequency_map)
df_train = convert_to_minmax(df_train, "freq")
df_test = convert_to_minmax(df_test, "freq")

print("Extracting kcore features...")
all_df = pd.concat([df_train, df_test])
kcore_dict = get_kcore_dict(all_df)
df_train = get_kcore_features(df_train, kcore_dict)
df_test = get_kcore_features(df_test, kcore_dict)
df_train = convert_to_minmax(df_train, "kcore")
df_test = convert_to_minmax(df_test, "kcore")

cols = ["min_kcore", "max_kcore", "common_neighbor_count", "common_neighbor_ratio", "min_freq", "max_freq"]
df_train.loc[:, cols].to_csv(data_root + "structural_features_train.csv", index=False)
df_test.loc[:, cols].to_csv(data_root + "structural_features_test.csv", index=False)
end = time.time()
print('\nFeatures extracted in %fs' % (end - start))

#### Plotting
# upper bound set to suffciently large for plotting
cnc = df_train['common_neighbor_count'].value_counts()
df_mean_cnc = df_train.groupby('common_neighbor_count')['is_duplicate'].aggregate(np.mean).reset_index()

plt.figure(figsize=(12,8))
sns.barplot(cnc.index, np.log1p(cnc.values)) # log bar plot
plt.xlabel('Number of Common Neighbours', fontsize=12)
plt.ylabel('Log of the Number of Questions', fontsize=12)
plt.xticks(rotation='vertical')
ax2 = plt.twinx()
sns.pointplot(df_mean_cnc["q1_q2_intersect"].values, df_mean_cnc["is_duplicate"].values, ax = ax2)
plt.ylabel('Average "is_duplicate" value', fontsize=12)
plt.show()

minfre = df_train['min_freq'].value_counts()
plt.figure(figsize=(12,8))
sns.barplot(minfre.index, np.log1p(minfre.values), alpha=0.8)
plt.xlabel('Number of neighbors of the question asked less often', fontsize=12)
plt.ylabel('Log of the Number of Questions', fontsize=12)
plt.xticks(rotation='vertical')
mean_minfre = df_train.groupby('min_freq')['is_duplicate'].aggregate(np.mean).reset_index()
ax2 = plt.twinx()
sns.pointplot(mean_minfre["min_freq"].values, mean_minfre["is_duplicate"].values, ax=ax2)
plt.ylabel('Average "is_duplicate" value', fontsize=12)
plt.show()

# similar as above; not used in the report
# maxfre = df_train['max_freq'].value_counts()
# plt.figure(figsize=(12,8))
# sns.barplot(maxfre.index, np.log1p(maxfre.values), alpha=0.8)
# plt.xlabel('Number of neighbors of the question asked more often', fontsize=12)
# plt.ylabel('Log of the Number of Questions', fontsize=12)
# plt.xticks(rotation='vertical')
# mean_minfre = df_train.groupby('max_freq')['is_duplicate'].aggregate(np.mean).reset_index()
# ax2 = plt.twinx()
# sns.pointplot(mean_maxfre["max_freq"].values, mean_maxfre["is_duplicate"].values, ax=ax2)
# plt.ylabel('Average "is_duplicate" value', fontsize=12)
# plt.show()

