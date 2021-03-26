import numpy as np
import pandas as pd

path = "D:/University/Assignment/2019 - 2020/2019 Fall Semester/Nonlinear Optimization/Project/"
train = np.loadtxt(open(path + "feature_train.csv", "rb"), delimiter=",", skiprows = 1)
test = np.loadtxt(open(path + "feature_test.csv", "rb"), delimiter=",", skiprows = 1)
b = np.loadtxt(open(path + "train_label.csv", "rb"), delimiter=",", skiprows = 1).reshape(-1, 1)

[M,N] = train.shape

def svm(A,b,M,c,step,max_iter):
    l = np.random.randn(M, 1)
    M1 = np.hstack((l, np.zeros([M,1])))
    min1 = M1.min(axis = 1)
    min1 = min1.reshape(-1, 1)
    M2 = np.hstack((min1, np.zeros([M,1])))
    max2 = M2.max(axis = 1)
    max2 = max2.reshape(-1, 1)
    l = max2
    x = np.matmul(np.transpose(A),np.multiply(l,b))
    for t in range(max_iter):
        for i in range(M):
            print(str(t) + " " + str(i))
            d =  1 - b[i]*np.matmul(A[i,:],x)
            x = np.subtract(x, l[i]*b[i]*np.transpose(A[i,:]).reshape(-1, 1))
            l[i] = l[i] + step*d
            l[i] = max(min(l[i],c), 0)
            x = np.add(x, l[i]*b[i]*np.transpose(A[i,:]).reshape(-1, 1))
    return x

x = svm(train,b,M,1,0.0001,100)

is_duplicate = np.matmul(test,x)

df = pd.DataFrame()
df['test_id'] = np.arange(0, len(b), 1)
df['is_duplicate'] = is_duplicate[:,1]
df.to_csv(path + "SVM Predicted.csv", index=False)