import numpy as np
import math
import pandas as pd
from scipy.linalg import norm

path = "D:/University/Assignment/2019 - 2020/2019 Fall Semester/Nonlinear Optimization/Project/"
train = np.loadtxt(open(path + "feature_train.csv", "rb"), delimiter=",", skiprows = 1)
test = np.loadtxt(open(path + "feature_test.csv", "rb"), delimiter=",", skiprows = 1)
b = np.loadtxt(open(path + "train_label.csv", "rb"), delimiter=",", skiprows = 1).reshape(-1, 1)

[M,N] = train.shape

def logreg(A,b,M,max_iter):
    x = np.zeros([N,1])
    sum1 = 0
    sum2 = 0
    for i in range(M):
        print("0 " + str(i))
        sum1 += b[i]*np.transpose(A[i,:])/(1 + math.exp(b[i]*np.matmul(A[i,:],x)))
        sum2 += norm(A[i,:])*norm(A[i,:])
    g = -sum1/M
    g = g.reshape(-1, 1)
    a = M/sum2
    k = 1
    while k < max_iter:
        x = x - a*g
        s = 0
        for i in range(M):
            s += b[i]*np.transpose(A[i,:])/(1 + math.exp(b[i]*np.matmul(A[i,:],x)))
            print(str(k) + " " + str(i))
        g = -s/M
        g = g.reshape(-1, 1)
        k += 1
        print(k)
    return x

x = logreg(train,b,M,100)

is_duplicate = np.matmul(test,x)

df = pd.DataFrame()
df['test_id'] = np.arange(0, len(b), 1)
df['is_duplicate'] = is_duplicate[:,1]
df.to_csv(path + "Logistic Regression Predicted.csv", index=False)