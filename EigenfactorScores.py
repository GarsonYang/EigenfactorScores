import pandas as pd
import numpy as np

# 1.1 input data
data = pd.read_csv('links.txt', header = None)

# set constants 
a = 0.85
e = 0.00001

# 1.2 create an adjacency matrix
journals = list()
for i in range(len(data)):
    journals.append(data[1][i])
journals = set(journals)
size = len(journals)
Z = np.zeros((size,size),float)
for i in range(len(data)):
    Z[data[1][i], data[0][i]] = data[2][i]

# 1.3 set diagnoal to 0
np.fill_diagonal(Z,0)

# 1.3 normalization
def sumCol(array):
    row=array.shape[0]
    col=array.shape[1]
    col_sum = np.zeros(col,float)
    for j in range(col):
        for i in range(row):
            col_sum[j] += array[i][j]
    return col_sum

def normalize(array):
    row=array.shape[0]
    col=array.shape[1]
    col_sum = sumCol(array)
    H = np.zeros((row,col),float)
    for j in range(col):
        if col_sum[j] != 0:
            for i in range(row):
                H[i,j]=array[i,j]/col_sum[j]
    return H

H = normalize(Z)

# 1.4 dangling vector d
d = np.zeros(size,int)
col_sum = sumCol(Z)
for j in range(size):
    if col_sum[j] == 0:
        d[j] = 1

# 1.5 article vetor A
A = np.ones((size,1),float)
summ = np.sum(A)
for j in range(size):
    A[j] = A[j]/summ

# 1.5 initial start vector pai
pai = np.zeros((size,1),float)
for i in range(size):
    pai[i] = 1/size

# 1.5 iteratively calculate pai to converge to the eigenvector P
count = 0
while (True):
    pai_ = a*np.dot(H,pai)+(a*np.dot(d,pai)+(1-a))*A
    r = np.subtract(pai, pai_)
    pai = pai_
    count += 1
    if np.sum(abs(r)) < e:
        break

# 1.6 calculate eigenfactor EF
EF = normalize(np.dot(H, pai))*100

# result output
print(count)
top20 = np.flip(np.sort(EF,axis=0))[:20]
for i in top20:
    print(np.where(EF == i)[0],i)