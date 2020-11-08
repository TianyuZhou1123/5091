import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
## Step 1: Find a data set
#configurate database
from sqlalchemy import create_engine
#create_engine("database type+database driver://database username:database password@IP address:port/database"，other parameters)
#modify password and database username
engine = create_engine('mysql+pymysql://root:147258@localhost:3306/mydatabase')

#read data and save the data to SQL
financial_data = pd.read_excel('MFM_data.xlsx')
financial_data.to_sql('financial_data', engine, if_exists='replace', index=False)

#read data from SQL
financial_data = pd.read_sql('select * from financial_data', engine, index_col='公司名称')

#standalize
financial_data = financial_data.apply(lambda x:(x-x.mean())/x.std())
financial_data
## Step 2: Build a covariance matrix
cov_array = np.cov(financial_data, rowvar =False)

## Step 3: Perform eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_array)
# Calculate the explanation proportion of each eigenvalues
prop = np.array([i/np.sum(eigenvalues) for i in eigenvalues])
reduce_n = np.argwhere(np.cumsum(prop) > 0.9)[0,0]

## Step 4: Reduce dimensionality
size = financial_data.shape[0]
norm_g_reduce = np.zeros(size * financial_data.shape[1]).reshape(size,financial_data.shape[1])
for i in range(financial_data.shape[1]):
    norm_g_reduce[:,i] = financial_data.iloc[:,i] - np.mean(financial_data.iloc[:,i])
reduce_dimension = np.dot(norm_g_reduce, eigenvectors[np.arange(0,reduce_n+1)].T)

## Step 5: Reconstruct original data
OG = reduce_dimension.copy()
for i in range(OG.shape[1]):
    OG[:,i] = OG[:,i] + np.mean(financial_data.iloc[:,i])
OG
## Some figures
# Original Data
fig1 = plt.figure(3)
plt.subplot(211)
plt.plot(np.array(financial_data),'o')
plt.title('Original Data')
plt.show()
# Reconstruct Original Data
plt.subplot(212)
plt.plot(OG,'o')
plt.title('Reconstruct Original Data')
plt.show()
