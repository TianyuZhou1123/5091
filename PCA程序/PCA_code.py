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

'''
C:\User_IDE\Anaconda\lib\site-packages\pymysql\cursors.py:170: Warning: (1366, "Incorrect string value: '\\xD6\\xD0\\xB9\\xFA\\xB1\\xEA...' for column 'VARIABLE_VALUE' at row 485")
 result = self._query(query)
净资产收益率(%)	销售净利率(%)	资产负债率(%)	流动比率	产权比率	存货周转率(次)	营业收入同比增长率(%)
公司名称							
美的集团	1.440081	0.830293	0.377623	-0.100267	0.135781	0.223424	0.639736
长虹美菱	-0.751914	-0.794639	0.355726	-0.401821	0.102529	0.623674	0.059897
格力电器	1.292375	1.463336	0.153035	-0.389759	-0.174576	0.274436	0.309460
海信家电	1.100094	0.116280	0.314739	-0.486257	0.036024	1.298605	0.493152
奥马电器	-0.661018	-0.737591	0.740894	-0.377697	0.823001	0.521649	0.064536
澳柯玛	0.022451	-0.268330	0.531466	-0.618941	0.390718	-0.259231	0.956095
海尔智家	0.792446	0.287423	0.429840	-0.643065	0.213371	-0.094422	0.728335
春兰股份	-0.732686	1.516703	-2.603779	3.192709	-1.726362	-2.178076	-2.935779
惠而浦	-1.567357	-1.966872	-0.403381	-0.124391	-0.739869	-0.200370	-0.432269
深康佳A	-0.621688	-0.730230	1.067669	-0.522443	1.776241	1.506578	1.212616
四川九洲	-0.751914	-0.686065	-0.127138	0.128915	-0.496017	-1.157831	-0.676729
ST毅昌	1.845618	-0.139513	1.050264	-0.763687	1.709736	0.019375	-1.002367
兆驰股份	0.223471	0.787968	-0.119277	-0.148516	-0.484933	0.278360	0.465320
银河电子	-0.307922	1.500141	-1.796386	1.274822	-1.482510	-1.679726	-0.665133
海信视像	-0.511564	-0.406348	-0.743631	0.611402	-0.983722	1.051392	0.173546
四川长虹	-0.810472	-0.772556	0.772336	-0.631003	0.900590	-0.227838	0.609584
'''
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
'''
array([[-0.9281547 , -0.0597476 , -1.25436302],
       [ 0.3697319 ,  0.71416796,  0.42062077],
       [-1.54794589, -0.34926737, -0.8277419 ],
       [-0.88480436,  0.93652448, -0.65223569],
       [ 0.61814053,  0.38988667,  0.22805551],
       [ 0.30805952,  0.05023352, -0.91110212],
       [-0.35065584,  0.0236765 , -1.03405142],
       [-0.02225511, -2.34478172,  2.53619608],
       [ 1.49229552,  1.08666863,  1.23520774],
       [ 0.55020154,  0.51242129, -0.5929361 ],
       [ 0.91789112, -0.11115907,  0.69280373],
       [ 0.31650382,  0.08462743, -0.28058143],
       [-0.95363778, -0.11141698, -0.37990628],
       [-0.56704727, -1.71204473,  0.61461163],
       [-0.29112807,  0.88333591,  0.49843629],
       [ 0.97280507,  0.00687509, -0.29301381]])
'''
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






