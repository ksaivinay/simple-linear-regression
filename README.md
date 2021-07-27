# simple-linear-regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv(r"""C:\Users\guntu\Desktop\data science\linear regression\test.csv""")
print(data)

data.isnull().sum() #There are no null value in the training dataset

data.shape #There are 300 rows and 2 columsn in the dataset

data_1 = pd.read_csv(r"""C:\Users\guntu\Desktop\data science\linear regression\train.csv""")
print(data_1)

data_1.isnull().sum() #There is one missing values in our training dataset
data_1.shape

# Dropping the missing value as the missing value count is less
data_2 = data_1.dropna()
data_2.shape 
# As we dropped the one missing value the shape of the data is 699 rows and two columns

# As the testing data has no decimals like we have in training set we should reshape the data same as test data before training

# Resize to fit
x_train = np.array(data_2.iloc[:,0].values)
x_train = x_train.reshape(-1,1)
x_train[:20]

y_train = np.array(data_2.iloc[:,1].values)
y_train[:20]

x_test = np.array(data.iloc[:,0].values)
y_test = np.array(data.iloc[:,1].values)
x_test = x_test.reshape(-1,1)
print(x_test[:20])
print(y_test[:20])

reg = LinearRegression()
reg.fit(x_train,y_train)

y_predict = reg.predict(x_test)
y_predict[:20]

# scatter plot
plt.scatter(x_test,y_test,color="black")
plt.plot(x_test,y_predict,color="yellow")
plt.title("Linear Regression")
plt.show()

print("Model Accuracy : {accuracy}".format(accuracy = reg.score(x_test,y_test)*100))
