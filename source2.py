import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
df = sns.load_dataset('taxis')
df
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
df
x = df[['distance', 'fare', 'tip']]
y = df.total
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.15, random_state=42)
from ctypes import LibraryLoader
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
model.score(x_test, y_test)
from sklearn.tree import DecisionTreeRegressor
model2 = DecisionTreeRegressor(criterion= 'friedman_mse', random_state=0)
model2.fit(x_train,y_train)
model2.score(x_test, y_test)
import pickle
filename = 'taxis_regression.sav'
pickle.dump(model, open(filename, 'wb'))