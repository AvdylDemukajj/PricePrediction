import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression

data = pd.read_csv('car_purchasing_data.csv')
data

X = data[['gender', 'age', 'salary', 'debt', 'worth']]

y = data['amount']

model = LinearRegression()
model.fit(X, y)

new_data = np.array([[1, 33, 75500, 15645, 335845]])

predicted_price = model.predict(new_data)
print('purchasing price will be: ', predicted_price)