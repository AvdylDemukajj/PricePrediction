import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car_purchasing_data.csv')

le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])

X = data[['gender', 'age', 'salary', 'debt', 'worth']]
y = data['amount']

model = LinearRegression()
model.fit(X, y)

new_data = np.array([[1, 33, 75500, 15645, 335845]])
predicted_price = model.predict(new_data)

print('Predicted purchasing price will be:', predicted_price[0])
