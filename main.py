import pandas as pd
import keras
from keras import layers


cars_df = pd.read_csv('./datasets/car_prices.csv')

print(cars_df.head())
print(cars_df.info())
print(cars_df[['year', 'condition', 'odometer', 'mmr', 'sellingprice']])
print(cars_df.columns)

# We will try and see how well year, condition, and odometer can predict the selling price
X = cars_df[['year', 'condition', 'odometer']]
y = cars_df[['sellingprice']]


model = keras.Sequential()
model.add(keras.Input(shape=(3,)))
model.add(layers.Dense(units=4, activation='relu'))
model.add(layers.Dense(units=3, activation='relu'))
model.add(layers.Dense(units=1))

model.compile(optimizer='adam', loss='mae')

histories = model.fit(X, y, batch_size=1000, epochs=10)
print(histories.history['loss'])