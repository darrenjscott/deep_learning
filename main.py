from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_shape=(2,), activation='relu'))
model.add(Dense(1))

print(model.summary())

model.compile(optimizer='adam', loss='mse')

## Need to add some training and test data
model.fit(X_test, y_test, epochs=20, validation_split=0.2)

print("Final loss value:",model.evaluate(X_train, y_train))
