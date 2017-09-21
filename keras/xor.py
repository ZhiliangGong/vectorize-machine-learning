import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation

np.random.seed(42)
X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y = np.array([[0],[1],[1],[0]]).astype('float32')

xor = Sequential()
xor.add(Dense(2, input_dim = 2))
xor.add(Activation('tanh'))
xor.add(Dense(1))
xor.add(Activation('sigmoid'))

xor.compile(loss="binary_crossentropy", optimizer="adam", metrics = ["accuracy"])
xor.summary()
history = xor.fit(X, y, epochs = 100, verbose = 0)

score = xor.evaluate(X, y)
print("\nAccuracy: ", score[-1])
print("\nPredictions:")
print(xor.predict_proba(X))
