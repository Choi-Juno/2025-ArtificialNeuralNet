import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

Data_set = np.loadtxt("./data/ThoraricSurgery3.csv", delimiter=",")
x = Data_set[:, 0:16]
y = Data_set[:, 16]

model = Sequential()
model.add(Dense(30, input_dim=16, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(x, y, epochs=5, batch_size=16)
