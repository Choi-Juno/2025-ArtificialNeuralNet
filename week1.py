import numpy as np
import tensorflow as tf

Data_set = np.loadtxt("./data/ThoraricSurgery3.csv", delimiter=",")
x = Data_set[:, 0:16]
y = Data_set[:, 16]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(30, input_dim=16, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(x, y, epochs=5, batch_size=16)
