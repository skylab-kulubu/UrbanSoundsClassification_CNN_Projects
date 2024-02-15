import tensorflow as tf
import numpy as np

x_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
x_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")
x_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(12, 3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(24, 3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=10, validation_data=(x_val, y_val))

test_loss, test_acc = model.evaluate(x_test, y_test)

print('Test accuracy:', test_acc)
