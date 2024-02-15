import tensorflow as tf
import numpy as np

x_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
x_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")
x_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

def apply_gradients(model, x, y):
    with tf.GradientTape() as tape:
        predictions = model(x)
        loss = loss_object(y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size):
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_train_predictions = 0
        total_train_samples = 0
        
        for batch_start in range(0, len(x_train), batch_size):
            batch_x = x_train[batch_start:batch_start + batch_size]
            batch_y = y_train[batch_start:batch_start + batch_size]
            
            loss = apply_gradients(model, batch_x, batch_y)
            epoch_loss += loss.numpy() * len(batch_x)

            predictions = model(batch_x)
            correct_train_predictions += np.sum(np.argmax(predictions, axis=1) == batch_y)
            total_train_samples += len(batch_x)
            
        train_loss = epoch_loss / len(x_train)
        train_accuracy = correct_train_predictions / total_train_samples

        val_predictions = model(x_val)
        val_loss = loss_object(y_val, val_predictions)
        val_accuracy = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y_val, val_predictions))

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss.numpy()}, Val Accuracy: {val_accuracy.numpy()}')

train_model(model, x_train, y_train, x_val, y_val, epochs=10, batch_size=64)

test_predictions = model(x_test)
test_loss = loss_object(y_test, test_predictions)
test_accuracy = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(y_test, test_predictions))

print('Test accuracy:', test_accuracy.numpy())
