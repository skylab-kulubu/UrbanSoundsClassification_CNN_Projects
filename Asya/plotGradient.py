import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
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

def calculate_accuracy(y_true, y_pred):
    correct_predictions = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_pred, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy

def train_model(model, x_train, y_train, x_val, y_val, epochs, batch_size):
    history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for batch_start in range(0, len(x_train), batch_size):
            batch_x = x_train[batch_start:batch_start + batch_size]
            batch_y = y_train[batch_start:batch_start + batch_size]

            with tf.GradientTape() as tape:
                predictions = model(batch_x)
                loss = loss_object(batch_y, predictions)

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            epoch_loss += loss.numpy()
            epoch_accuracy += calculate_accuracy(batch_y, predictions).numpy()

        # Average loss and accuracy for the epoch
        epoch_loss /= (len(x_train) / batch_size)
        epoch_accuracy /= (len(x_train) / batch_size)

        # Validation loss and accuracy at the end of each epoch
        val_predictions = model(x_val)
        val_loss = loss_object(y_val, val_predictions).numpy()
        val_accuracy = calculate_accuracy(y_val, val_predictions).numpy()

        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')

    return history

training_history = train_model(model, x_train, y_train, x_val, y_val, epochs=12, batch_size=64)

test_predictions = model(x_test)
test_loss = loss_object(y_test, test_predictions).numpy()
test_accuracy = calculate_accuracy(y_test, test_predictions).numpy()

print('Test accuracy:', test_accuracy)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(training_history['loss'], label='Train')
plt.plot(training_history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(training_history['accuracy'], label='Train')
plt.plot(training_history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()