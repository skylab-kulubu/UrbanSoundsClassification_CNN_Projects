from sklearn.model_selection import train_test_split
import cv2
import os
import numpy as np

x = []
y = []

main_folder = sorted(os.listdir("spectrograms"))

for folder in main_folder:
    path_class = os.path.join("spectrograms", folder)

    for file in os.listdir(path_class):
        path_file = os.path.join(path_class, file)
        
        image = cv2.imread(path_file, cv2.IMREAD_GRAYSCALE) # grayscale
        image = cv2.resize(image, (128, 128)) # resize
        image = image / 255.0 # normalleştirme

        x.append(image)
        y.append(int(folder))

x = np.array(x)
y = np.array(y)

x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

x_train = x_train.reshape(x_train.shape + (1,))
x_val = x_val.reshape(x_val.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

np.save("X_train.npy", x_train)
np.save("y_train.npy", y_train)
np.save("X_val.npy", x_val)
np.save("y_val.npy", y_val)
np.save("X_test.npy", x_test)
np.save("y_test.npy", y_test)