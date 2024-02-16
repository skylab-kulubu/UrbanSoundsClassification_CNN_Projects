import os
import cv2
import random
import librosa
import numpy as np
import matplotlib.pyplot as plt

def create_spectrogram(y):
    """
    Create spectrograms from audio signals using librosa library.
    """
    spec = librosa.feature.melspectrogram(y=y)
    spec_conv = librosa.amplitude_to_db(spec, ref=np.max)
    return spec_conv

def audio_to_spectrogram(audio_folder_path, image_folder_path):
    """
    Convert audio files to spectrograms and save them in the specified output folder.
    """
    for root, _, files in os.walk(audio_folder_path):
        for file in files:
            if file.endswith(".wav"):
                filepath = os.path.join(root, file)
                label = int(file.split("-")[1])
                output_folder = os.path.join(image_folder_path, str(label))
                os.makedirs(output_folder, exist_ok=True)
                output_filename = file.replace(".wav", ".png")
                output_path = os.path.join(output_folder, output_filename)
                y, sr = librosa.load(filepath, sr=None)
                spec = create_spectrogram(y)
                spec_resized = cv2.resize(spec, (400, 300))
                spec_flipped = np.flipud(spec_resized)
                plt.imsave(output_path, spec_flipped, cmap='magma')

def preprocess_image(image, size):
    """
    Preprocess the image: convert to grayscale, resize, and normalize.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, size)
    normalized_image = cv2.normalize(resized_image, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return normalized_image

def load_images_from_folder(folder_path, label):
    """
    Load images from the given folder path and assign labels.
    """
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            processed_img = preprocess_image(img, (128, 128))
            images.append(processed_img)
            labels.append(label)
    return images, labels

def preprocess_data(image_folder_path):
    """
    Preprocess data by loading images and corresponding labels.
    """
    X = []
    y = []
    for i in range(10):
        class_folder = os.path.join(image_folder_path, str(i))
        images, labels = load_images_from_folder(class_folder, i)
        X.extend(images)
        y.extend(labels)
    X = np.array(X)
    y = np.array(y)
    return X, y

def select_random_audio_file(dataset_path):
    """
    Selects a random audio file from the dataset and returns its file path along with the corresponding class.
    """
    folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    random_folder = random.choice(folders)
    random_folder_path = os.path.join(dataset_path, random_folder)
    audio_files = [f for f in os.listdir(random_folder_path) if f.endswith('.wav')]
    random_audio = random.choice(audio_files)
    audio_file_path = os.path.join(random_folder_path, random_audio)
    class_index_str = random_audio.split("-")[1]
    class_index = int(class_index_str)
    return audio_file_path, class_index
