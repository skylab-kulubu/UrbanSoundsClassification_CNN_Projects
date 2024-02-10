import os
import librosa
import numpy as np
import shutil
from PIL import Image


def create_spectrogram(y):
    """
        convert audio to spectogram using librosa lib
    """
    spec = librosa.feature.melspectrogram(y=y)
    spec_conv = librosa.amplitude_to_db(spec, ref=np.max)
    return spec_conv


def save_image(spec, path, size=(128, 128), fmt='png'):
    """
        alter and save images using pillow lib
    """
    spec_normalized = (spec - spec.min()) / (spec.max() - spec.min())
    spec_scaled = (spec_normalized * 255).astype(np.uint8)
    
    image = Image.fromarray(spec_scaled)
    image = image.convert("L")
    image = image.resize(size, Image.Resampling.LANCZOS)

    image.save(path, format=fmt)


def convert_audio_to_spectrogram(audio_path, target_folder):
    """
        using predifend funcitons...
        convert audio to spectogram ,
        alter spectogram,
        save image to class folder
    """
    print(f"Processing: {audio_path}")
    y, _ = librosa.load(audio_path)
    spec = create_spectrogram(y)
    target_path = os.path.join(target_folder, os.path.basename(audio_path).replace(".wav", ".png"))
    save_image(spec, target_path)


def all_operations(source_path, class_id, audio_target_directory, image_target_directory):
    """
        call helper funcitons
    """
    audio_target_folder = os.path.join(audio_target_directory, class_id)
    image_target_folder = os.path.join(image_target_directory, class_id)
    os.makedirs(audio_target_folder, exist_ok=True)
    os.makedirs(image_target_folder, exist_ok=True)

    audio_target_path = os.path.join(audio_target_folder, os.path.basename(source_path))
    shutil.copy(source_path, audio_target_path)

    convert_audio_to_spectrogram(source_path, image_target_folder)


def organize_files_by_class(source_directory, audio_target_directory, image_target_directory):
    """
        using predifend functions...
        put every audio file into his corresponding class folder,
        using organized audio files create spectrograms,
        alter created spectograms,
        save them into class image folders
    """
    for folder in os.listdir(source_directory):
        fold_path = os.path.join(source_directory, folder)
        if os.path.isdir(fold_path):
            for filename in os.listdir(fold_path):
                if filename.endswith(".wav"):
                    class_id = str(filename.split("-")[1])
                    if class_id:
                        source_path = os.path.join(fold_path, filename)
                        all_operations(source_path, class_id, audio_target_directory, image_target_directory)


if __name__ == "__main__":
    source_directory = './UrbanSound8K/audio'
    audio_target_directory = './UrbanSound'
    image_target_directory = './UrbanImage'
    organize_files_by_class(source_directory, audio_target_directory, image_target_directory)
