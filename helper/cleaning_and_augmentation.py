import os
import imgaug.augmenters as iaa
import imgaug.augmenters.contrast as iaa_contrast
import numpy as np
import cv2


def clean_folder(ROOT_DIR):
    for folder_name in os.listdir(ROOT_DIR):
        if 'vowels' in folder_name:
            folder_path = os.path.join(ROOT_DIR, folder_name)

            for filenames in os.listdir(folder_path):
                if '(2)' in filenames:
                    filename = os.path.join(folder_path, filenames)
                    os.remove(filename)


def augment_image(image):
    seq = iaa.Sequential([
        iaa.Add((-10, 10), per_channel=0.5),
        iaa.Multiply((0.9, 1.1), per_channel=0.5),
        iaa.Affine(rotate=(-10, 10),
                   shear=(-5, 5)),
        iaa_contrast.LinearContrast((0.8, 1.2), per_channel=0.5),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
    ])

    # Augment the image
    augmented_image = seq.augment_image(image)
    return augmented_image


def augment_images(input_folder, num_images):
    for root, dirs, files in os.walk(input_folder):
        for dir in dirs:
            if 'vowels' in dir:
                num_images = 3

            subfolder_path = os.path.join(root, dir)

            for filename in os.listdir(subfolder_path):
                if filename.endswith(".jpg"):
                    input_path = os.path.join(subfolder_path, filename)
                    image = cv2.imread(input_path)
                    if image is None:
                        continue

                    for i in range(num_images):
                        augmented_image = augment_image(image)

                        mask = np.all(augmented_image == [0, 0, 0], axis=-1)
                        augmented_image[mask] = [255, 255, 255]

                        output_filename = f"{os.path.splitext(filename)[0]}_{i}.jpg"
                        output_path = os.path.join(subfolder_path, output_filename)
                        cv2.imwrite(output_path, augmented_image)


ROOT_DIR = 'data/Aksara-Sunda'
NUM_IMAGES = 4
clean_folder(ROOT_DIR)
augment_images(ROOT_DIR, NUM_IMAGES)
