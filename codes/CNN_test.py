import os
import numpy as np
import tensorflow as tf
import cv2


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (150, 150))
    image = np.expand_dims(image, axis=0)

    return image


def predict_image(model_path, image_path, class_labels):
    model = tf.keras.models.load_model(model_path)
    image = preprocess_image(image_path)
    predictions = model.predict(image)
    prediction = np.argmax(predictions)
    predicted_class = class_labels[prediction]

    return predicted_class


def predict_folder(model_path, folder_path, class_labels):
    model = tf.keras.models.load_model(model_path)
    total_predictions = 0
    correct_predictions = 0

    for category in os.listdir(folder_path):
        category_path = os.path.join(folder_path, category)

        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                if filename.endswith('.jpg'):
                    test_image_path = os.path.join(category_path, filename)
                    input_image = preprocess_image(test_image_path)
                    predictions = model.predict(input_image)
                    prediction = np.argmax(predictions)
                    predicted_class = class_labels[prediction]
                    # predicted_class = class_labels[tf.argmax(predictions, axis=1).numpy()[0]]

                    prediction_is_correct = str(predicted_class).strip() == str(category).strip()
                    total_predictions += 1
                    correct_predictions += int(prediction_is_correct)

                    print(f'Image: {filename}, Predicted Class: {predicted_class}, True Class: {category},'
                          f'Prediction is: {prediction_is_correct}')

    print(f'\nOverall Accuracy: {((correct_predictions / total_predictions) * 100):.2f}%')


# model_path = '../models/model_trained_efficientNetV2B09897.h5' 99.53
# 99.22
# model_path = "../models/model_trained_efficientNetV2B09998A.h5"
# 99.53
model_path = '../models/model_trained_efficientNetV2B09997.h5'

# buat predict subfolder di satu folder
folder_path = '../data/Aksara_Sunda/valid_labeled/'

# kalau predict image pake ini aja
image_path = '../data/COBA/sa.jpg'

class_labels = ['a', 'ae', 'ba', 'ca', 'da', 'e', 'eu', 'fa', 'ga', 'ha', 'i',
                'ja', 'ka', 'la', 'ma', 'na', 'nga', 'nya', 'o', 'pa', 'qa', 'ra',
                'sa', 'ta', 'u', 'va', 'vowels_e', 'vowels_ee', 'vowels_eu', 'vowels_h',
                'vowels_i', 'vowels_la', 'vowels_ng', 'vowels_o', 'vowels_r', 'vowels_u',
                'vowels_x', 'vowels_ya', 'wa', 'xa', 'ya', 'za']
# predict 1 folder
# predict_folder(model_path, folder_path, class_labels)

# predict 1 image
print(predict_image(model_path, image_path, class_labels))
