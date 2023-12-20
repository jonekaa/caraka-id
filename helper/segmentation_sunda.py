import cv2
import numpy as np
import tensorflow as tf


def load_your_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


def preprocess_input_image(image, target_size=(150, 150)):
    resized_image = cv2.resize(image, target_size)
    if resized_image.shape[-1] == 1:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)

    return resized_image


def predict_character(model, character_image, target_size=(150, 150)):
    input_image = preprocess_input_image(character_image, target_size)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image, axis=0)

    predictions = model.predict(input_image)
    # print('Predictions:', predictions)

    predicted_label_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_label_index]

    return predicted_label


def segment_characters(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    threshold = cv2.threshold(blur, 0.5, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

    boxes.sort(key=lambda x: (x[0], x[1]))

    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return boxes


def predict_characters(model, image, list_of_boxes):
    predicted_labels = []

    for box in list_of_boxes:
        x1, y1, x2, y2 = box
        character_image = image[y1:y2, x1:x2]
        predicted_label = predict_character(model, character_image)
        predicted_labels.append(predicted_label)

        print(f'Bounding Box Coordinates: ({x1}, {y1}, {x2}, {y2})')

    return predicted_labels


def predict_words(labels):
    for id, label in enumerate(labels):
        print(f'Predicted Class {id}: {label}')

    words = []

    for id, label in enumerate(labels):
        if label == 'sa' and labels[id + 1] == 'sa':
            label = ''

        if label == 'ka' and labels[id + 1] == 'ka':
            label = ''

        if label == 'vowels_o':
            label = labels[id - 1].replace(list(labels[id - 1])[1], 'o')
            words.pop()
        elif label == 'vowels_e':
            labels[id + 1] = labels[id + 1].replace(list(labels[id + 1])[1], 'e')
            label = ''
        elif label == 'vowels_ee':
            labels[id + 1] = labels[id + 1].replace(list(labels[id + 1])[1], 'ee')
            label = ''
        elif label == 'vowels_eu':
            labels[id + 1] = labels[id + 1].replace(list(labels[id + 1])[1], 'eu')
            label = ''
        elif label == 'vowels_i':
            labels[id + 1] = labels[id + 1].replace(list(labels[id + 1])[1], 'i')
            label = ''
        elif label == 'vowels_u':
            label = labels[id - 1].replace(list(labels[id - 1])[1], 'u')
            words.pop()
        elif label == 'vowels_la':
            label = labels[id - 1].replace(list(labels[id - 1])[1], 'la')
            words.pop()
        elif label == 'vowels_ra':
            label = labels[id - 1].replace(list(labels[id - 1])[1], 'ra')
            words.pop()
        elif label == 'vowels_ya':
            label = labels[id - 1].replace(list(labels[id - 1])[1], 'ya')
            words.pop()
        elif label == 'vowels_x':
            label = labels[id - 1].replace(list(labels[id - 1])[1], '')
            words.pop()

        elif label == 'vowels_h':
            label = 'h'
        elif label == 'vowel_r':
            labels[id + 1] = (labels[id + 1]) + 'r'
            label = ''
        elif label == 'vowels_ng':
            labels[id + 1] = (labels[id + 1]) + 'ng'
            label = ''

        words.append(label)

    return ''.join(word for word in words).lower()


class_labels = ['a', 'ba', 'ca', 'da', 'e',
                'ee', 'eu', 'fa', 'ga', 'ha',
                'i', 'ja', 'ka', 'la', 'ma', 'na',
                'nga', 'nya', 'ou', 'pa', 'qa', 'ra',
                'sa', 'ta', 'u', 'va', 'vowels_e',
                'vowels_ee', 'vowels_eu', 'vowels_h',
                'vowels_i', 'vowels_la', 'vowels_ng',
                'vowels_o', 'vowels_r', 'vowels_ra',
                'vowels_u', 'vowels_x', 'vowels_ya',
                'wa', 'xa', 'ya', 'za']

image_path = '../test_images/Aksara_Sunda/bala.jpg'
model_path = '../models/model_sunda_v2.h5'

model = load_your_model(model_path)
image = cv2.imread(image_path)

list_of_boxes = segment_characters(image_path)
predicted_labels = predict_characters(model, image, list_of_boxes)

predicted_words = predict_words(predicted_labels)
print(predicted_words)
