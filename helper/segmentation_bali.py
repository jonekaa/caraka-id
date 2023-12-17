import cv2
import numpy as np
import tensorflow as tf


def load_model(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


def preprocess_input_image(image):
    resized_image = cv2.resize(image, (150, 150))
    if resized_image.shape[-1] == 1:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)

    return resized_image


def predict_character(model, character_image, class_labels):
    input_image = preprocess_input_image(character_image)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image, axis=0)

    predictions = model.predict(input_image)
    predicted_label_index = np.argmax(predictions)

    return class_labels[predicted_label_index]


def predict_characters(model, image_path, list_of_boxes, class_labels):
    image = cv2.imread(image_path)

    predicted_labels = []
    for box in list_of_boxes:
        x1, y1, x2, y2 = box
        character_image = image[y1:y2, x1:x2]
        predicted_label = predict_character(model, character_image, class_labels)
        predicted_labels.append(predicted_label)

        print(f'Bounding Box Coordinates: ({x1}, {y1}, {x2}, {y2})')

    return predicted_labels


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

    cv2.imshow('Image with Bounding Box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return boxes


def predict_words(labels):
    for id, label in enumerate(labels):
        print(f'Predicted Class {id}: {label}')

    words = []

    for id, label in enumerate(labels):
        if label == 'Pengangge tengenan - Bisah (h)':
            label = 'h'

        elif label == 'Pengangge tengenan - Cecek (ng)':
            label = 'ng'

        elif label == 'Pengangge tengenan - Surang (r)':
            label = 'r'

        elif label == 'Pengangge tengenan - Adeg Adeg' and id == len(labels):
            label = labels[id - 1].replace(list(labels[id - 1])[1], '')
            words.pop()

        elif label == 'Pengangge tengenan - Adeg Adeg' and id != len(labels):
            label = 'Pengangge suara - Taleng'

        elif label == 'Pengangge suara - Suku':
            label = labels[id - 1].replace(list(labels[id - 1])[1], 'u')
            words.pop()

        elif label == 'Pengangge suara - Ulu':
            label = labels[id - 1].replace(list(labels[id - 1])[1], 'i')
            words.pop()

        elif label == 'Pengangge suara - Pepet':
            label = labels[id + 1].replace(list(labels[id + 1])[1], 'ee')
            labels[id + 1] = ''

        if label == 'Pengangge suara - Taleng':
            label = labels[id + 1].replace(list(labels[id + 1])[1], 'e')
            labels[id + 1] = ''

        words.append(label)

    return ''.join(word for word in words).lower()


if __name__ == '__main__':
    MODEL_PATH = '../models/model_bali_conv_2_99_99.h5'
    IMAGE_PATH = '../test_images/Aksara_Bali/bali-pergi_bersama_kadek.png'

    CLASS_LABELS = ['Ba', 'Ca', 'Da', 'Ga', 'Ha',
                    'Ja', 'Ka', 'La', 'Ma', 'Na',
                    'Nga', 'Nya', 'Pa',
                    'Pengangge suara - Pepet',
                    'Pengangge suara - Suku',
                    'Pengangge suara - Taleng',
                    'Pengangge suara - Taleng Tedong',
                    'Pengangge suara - Tedong',
                    'Pengangge suara - Ulu',
                    'Pengangge tengenan - Adeg Adeg',
                    'Pengangge tengenan - Bisah (h)',
                    'Pengangge tengenan - Cecek (ng)',
                    'Pengangge tengenan - Surang (r)',
                    'Ra', 'Sa', 'Ta', 'Wa', 'Ya']

    model = load_model(MODEL_PATH)

    list_of_boxes = segment_characters(IMAGE_PATH)
    predicted_labels = predict_characters(model, IMAGE_PATH, list_of_boxes, CLASS_LABELS)

    predicted_words = predict_words(predicted_labels)
    print(predicted_words)
