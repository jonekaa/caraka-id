import tensorflow as tf


def tflite_converter(MODEL_PATH, AKSARA_TYPE=None):
    model = tf.keras.models.load_model(MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(f'model_{AKSARA_TYPE}.tflite', 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    AKSARA_TYPE = 'bali'

    if AKSARA_TYPE == 'bali':
        MODEL_PATH = '../models/model_bali.h5'
    elif AKSARA_TYPE == 'sunda':
        MODEL_PATH = '../models/model_sunda.h5'
    elif AKSARA_TYPE == 'lampung':
        MODEL_PATH = '../models/model_lampung.h5'
    else:
        raise ValueError('AKSARA NOT IN OUR REACHED')

tflite_converter(MODEL_PATH, AKSARA_TYPE)
