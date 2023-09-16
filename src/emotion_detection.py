from tensorflow.keras.models import load_model
import tf2onnx
import tensorflow as tf
from keras.models import model_from_json

def model_to_onnx():
    with open('emotion_model.json', 'r') as f:
        model = model_from_json(f.read())

    model.load_weights('emotion_model.h5')

    onnx_model, _ = tf2onnx.convert.from_keras(model)

    with open("emotion_detection.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
