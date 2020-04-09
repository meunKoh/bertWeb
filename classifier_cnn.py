import numpy as np
#from keras.models import load_model
from tensorflow import keras

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

class Predictor(object):

    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)

    def predict(self, text):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(text)
        return self.model.predict(pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=1024))