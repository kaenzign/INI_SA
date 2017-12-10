from keras import backend as K
import numpy as np
from keras.models import load_model

MODEL_NAME = 'weights.09-0.39.h5'
model_path = './model/DVS63_EVTACC_10E/' + MODEL_NAME

model = load_model(model_path)

model.summary()