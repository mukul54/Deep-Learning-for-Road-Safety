import cv2
import numpy as np
import dlib
import glob
from scipy.spatial import distance
from imutils import face_utils
from keras.models import load_model
import time
from pygame import mixer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

driver_model = load_model('weights_resnet50.h5')

