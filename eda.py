import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

data_df = pd.read_csv('data/original/Data_Entry_2017.csv')

PATH = 'images/original/images'

def load_image(fileName, subFolder):
    try:
      folder = PATH + subFolder + "/"  # Mac
      filePath = folder+ fileName
      img = cv2.imread(filePath)
      return img
    except:
      return np.NaN

def extract_labels(file_name):
    pass