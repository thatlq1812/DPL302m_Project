# Python library
import os
import zipfile
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
import pickle as pkl
import yaml
import albumentations as A

from glob import glob
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# Sklearn library
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Pytorch library
import torch

from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms

# CV2 library
import cv2



# Read yaml file
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Necessary variables

model_name = 'densenet'
num_classes = config['hyperparameters']['num_classes']
feature_extract = config['hyperparameters']['feature_extract']
epochs = config['hyperparameters']['epochs']

# Initialize the model

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    

