import os
import sys
os.system("ls ../ThirdParty/ddsp")
sys.path.append('../ThirdParty/ddsp/')
import warnings
warnings.filterwarnings("ignore")
import time
import config
import ddsp
from ddsp.training import (data, decoders, encoders, models, preprocessing,
                          train_util, trainers)
import gin
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

sample_rate = 16000

data_provider = ddsp.training.data.TFRecordProvider(config.TRAIN_TFRECORD_FILEPATTERN)
dataset = data_provider.get_dataset(shuffle=False)
dataset = data_provider.get_batch(batch_size=1, shuffle=False).take(1).repeat()
batch = next(iter(dataset))
audio = batch['audio']
n_samples = audio.shape[1]
print(n_samples)