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

strategy = train_util.get_strategy()


# Create Neural Networks.
preprocessor = preprocessing.F0LoudnessPreprocessor(time_steps=config.TIME_STEPS)

decoder = decoders.RnnFcDecoder(rnn_channels = 256,
                                rnn_type = 'gru',
                                ch = 256,
                                layers_per_stack = 1,
                                input_keys = ('ld_scaled', 'f0_scaled'),
                                output_splits = (('amps', 1),
                                                 ('harmonic_distribution', 45),
                                                 ('noise_magnitudes', 45)))

# Create Processors.
harmonic = ddsp.synths.Harmonic(n_samples=n_samples,
                                sample_rate=sample_rate,
                                name='harmonic')

noise = ddsp.synths.FilteredNoise(window_size=0,
                                  initial_bias=-10.0,
                                  name='noise')
add = ddsp.processors.Add(name='add')

# Create ProcessorGroup.
dag = [(harmonic, ['amps', 'harmonic_distribution', 'f0_hz']),
       (noise, ['noise_magnitudes']),
       (add, ['noise/signal', 'harmonic/signal'])]

processor_group = ddsp.processors.ProcessorGroup(dag=dag,
                                                 name='processor_group')


# Loss_functions
spectral_loss = ddsp.losses.SpectralLoss(loss_type='L1',
                                         mag_weight=1.0,
                                         logmag_weight=1.0)

with strategy.scope():
    # Put it together in a model.
    model = models.Autoencoder(preprocessor=preprocessor,
                               encoder=None,
                               decoder=decoder,
                               processor_group=processor_group,
                               losses=[spectral_loss])
    trainer = trainers.Trainer(model, strategy, learning_rate=1e-3)


dataset = trainer.distribute_dataset(dataset)
trainer.build(next(iter(dataset)))
