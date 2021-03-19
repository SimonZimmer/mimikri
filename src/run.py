import os
import dataset_preparation
import autoencoder
import config
import sys
import librosa
import ddsp.processors
from ddsp.training import (data, trainers)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if 'absl.logging' in sys.modules:
    import absl.logging
    absl.logging.set_verbosity('info')
    absl.logging.set_stderrthreshold('info')


dataset_preparation.clean_dataset()
dataset_preparation.convert_to_tfrecord()
dataset_preparation.extract_statistics()

model, strategy = autoencoder.build()
trainer = trainers.Trainer(model, strategy, learning_rate=1e-3)
data_provider = ddsp.training.data.TFRecordProvider(config.dataset_tfrecord_filepattern)
autoencoder.train(data_provider, trainer)
autoencoder.sample(data_provider, model, trainer)

audio, sr = librosa.load('../input_audio/taiko.wav', sr=config.sample_rate)
encoder, dataset_stats = autoencoder.load(audio)
autoencoder.sample(data_provider, encoder, trainer)
