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


def train_new_model():
    dataset_preparation.clean_dataset()
    dataset_preparation.convert_to_tfrecord()
    dataset_preparation.extract_statistics()
    model, strategy = autoencoder.build()
    trainer = trainers.Trainer(model, strategy, learning_rate=1e-3)
    data_provider = ddsp.training.data.TFRecordProvider(config.dataset_tfrecord_filepattern)
    autoencoder.train(data_provider, trainer)


def timbre_transfer(audio_in_path):
    audio, sr = librosa.load(audio_in_path, sr=config.sample_rate)
    trained_model, audio_features = autoencoder.load(audio)
    autoencoder.sample(trained_model, audio_features)


train_new_model()
