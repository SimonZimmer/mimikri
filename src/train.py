import argparse

import ddsp.processors
from ddsp.training import (data, trainers)

import dataset_preparation
import autoencoder
import config


def main():
    def train_new_model():
        dataset_preparation.clean_dataset()
        dataset_preparation.convert_to_tfrecord()
        dataset_preparation.extract_statistics()
        model, strategy = autoencoder.build()
        trainer = trainers.Trainer(model, strategy, learning_rate=1e-3)
        data_provider = ddsp.training.data.TFRecordProvider(config.dataset_tfrecord_filepattern)
        autoencoder.train(data_provider, trainer)

    parser = argparse.ArgumentParser(description="trains a new mimicry model "
                                                 "or performs timbre transfer on a given audio sample")

    parser.add_argument("-ns", "--n_samples", default=config.n_samples,
                        help="number of samples for each training sample")
    parser.add_argument("-sr", "--sample_rate", default=config.sample_rate,
                        help="audio sample rate")
    parser.add_argument("-st", "--n_steps", default=config.n_steps,
                        help="number of steps to train")
    parser.add_argument("-bs", "--batch_size", default=config.batch_size,
                        help="size of training batch")
    parser.add_argument("-si", "--steps_per_save", default=config.steps_per_save,
                        help="saving interval in number of steps")
    parser.add_argument("-es", "--early_stop", default=config.early_stop_loss_value,
                        help="spectral loss at which training stops")

    args = parser.parse_args()

    config.n_samples = args.n_samples
    config.sample_rate = args.sample_rate
    config.n_steps = args.n_steps
    config.batch_size = args.batch_size
    config.steps_per_summary = args.steps_per_save
    config.steps_per_save = args.steps_per_save
    config.early_stop_loss_value = args.early_stop

    train_new_model()


if __name__ == "__main__":
    main()
