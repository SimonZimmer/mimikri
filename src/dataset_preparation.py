import os
import subprocess

import ddsp.training
from ddsp.colab import colab_utils

import config


def clean_dataset():
    audio_files = os.listdir(config.dataset_audio_dir)
    for file in audio_files:
        if os.path.splitext(file)[1] != '.wav':
            print(f"found non-wav file '{file}'. Removing...")
            os.remove(os.path.join(config.dataset_audio_dir, file))
    for file in os.listdir(config.dataset_tfrecord_dir):
        tfrecord = os.path.join(config.dataset_tfrecord_dir, file)
        try:
            os.remove(tfrecord)
        except IsADirectoryError:
            os.removedirs(tfrecord)


def convert_to_tfrecord():
    command = [
        "python",
        "../ThirdParty/ddsp/ddsp/training/data_preparation/ddsp_prepare_tfrecord.py",
        f"--input_audio_filepatterns={config.dataset_audio_filepattern}",
        f"--output_tfrecord_path={config.dataset_tfrecord}",
        "--alsologtostderr"
    ]
    subprocess.run(command)


def extract_statistics():
    data_provider = ddsp.training.data.TFRecordProvider(config.dataset_tfrecord_filepattern)
    pickle_file_path = os.path.join(config.dataset_statistics_dir, 'dataset_statistics.pkl')

    colab_utils.save_dataset_statistics(data_provider, pickle_file_path, batch_size=1)
