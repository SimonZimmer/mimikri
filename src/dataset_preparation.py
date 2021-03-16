import os
import config


def clean_dataset():
    audio_files = os.listdir(config.audio_dir)
    for file in audio_files:
        if os.path.splitext(file)[1] != '.wav':
            print(f"found non-wav file '{file}'. Removing...")
            os.remove(os.path.join(config.audio_dir, file))
    for file in os.listdir(config.train_tfrecord_dir):
        tfrecord = os.path.join(config.train_tfrecord_dir, file)
        try:
            os.remove(tfrecord)
        except IsADirectoryError:
            os.removedirs(tfrecord)


def convert_to_tfrecord():
    clean_dataset()
    os.system(f"python ../ThirdParty/ddsp/ddsp/training/data_preparation/ddsp_prepare_tfrecord.py \
    --input_audio_filepatterns={config.audio_filepattern} \
    --output_tfrecord_path={config.train_tfrecord} \
    --num_shards=10 \
    --alsologtostderr")
