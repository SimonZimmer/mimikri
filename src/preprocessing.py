import os
import glob
import config


def convert_to_tfrecord():
    dataset_dir = os.path.join('dataset')
    dataset_files = glob.glob(dataset_dir + '/*')

    if os.listdir('dataset/tfrecords'):
        return

    os.system(f"python ThirdParty/ddsp/ddsp/training/data_preparation/ddsp_prepare_tfrecord.py \
    --input_audio_filepatterns={config.audio_filepattern} \
    --output_tfrecord_path={config.train_tfrecord} \
    --num_shards=10 \
    --alsologtostderr")


convert_to_tfrecord()
