import os

dataset_audio_dir = os.path.join(os.pardir, 'dataset', 'audio_test')
dataset_audio_filepattern = dataset_audio_dir + '/*'
dataset_tfrecord_dir = os.path.join(os.pardir, 'dataset', 'tfrecords')
dataset_tfrecord = os.path.join(dataset_tfrecord_dir, 'train.tfrecord')
dataset_tfrecord_filepattern = dataset_tfrecord + '*'

n_samples = 64000
sample_rate = 16000
frame_rate = 16000
time_steps = 1000

num_steps = 30000
batch_size = 1
steps_per_summary = 1
steps_per_save = 1
early_stop_loss_value = 5

dataset_statistics_dir = os.path.join(os.pardir, 'training_artefacts', 'dataset_statistics')
saved_models_dir = os.path.join(os.pardir, 'training_artefacts', 'saved_models', 'checkpoints')
save_dir = os.path.join(os.pardir, 'training_artefacts', 'saved_models')
restore_dir = os.path.join(os.pardir, 'training_artefacts', 'restoration')
audio_out_dir = os.path.join(os.pardir, 'training_artefacts', 'generated_audio')
