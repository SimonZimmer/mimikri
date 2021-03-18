audio_dir = '../dataset/audio'
audio_filepattern = audio_dir + '/*'
train_tfrecord_dir = '../dataset/tfrecords/'
train_tfrecord = train_tfrecord_dir + 'train.tfrecord'
train_tfrecord_filepattern = train_tfrecord + '*'
train_statistics_dir = '../dataset/statistics'

audio_out_dir = '../training_artefacts/audio'
num_steps = 1
batch_size = 1

n_samples = 64000
sample_rate = 16000
frame_rate = 16000
time_steps = 1000

steps_per_summary = 1
steps_per_save = 1
save_dir = '../training_artefacts/save_dir'
restore_dir = '../training_artefacts/restore_dir'
early_stop_loss_value = 5
