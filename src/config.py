audio_dir = '../dataset/audio'
audio_filepattern = audio_dir + '/*'
train_tfrecord_dir = '../dataset/tfrecords/'
train_tfrecord = train_tfrecord_dir + 'train.tfrecord'
train_tfrecord_filepattern = train_tfrecord + '*'

audio_out_dir = '../training_artefacts/audio'
num_epochs = 300
batch_size = 1

n_samples = 64000
sample_rate = 16000
time_steps = 1000
