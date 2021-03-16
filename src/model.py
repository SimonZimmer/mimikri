import os
import sys
import soundfile
os.system("ls ../ThirdParty/ddsp")
sys.path.append('../ThirdParty/ddsp/')
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import time
import config
import dataset_preparation
import ddsp
from ddsp.training import (data, decoders, models, preprocessing,
                           train_util, eval_util, trainers)


def write_audio_file(filepath, audio):
    encoded_audio = np.swapaxes(np.array(audio), 0, 1)
    soundfile.write(filepath, encoded_audio, config.sample_rate)


def create_model(processor_group, spectral_loss):
    preprocessor = preprocessing.F0LoudnessPreprocessor(time_steps=config.time_steps)

    decoder = decoders.RnnFcDecoder(rnn_channels=256,
                                    rnn_type='gru',
                                    ch=256,
                                    layers_per_stack=1,
                                    input_keys=('ld_scaled', 'f0_scaled'),
                                    output_splits=(('amps', 1),
                                                   ('harmonic_distribution', 45),
                                                   ('noise_magnitudes', 45)))

    strategy = train_util.get_strategy()
    with strategy.scope():
        model = models.Autoencoder(preprocessor=preprocessor,
                                   encoder=None,
                                   decoder=decoder,
                                   processor_group=processor_group,
                                   losses=[spectral_loss])

    return model, strategy


def create_processor_group():
    harmonic = ddsp.synths.Harmonic(n_samples=config.n_samples,
                                    sample_rate=config.sample_rate,
                                    name='harmonic')

    noise = ddsp.synths.FilteredNoise(window_size=0,
                                      initial_bias=-10.0,
                                      name='noise')
    add = ddsp.processors.Add(name='add')

    dag = [(harmonic, ['amps', 'harmonic_distribution', 'f0_hz']),
           (noise, ['noise_magnitudes']),
           (add, ['noise/signal', 'harmonic/signal'])]

    return dag


def train(data_provider, trainer):
    train_util.train(data_provider, trainer,
                     config.batch_size, config.num_epochs,
                     config.steps_per_summary,
                     config.steps_per_save,
                     config.save_dir,
                     config.restore_dir,
                     config.early_stop_loss_value)


def sample(data_provider, model):
    dataset = data_provider.get_batch(config.batch_size, shuffle=True, repeats=-1)
    dataset = trainer.distribute_dataset(dataset)
    dataset_iter = iter(dataset)

    start_time = time.time()
    controls = model(next(dataset_iter))
    audio_gen = model.get_audio_from_outputs(controls)
    audio_noise = controls['noise']['signal']
    print('Prediction took %.1f seconds' % (time.time() - start_time))

    write_audio_file(os.path.join(config.audio_out_dir, 'audio_resynthesized.wav'), audio_gen)
    write_audio_file(os.path.join(config.audio_out_dir, 'audio_noise_modelled.wav'), audio_noise)


dataset_preparation.convert_to_tfrecord()

processor_group = ddsp.processors.ProcessorGroup(dag=create_processor_group(),
                                                 name='processor_group')
spectral_loss = ddsp.losses.SpectralLoss(loss_type='L1',
                                         mag_weight=1.0,
                                         logmag_weight=1.0)

model, strategy = create_model(processor_group, spectral_loss)
trainer = trainers.Trainer(model, strategy, learning_rate=1e-3)
data_provider = ddsp.training.data.TFRecordProvider(config.train_tfrecord_filepattern)
train(data_provider, trainer)
sample(data_provider, model)
