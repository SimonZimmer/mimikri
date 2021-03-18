import os
import soundfile
import numpy as np
import time
import config
import ddsp
import gin
import pickle
import absl.logging as logging
import tensorflow as tf
from ddsp.training import (data, decoders, models, preprocessing, train_util)


def write_audio_file(filepath, audio):
    encoded_audio = np.swapaxes(np.array(audio), 0, 1)
    soundfile.write(filepath, encoded_audio, config.sample_rate)


def build():
    processor_group = ddsp.processors.ProcessorGroup(dag=create_processor_group(),
                                                     name='processor_group')
    spectral_loss = ddsp.losses.SpectralLoss(loss_type='L1',
                                             mag_weight=1.0,
                                             logmag_weight=1.0)
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


def parse_gin():
    # Parse gin configs, later calls override earlier ones.
    with gin.unlock_config():
        # Optimization defaults.
        opt_default = 'base.gin'
        gin.parse_config_file(os.path.join('../ThirdParty', 'ddsp', 'ddsp', 'training',
                                           'gin', 'optimization', opt_default))
        eval_default = 'basic.gin'
        gin.parse_config_file(os.path.join('../ThirdParty', 'ddsp', 'ddsp', 'training',
                                           'gin', 'eval', eval_default))

        # Load operative_config if it exists (model has already trained).
        operative_config = train_util.get_latest_operative_config(config.save_dir)
        if tf.io.gfile.exists(operative_config):
            logging.info('Using operative config: %s', operative_config)
            gin.parse_config_file(operative_config, skip_unknown=True)


def train(data_provider, trainer):
    restore_dir = os.path.expanduser(config.restore_dir)
    save_dir = os.path.expanduser(config.save_dir)
    restore_dir = save_dir if not restore_dir else restore_dir
    logging.info('Restore Dir: %s', restore_dir)
    logging.info('Save Dir: %s', save_dir)

    parse_gin()
    train_util.gin_register_keras_layers()

    train_util.train(data_provider, trainer,
                     config.batch_size, config.num_steps,
                     config.steps_per_summary,
                     config.steps_per_save,
                     config.save_dir,
                     config.restore_dir,
                     config.early_stop_loss_value)
    trainer.model.save_weights(os.path.join(config.save_dir, 'my_model_weights.h5'))


def load(audio):
    gin_file = os.path.join(config.save_dir, 'operative_config-0.gin')

    # Load the dataset statistics.
    datasets_stats = None
    dataset_stats_file = os.path.join(config.train_statistics_dir, 'dataset_statistics.pkl')
    print(f'Loading dataset statistics from {dataset_stats_file}')
    try:
        if tf.io.gfile.exists(dataset_stats_file):
            with tf.io.gfile.GFile(dataset_stats_file, 'rb') as f:
                dataset_stats = pickle.load(f)
    except Exception as err:
        print('Loading dataset statistics from pickle failed: {}.'.format(err))

    # Parse gin config
    with gin.unlock_config():
        gin.parse_config_file(gin_file, skip_unknown=True)

    # Assumes only one checkpoint in the folder, 'ckpt-[iter]`
    ckpt_files = [f for f in tf.io.gfile.listdir(config.save_dir) if 'ckpt' in f]
    ckpt_name = ckpt_files[0].split('.')[0]
    ckpt = os.path.join(config.save_dir, ckpt_name)

    # Ensure dimensions and sampling rates are equal
    time_steps_train = config.time_steps
    n_samples_train = config.n_samples
    hop_size = int(n_samples_train / time_steps_train)

    time_steps = int(config.n_samples / hop_size)
    n_samples = time_steps * hop_size

    gin_params = [
        'Harmonic.n_samples = {}'.format(n_samples),
        'FilteredNoise.n_samples = {}'.format(n_samples),
        'F0LoudnessPreprocessor.time_steps = {}'.format(time_steps),
        'oscillator_bank.use_angular_cumsum = True',  # Avoids cumsum accumulation errors
    ]

    with gin.unlock_config():
        gin.parse_config(gin_params)

    # Compute features
    start_time = time.time()
    audio_features = ddsp.training.metrics.compute_audio_features(audio)
    audio_features['loudness_db'] = audio_features['loudness_db'].astype(np.float32)
    print('Audio features took %.1f seconds' % (time.time() - start_time))

    # Trim all input vectors to correct lengths
    for key in ['f0_hz', 'f0_confidence', 'loudness_db']:
        audio_features[key] = audio_features[key][:time_steps]
    audio_features['audio'] = audio_features['audio'][:n_samples]

    # Set up the model just to predict audio given new conditioning
    model = build()[0]

    # Build model by running a batch through it.
    start_time = time.time()
    _ = model(audio_features, training=False)

    model.load_weights(os.path.join(config.save_dir, 'my_model_weights.h5'))
    print('Restoring model took %.1f seconds' % (time.time() - start_time))

    return model, dataset_stats


def sample(data_provider, model, trainer):
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