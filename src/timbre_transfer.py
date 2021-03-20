import argparse
import os

import librosa

import autoencoder
import config


def main():
    def timbre_transfer(audio_in_path):
        audio, sr = librosa.load(audio_in_path, sr=config.sample_rate)
        trained_model, audio_features = autoencoder.load(audio)
        autoencoder.sample(trained_model, audio_features)

    parser = argparse.ArgumentParser(description="performs timbre transfer using a saved model and an audio file")
    parser.add_argument("-m", "--model", default=os.path.join(config.saved_models_dir, "weights.h5"),
                        help="pretrained model weights to use")
    parser.add_argument("-a", "--audio_file", default="../input_audio/kick.wav",
                        help="audio file path to use")
    args = parser.parse_args()

    timbre_transfer(args.audio_file)


if __name__ == "__main__":
    main()
