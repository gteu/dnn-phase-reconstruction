import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import cmath
import glob
import os
import random

prepare_feats = True
norm_feats = True

data_shuffle = True

inp_dim = 513

# from .wav to .ab(magnitude of frequency) and .ph(phase of frequency)
if prepare_feats:
    wavs = glob.glob("data/drums/*/*.wav")

    if data_shuffle:
        random.shuffle(wavs)

    for i in range(len(wavs)):
        wav = wavs[i]
        print(wav)
        y, sr = librosa.load(wav, sr=22050)
        D = librosa.stft(y, n_fft=1024, hop_length=128)
        ab = np.abs(D)
        phase = np.angle(D)
        ab_path = os.path.join("data/abs", str(i) + ".ab")
        phase_path = os.path.join("data/phase", str(i) + ".ph")
        ab.tofile(ab_path)
        phase.tofile(phase_path)


# normalize input data to have zero-mean unit-variance
if norm_feats:
    ab_files = glob.glob("data/abs/*.ab")
    num_of_ab = len(ab_files)

    norm_ab_dir = "data/norm_abs"
    if not os.path.exists(norm_ab_dir):
        os.makedirs(norm_ab_dir)

    # for f in range(num_of_ab):
    #     file_x = np.fromfile(inp_file_list[i], dtype=np.float32)
    #     file_x = np.reshape(file_x, (inp_dim, -1))
    #     file_x = file_x.T
