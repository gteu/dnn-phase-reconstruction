import librosa
import numpy as np
import librosa.display
import cmath
import glob
import os
import random
np.set_printoptions(threshold=np.inf)

from src import utils

prepare_feats = True
norm_feats = True

data_shuffle = True

ab_dim = 1285

inp_stats_file = "data/input_MVN_1285.norm"

# from .wav to .ab(magnitude of frequency) and .ph(phase of frequency)
if prepare_feats:
    wavs = glob.glob("data/drums/*/*.wav")
    num_of_wavs = len(wavs)

    if data_shuffle:
        random.shuffle(wavs)

    # create file_id_list.scp
    fid = open("file_id_list.scp", mode="w")

    for i in range(num_of_wavs):
        # wav_name = os.path.splitext(os.path.basename(wavs[i]))[0]
        # fid.write(wav_name)
        fid.write(str(i))
        if not i == num_of_wavs - 1:
            fid.write("\n")

    fid.close()

    # create .ab and .ph
    for i in range(num_of_wavs):
        wav = wavs[i]
        # wav_name = os.path.splitext(os.path.basename(wavs[i]))[0]
        print(i, wav)
        y, sr = librosa.load(wav, sr=16000)
        D = librosa.stft(y, n_fft=512, hop_length=80, win_length=400)
        ab = np.abs(D)
        ph = np.angle(D)

        joint_ab = np.empty((ab.shape[0] * 5, ab.shape[1]), dtype=np.float32)

        for j in range(ab.shape[1]):
            temp_ab = np.zeros((ab.shape[0], 5), dtype=np.float32)
            for k in range(-2, 3):
                if j + k < 0:
                    temp_ab[:,2+k] = ab[:,0]
                elif j + k > ab.shape[1] - 1:
                    temp_ab[:,2+k] = ab[:,ab.shape[1] - 1]
                else:
                    temp_ab[:,2+k] = ab[:,j+k]

            joint_ab[:,j] = np.concatenate([temp_ab[:,0],temp_ab[:,1],temp_ab[:,2],temp_ab[:,3],temp_ab[:,4]], axis=0)

        # ab_path = os.path.join("data/abs", wav_name + ".ab")
        # ph_path = os.path.join("data/phase", wav_name + ".ph")
        ab_path = os.path.join("data/abs", str(i) + ".ab")
        ph_path = os.path.join("data/phase", str(i) + ".ph")
        joint_ab.tofile(ab_path)
        ph.tofile(ph_path)


# normalize input data(.ab) to have zero-mean unit-variance
if norm_feats:
    ab_files = glob.glob("data/abs/*.ab")
    num_of_ab = len(ab_files)

    ab_dir = "data/abs"
    norm_ab_dir = "data/norm_abs"
    if not os.path.exists(norm_ab_dir):
        os.makedirs(norm_ab_dir)

    ab_file_id_list = []
    fid = open("file_id_list.scp", mode="r")
    for line in fid.readlines():
        line = line.strip()
        if len(line) < 1:
            continue
        ab_file_id_list.append(line)
    fid.close()

    ab_file_list = utils.prepare_file_path_list(ab_file_id_list, ab_dir, ".ab")
    all_ab = utils.read_data_from_one_list(ab_file_list, ab_dim)

    inp_scaler = utils.compute_norm_stats(all_ab, inp_stats_file)

    for i in range(num_of_ab):
        ab = np.fromfile(ab_file_list[i], dtype=np.float32)
        ab = np.reshape(ab, (ab_dim, -1))
        ab = ab.T

        norm_ab = inp_scaler.transform(ab)
        norm_ab_path = os.path.join(norm_ab_dir, os.path.basename(ab_file_list[i]))
        norm_ab = norm_ab.T

        norm_ab.tofile(norm_ab_path)
