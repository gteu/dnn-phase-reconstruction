import glob
import numpy as np

import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD, Adam, Adagrad

def read_data_from_list(inp_file_list, out_file_list, inp_dim, out_dim):
    temp_x = np.empty((FRAME_BUFFER_SIZE, inp_dim))
    temp_y = np.empty((FRAME_BUFFER_SIZE, out_dim))

    current_index = 0

    assert len(inp_file_list) == len(out_file_list)
    num_of_sounds = len(inp_file_list)

    for i in range(num_of_sounds):
        file_x = np.fromfile(inp_file_list[i], dtype=np.float32)
        file_x = np.reshape(file_x, (inp_dim, -1))
        file_x = file_x.T

        file_y = np.fromfile(out_file_list[i], dtype=np.float32)
        file_y = np.reshape(file_y, (out_dim, -1))
        file_y = file_y.T

        assert file_x.shape[0] == file_y.shape[0]
        frame_number = file_x.shape[0]

        temp_x[current_index:current_index+frame_number] = file_x[0:frame_number]
        temp_y[current_index:current_index+frame_number] = file_y[0:frame_number]

        current_index += frame_number

    temp_x = temp_x[0:current_index]
    temp_y = temp_y[0:current_index]

    return temp_x, temp_y

def prepare_file_path_list(file_id_list, file_dir, file_extension):
    file_path_list = []
    for file_id in file_id_list:
        file_path = file_dir + "/" + file_id + file_extension
        file_path_list.append(file_path)

    return file_path_list

def phase_loss(y_true, y_pred):
    return

FRAME_BUFFER_SIZE = 3000000

train_file_number = 6000
valid_file_number = 700
test_file_number = 49

inp_dim = 513
out_dim = 513

inp_feat_dir = "data/abs"
out_feat_dir = "data/phase"

inp_file_ext = ".ab"
out_file_ext = ".ph"

ab_files = glob.glob("data/abs/*.ab")
ph_files = glob.glob("data/phase/*.ph")

assert len(ab_files) == len(ph_files) == train_file_number + valid_file_number + test_file_number

file_id_list = []
fid = open("file_id_list.scp", mode="r")
for line in fid.readlines():
    line = line.strip()
    if len(line) < 1:
        continue
    file_id_list.append(line)
fid.close()

train_id_list = file_id_list[0: train_file_number]
valid_id_list = file_id_list[train_file_number: train_file_number + valid_file_number]
test_id_list  = file_id_list[train_file_number + valid_file_number: train_file_number + valid_file_number + test_file_number]

inp_train_file_list = prepare_file_path_list(train_id_list, inp_feat_dir, inp_file_ext)
out_train_file_list = prepare_file_path_list(train_id_list, out_feat_dir, out_file_ext)

inp_valid_file_list = prepare_file_path_list(valid_id_list, inp_feat_dir, inp_file_ext)
out_valid_file_list = prepare_file_path_list(valid_id_list, out_feat_dir, out_file_ext)

inp_test_file_list = prepare_file_path_list(test_id_list, inp_feat_dir, inp_file_ext)
out_test_file_list = prepare_file_path_list(test_id_list, out_feat_dir, out_file_ext)

train_x, train_y = read_data_from_list(inp_train_file_list, out_train_file_list, inp_dim, out_dim)
valid_x, valid_y = read_data_from_list(inp_valid_file_list, out_valid_file_list, inp_dim, out_dim)
