import glob
import numpy as np

import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD, Adam, Adagrad

from src import utils

def phase_loss(y_true, y_pred):
    return

# train_file_number = 6000
# valid_file_number = 700
# test_file_number = 49

train_file_number = 1850
valid_file_number = 100
test_file_number = 50

inp_dim = 1285
out_dim = 257

# DNN configuration
hidden_layer_type = ['relu', 'relu', 'relu']
hidden_layer_size = [1024, 1024, 1024]

output_type = 'linear'
dropout = 0.0
loss_function = phase_loss
optimizer = Adam()
shuffle_data = True

num_of_epochs = 100

model = Sequential()

for i in range(len(hidden_layer_type)):
    if i == 0:
        input_size = inp_dim
    else:
        input_size = hidden_layer_size[i - 1]

    model.add(Dense(
            units=hidden_layer_size[i],
            activation=hidden_layer_type[i],
            kernel_initializer="normal",
            input_dim=input_size))

    model.add(Dropout(dropout_rate))

final_layer = model.add(Dense(
    units=out_dim,
    activation=output_type.lower(),
    kernel_initializer="normal",
    input_dim=hidden_layer_size[-1]))

model.compile(loss=loss_function, optimizer=optimizer)

inp_feat_dir = "data/norm_abs"
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

inp_train_file_list = utils.prepare_file_path_list(train_id_list, inp_feat_dir, inp_file_ext)
out_train_file_list = utils.prepare_file_path_list(train_id_list, out_feat_dir, out_file_ext)

inp_valid_file_list = utils.prepare_file_path_list(valid_id_list, inp_feat_dir, inp_file_ext)
out_valid_file_list = utils.prepare_file_path_list(valid_id_list, out_feat_dir, out_file_ext)

inp_test_file_list = utils.prepare_file_path_list(test_id_list, inp_feat_dir, inp_file_ext)
out_test_file_list = utils.prepare_file_path_list(test_id_list, out_feat_dir, out_file_ext)

train_x, train_y = utils.read_data_from_list(inp_train_file_list, out_train_file_list, inp_dim, out_dim)
valid_x, valid_y = utils.read_data_from_list(inp_valid_file_list, out_valid_file_list, inp_dim, out_dim)
