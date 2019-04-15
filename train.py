import glob
import numpy as np
import os

import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.core import Lambda
from keras.optimizers import SGD, Adam, Adagrad
import keras.backend as K
from keras import layers
from keras import activations

from src import utils

def phase_group_loss(y_true, y_pred):
    phase_loss = K.sum(- K.cos(y_true - y_pred), axis=-1)

    dim = 257

    temp_y_true_1 = Lambda(lambda x: x[:, 1 : dim], output_shape=(256, ))(y_true)
    temp_y_true_2 = Lambda(lambda x: x[:, 0 : dim - 1], output_shape=(256, ))(y_true)
    temp_y_pred_1 = Lambda(lambda x: x[:, 1 : dim], output_shape=(256, ))(y_pred)
    temp_y_pred_2 = Lambda(lambda x: x[:, 0 : dim - 1], output_shape=(256, ))(y_pred)
    group_loss = K.sum(- K.cos( - ( temp_y_true_1 - temp_y_true_2 ) + ( temp_y_pred_1 - temp_y_pred_2 ) ), axis=-1)

    return phase_loss + 0.1 * group_loss
#    return phase_loss

def gated_linear_unit(x):
    gate_activation = activations.get("sigmoid")
    temp = x * gate_activation(x)
    output = x + temp
    return output
#    return temp

TRAINMODEL = True
TESTMODEL = True

train_file_number = 6000
valid_file_number = 700
test_file_number = 49

#train_file_number = 1850
#valid_file_number = 100
#test_file_number = 50

inp_dim = 1285
out_dim = 257

# DNN configuration
hidden_layer_type = ['relu', 'relu', 'relu', 'relu']
hidden_layer_size = [2048, 2048, 2048, 2048]

output_type = 'linear'
dropout_rate = 0
loss_function = phase_group_loss
optimizer = Adagrad(lr=0.0001)
shuffle_data = True
batch_size    = 256

num_of_epochs = 100

model = Sequential()

#for i in range(len(hidden_layer_type)):
#    if i == 0:
#        input_size = inp_dim
#    else:
#        input_size = hidden_layer_size[i - 1]
#
#    model.add(Dense(
#            units=hidden_layer_size[i],
#            activation=hidden_layer_type[i],
#            kernel_initializer="normal",
#            input_dim=input_size))
#
#    model.add(Dropout(dropout_rate))
#
#final_layer = model.add(Dense(
#    units=out_dim,
#    activation=output_type.lower(),
#    kernel_initializer="normal",
#    input_dim=hidden_layer_size[-1]))
#
#model.compile(loss=loss_function, optimizer=optimizer)

for i in range(len(hidden_layer_type)):
    if i == 0:
        input_size = inp_dim
    else:
        input_size = hidden_layer_size[i - 1]

    model.add(Dense(hidden_layer_size[i]))

    model.add(Lambda(gated_linear_unit))

    model.add(Dropout(dropout_rate))

final_layer = model.add(Dense(
    units=out_dim,
    activation=output_type.lower(),
    kernel_initializer="normal",
    input_dim=hidden_layer_size[-1]))

model.compile(loss=loss_function, optimizer=optimizer)

inp_feat_dir = "data/norm_abs"
out_feat_dir = "data/phase"
pred_feat_dir = "gen"

inp_file_ext = ".ab"
out_file_ext = ".ph"

ab_files = glob.glob("data/norm_abs/*.ab")
ph_files = glob.glob("data/phase/*.ph")

json_model_file = "weight/feed_forward_3_relu.json"
h5_model_file = "weight/feed_forward_3_relu.h5"

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
gen_test_file_list = utils.prepare_file_path_list(test_id_list, pred_feat_dir, out_file_ext)

if TRAINMODEL:
    train_x, train_y = utils.read_data_from_list(inp_train_file_list, out_train_file_list, inp_dim, out_dim)
    valid_x, valid_y = utils.read_data_from_list(inp_valid_file_list, out_valid_file_list, inp_dim, out_dim)

    model.fit(train_x, train_y, batch_size=batch_size, epochs=num_of_epochs, shuffle=shuffle_data, validation_data=(valid_x, valid_y))

    model_json = model.to_json()
    with open(json_model_file, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(h5_model_file)
    print("Saved model to disk")

if TESTMODEL:
    json_file = open(json_model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(h5_model_file)
    print("Loaded model from disk")

    model = loaded_model

    test_file_number = len(gen_test_file_list)
    print("generating features on held-out test data...")

    for i in range(test_file_number):
        inp_test_file_name = inp_test_file_list[i]
        gen_test_file_name = gen_test_file_list[i]
        temp_test_x        = np.fromfile(inp_test_file_name, dtype=np.float32)
        temp_test_x = np.reshape(temp_test_x, (inp_dim, -1))
        temp_test_x = temp_test_x.T
        num_of_rows        = temp_test_x.shape[0]
        print(temp_test_x.shape)

        predictions = model.predict(temp_test_x)
        predictions = np.array(predictions, 'float32')
#        predictions = predictions.T
        print(predictions.shape)

        fid = open(gen_test_file_name, 'wb')
        predictions.tofile(fid)
        fid.close()
