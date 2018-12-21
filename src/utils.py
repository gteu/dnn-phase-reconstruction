import numpy as np
from sklearn import preprocessing


FRAME_BUFFER_SIZE = 3000000

def read_data_from_list(inp_file_list, out_file_list, inp_dim, out_dim):
    temp_x = np.empty((FRAME_BUFFER_SIZE, inp_dim), dtype=np.float32)
    temp_y = np.empty((FRAME_BUFFER_SIZE, out_dim), dtype=np.float32)

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

def read_data_from_one_list(file_list, dim):
    temp_x = np.empty((FRAME_BUFFER_SIZE, dim), dtype=np.float32)

    current_index = 0

    num_of_sounds = len(file_list)

    for i in range(num_of_sounds):
        file_x = np.fromfile(file_list[i], dtype=np.float32)
        file_x = np.reshape(file_x, (dim, -1))
        file_x = file_x.T

        frame_number = file_x.shape[0]

        temp_x[current_index:current_index+frame_number] = file_x[0:frame_number]

        current_index += frame_number

    temp_x = temp_x[0:current_index]

    return temp_x

def prepare_file_path_list(file_id_list, file_dir, file_extension):
    file_path_list = []
    for file_id in file_id_list:
        file_path = file_dir + "/" + file_id + file_extension
        file_path_list.append(file_path)

    return file_path_list

def compute_norm_stats(data, stats_file):
    #### normalize training data ####

    scaler = preprocessing.StandardScaler().fit(data)
    norm_matrix = np.vstack((scaler.mean_, scaler.scale_))

    norm_matrix = norm_matrix.astype(np.float32)

    norm_matrix.tofile(stats_file)

    return scaler
