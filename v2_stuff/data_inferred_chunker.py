import numpy as np
import os

#this file chunks large long_trajectory into smaller ones


def chunk_and_save_data(model,data,letter):
    num_chunks = 50
    small_data_file = model.inferred_trajectories_path
    chunk_size = int(len(data))//int(num_chunks)

    for i in range(num_chunks):
        start = i*chunk_size
        end = (i+1)*chunk_size
        chunk = data[start:end]
        np.save(small_data_file+f'{letter}CHUNK_{i}.npy',chunk)
    return

def load_chunked_data(model,letter, short = False):
    data = []
    small_data_file = model.inferred_trajectories_path
    if short:
        for i in range(3):
            chunk = np.load((small_data_file+ f'{letter}CHUNK_{i}.npy'),allow_pickle=True)
            data.extend(chunk)
        return data
    for i in range(50):
        chunk = np.load((small_data_file+ f'{letter}CHUNK_{i}.npy'),allow_pickle=True)
        data.extend(chunk)
    return data

if __name__ == '__main__':

    if not os.path.exists('../inferred_trajectories'):
        os.makedirs('../inferred_trajectories')

    # for data_file in os.listdir(large_data_dir):
    #     #remove the .npy extension
    #     #data_file = data_file[:-4]
    #
    #     chunk_and_save_data(data_file)
    #     print(f'Chunked {data_file}')

    #test loading
    data = load_chunked_data()
    print(len(data))