import pyopencl as cl
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt


def display_mat_keys(file_path, key):
    mat_contents = sio.loadmat(file_path)
    print(mat_contents.keys())
    if key in mat_contents:
        print(f"Value for key '{key}': {mat_contents[key]}")
    else:
        print(f"Key '{key}' not found in the .mat file")

#plot out data in an array found in a .mat file
def plot_mat_data(file_path,T):
    mat_contents = sio.loadmat(file_path)
    #only first 10_000 points

    top = 'r_li'
    bot = 'e_li'

    dataa = mat_contents[top][0][:T]
    datab = mat_contents[top][1][:T]

    datac = mat_contents[bot][0][:T]
    datad = mat_contents[bot][1][:T]

    ts = np.linspace(0, T, len(dataa))

    #data = mat_contents[key]
    plt.figure()
    plt.plot(ts,dataa)
    plt.plot(ts,datab)

    plt.figure()
    plt.plot(ts, datac)
    plt.plot(ts, datad)

    plt.show()


if __name__ == '__main__':
    directory = 'Joch_data_given'
    #file_name = 'TwoChoiceTrajectoriesDensity_050.mat'  # Replace with your actual .mat file name
    #file_name = 'TwoChoiceAmputatedDensity_075_VC_100_HC_100.mat'  # Replace with your actual .mat file name
    file_name = 'TwoChoiceTrajectoriesSmallStepDensity_075.mat'  #  Replace with your actual .mat file name

    file_path = os.path.join(directory, file_name)
    display_mat_keys(file_path,'r_step')
    TMAX = 1_000_00
    print(os.cpu_count())
    data = sio.loadmat(file_path)
    print(data.keys())
    print(np.shape(data['r_li']))
    print(data['Tend'])
    print(data['dt'])


    plot_mat_data(file_path,TMAX)