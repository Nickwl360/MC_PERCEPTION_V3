from scipy.optimize import minimize

import pyopencl as cl
import os
import csv
import numpy as np
import scipy.io as sio
import time
from multiprocessing import Pool



def count_transitions(data):
    count = {}
    for i in range(len(data[0]) - 1):

        indices = (
        data[0][i], data[1][i], data[2][i], data[3][i], data[0][i + 1], data[1][i + 1], data[2][i + 1], data[3][i + 1])
        if indices not in count:
            count[indices] = 0
        count[indices] += 1
    return count
def calc_next_state1(model,params,current_state, prog_path):
    MAXTOP = model.maxtop
    MAXBOT = model.maxbot

    #halpha, ha, hbeta,hb,hgamma,hc,hdelta,hd, kcoop,kcomp,kdu,kud, kx = params
    # device = cl.get_platforms()[0].get_devices()[0]
    # ctx = cl.Context([device])
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    Result = np.zeros((MAXTOP * MAXTOP * MAXBOT * MAXBOT), dtype=np.float64)
    Result_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, Result.nbytes)

    params = np.array(params, dtype=np.float64)
    params_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=params)

    current_state = np.array(current_state, dtype=np.float64)
    current_state_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=current_state)


    with open(prog_path, "r") as f:
        program_source = f.read()
    program = cl.Program(ctx, program_source).build()

    global_size = ((MAXTOP*MAXTOP*MAXBOT*MAXBOT),)
    calc = program.calc_next_state

    try:
        calc(queue, global_size, None, Result_buf, params_buf, current_state_buf)
        queue.finish()
    except cl.LogicError as e:
        print("Error during kernel execution:", e)
        return None

    # Read the results back from the GPU to the host
    try:
        cl.enqueue_copy(queue, Result, Result_buf)
    except cl.LogicError as e:
        print("Error copying buffer to host:", e)
        return None
    return Result

def cl_likelyhood(params9, count, model):

    L = 0
    rmax = model.maxtop
    emax = model.maxbot

    (hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx) = params9
    params13 = (halpha, ha, halpha, ha, hgamma, hc, hgamma, hc, kcoop, kcomp, kdu, kud, kx)

    p_arr_cache = {}

    for idx,val in count.items():

        state = tuple(idx[:4])
        next_state = tuple(idx[4:])
        state_key = f"{state}"

        if state_key in p_arr_cache:
            p_arr = p_arr_cache[state_key]
        else:
            p_arr = calc_next_state1(model, params13, state, model.perception_cl_prog)
            try:
                p_arr /= np.sum(p_arr)
            except ZeroDivisionError:
                continue
            p_arr = p_arr.reshape((rmax, rmax, emax, emax))
            p_arr_cache[state_key] = p_arr

        p_val = p_arr[next_state]
        #print('p_val:', p_val, 'val:', val)
        nan_counter=0
        if np.isnan(p_val):
            nan_counter+=1
        if p_val != 0 and not np.isnan(p_val):
            L += val * np.log(p_val)

    print('Likelyhood: ', -1 * L /np.sum(list(count.values())),'nan_count',nan_counter)
    return -L / np.sum(list(count.values()))


def get_next_prob_arr(params9, state, model, context, queue):
    MAXTOP = model.maxtop
    MAXBOT = model.maxbot

    # halpha, ha, hbeta,hb,hgamma,hc,hdelta,hd, kcoop,kcomp,kdu,kud, kx = params
    # device = cl.get_platforms()[0].get_devices()[0]
    # ctx = cl.Context([device])
    ctx = context
    queue = queue
    (hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx) = params9
    params13 = (halpha, ha, halpha, ha, hgamma, hc, hgamma, hc, kcoop, kcomp, kdu, kud, kx)


    Result = np.zeros((MAXTOP * MAXTOP * MAXBOT * MAXBOT), dtype=np.float64)
    Result_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, Result.nbytes)

    params = np.array(params13, dtype=np.float64)
    params_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=params)

    current_state = np.array(state, dtype=np.float64)
    current_state_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=current_state)

    with open(model.perception_cl_prog, "r") as f:
        program_source = f.read()
    program = cl.Program(ctx, program_source).build()

    global_size = ((MAXTOP * MAXTOP * MAXBOT * MAXBOT),)
    calc = program.calc_next_state

    try:
        calc(queue, global_size, None, Result_buf, params_buf, current_state_buf)
        queue.finish()
    except cl.LogicError as e:
        print("Error during kernel execution:", e)
        return None

    # Read the results back from the GPU to the host
    try:
        cl.enqueue_copy(queue, Result, Result_buf)
    except cl.LogicError as e:
        print("Error copying buffer to host:", e)
        return None
    return Result

def cl_likelyhood_batch_worker(args):
    params9, count, model, batch_keys = args
    batch_L = 0
    p_arr_cache = {}
    context = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(context,device=context.devices[0])

    for idx in batch_keys:
        val = count[idx]
        state = tuple(idx[:4])
        next_state = tuple(idx[4:])
        state_key = f"{state}"

        if state_key in p_arr_cache:
            p_arr = p_arr_cache[state_key]
        else:
            p_arr = get_next_prob_arr(params9, state, model, context, queue)
            p_arr = p_arr.reshape((model.maxtop, model.maxtop, model.maxbot, model.maxbot))
            p_arr/=np.sum(p_arr)
            p_arr_cache[state_key] = p_arr
        if p_arr[next_state] > 0 and not np.isnan(p_arr[next_state]):
            batch_L += val * np.log(p_arr[next_state])
    return batch_L
def cl_likelyhood_batch(params9, count, model):
        total_size = sum(count.values())
        batch_size = total_size//os.cpu_count()
        keys = list(count.keys())
        num_batches= (len(keys) + batch_size - 1) // batch_size
        batches = [keys[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
        now = time.time()
        with Pool(processes=os.cpu_count()) as p:
            results = p.map(cl_likelyhood_batch_worker, [(params9, count, model, batch) for batch in batches])
        print('Batch likelyhood= ', -1 * sum(results) / total_size, 'time:', time.time() - now)
        return -1 * sum(results) / total_size

def maximize_likelyhood(model, count,initial):

    #maxL = minimize(cl_likelyhood, initial, args=(count,model,), method='Nelder-Mead')
    maxL = minimize(cl_likelyhood_batch, initial, args=(count,model,), method='Nelder-Mead')

    return maxL.x
def load_mat_data(file_path):
    mat_contents = sio.loadmat(file_path)
    a = mat_contents['r_li'][0]
    b = mat_contents['r_li'][1]
    c = mat_contents['e_li'][0]
    d = mat_contents['e_li'][1]
    return a, b, c, d

def save_inferred_model_csv(I_test, label, params):


    params_names = ['hgamma', 'hc', 'halpha', 'ha', 'kcoop', 'kcomp', 'kdu', 'kud', 'kx']
    dir_path = 'Infered_parameters'
    os.makedirs(dir_path, exist_ok=True)

    #file_path = os.path.join(dir_path, f'{I_test}_inferred_params.csv')
    file_path = os.path.join(dir_path, f'{I_test}_{label}_inferred_params.csv')

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['I',params_names])
        writer.writerow([I_test, list(params)])



if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    perception_cl_prog  = os.path.join(current_dir, 'mc_perception_opencl.cl')

    directory = 'Joch_data_given'
    I_test = '075'

    counts_file = os.path.join(directory, f'counts_{I_test}.npy')
    if os.path.exists(counts_file):
        count = np.load(counts_file, allow_pickle=True).item()
        print('LOADED COUNTS FROM FILE')
    else:
        joch_a, joch_b, joch_c, joch_d = load_mat_data(f'Joch_data_given/TwoChoiceTrajectoriesDensity_{I_test}.mat')
        count = count_transitions((joch_a, joch_b, joch_c, joch_d))
        np.save(counts_file, count)
        print('FOUND COUNTS AND SAVED')
    #print(count)


    t0 = time.time()
    #initial_guess= (-8.96733557, -7.73231853, -6.01935508 ,-0.99322105,  4.7228139 ,  1.98114397 ,6.05944224 , 0.29747507 , 1.53067954)#old
    #initial_guess=(-11.45954105,- 9.81027345, - 10.15358925,- 1.49456199,  0.93641602, 1.79710763, 2.86152824, 0.11585655, 0.56313622)#000   .013
    #initial_guess= (-11.2481983, - 10.05704405,- 8.5134479, - 3.30299464 ,  0.79097099,  1.89522803, 2.70952583 , 0.11745314 , 0.6743023)#025#.01295
    #initial_guess= (-11.03212318, - 10.28098307, - 8.55471608 ,- 3.29921075,  0.80347528, 1.85855315,  2.67696223,  0.11808324,  0.70043714)# 050 .0132
    initial_guess = ( -10.57359064 ,- 10.74478163, - 8.46817699, - 3.50334181,   0.81771474,  1.87695072,   2.6787412, 0.12152318,    0.71633869)#100 .01913

    #Save initial guess in a csv file with the names of parameters, and I_test, and data length.


    #save_inferred_model_csv(I_test, initial_guess)
    max_params = maximize_likelyhood(count, initial_guess,perception_cl_prog)
    print(max_params)
    save_inferred_model_csv(I_test, max_params)

    #print(max_params,'time:',time.time()-t0)


