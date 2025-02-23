from scipy.optimize import minimize
import pyopencl as cl
import os
import numpy as np
import time
from multiprocessing import Pool
import random as rng

def count_transitions(data):
    count = {}
    for i in range(len(data[0]) - 1):

        indices = (
        data[0][i], data[1][i], data[2][i], data[3][i], data[0][i + 1], data[1][i + 1], data[2][i + 1], data[3][i + 1])
        if indices not in count:
            count[indices] = 0
        count[indices] += 1
    return count

#for inference
def get_next_prob_arr(params9, state, model, context, queue):
    MAXTOP = model.NR
    MAXBOT = model.NE

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
            p_arr = p_arr.reshape((model.NR, model.NR, model.NE, model.NE))
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
    maxL = minimize(cl_likelyhood_batch, initial, args=(count,model,), method='Nelder-Mead')
    return maxL.x

#for forward modelling
def calc_next_state(model,params,current_state, prog_path):
    MAXTOP = model.NR
    MAXBOT = model.NE

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
def faster_function_nopij(Parr):
    randnum = rng.random()

    shape = Parr.shape
    flat_Parr = Parr.reshape(-1)  # Flatten the Parr array
    cumsum = np.cumsum(flat_Parr)  # Compute cumulative sum
    index = np.searchsorted(cumsum, randnum)  # Find index where randnum fits in cumsum

    if index < len(cumsum):
        NAm, NBn, NCo, NDp = np.unravel_index(index, shape)
        return NAm, NBn, NCo, NDp
    else:
        return 0, 0, 0, 0
def simulation_nopij(model,Nstart,Tmax,params,file_path):
    epsilon1, epsilon2 = 0.0, 0.0
    print(params)


    (hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx) = params
    params = (
    halpha, ha, halpha - epsilon1, ha + epsilon1, hgamma, hc, hgamma - epsilon2, hc + epsilon2, kcoop, kcomp, kdu, kud,
    kx)

    MAXTOP = model.NR
    MAXBOT = model.NE
    NA = Nstart[0]
    NB = Nstart[1]
    NC = Nstart[2]
    ND = Nstart[3]
    t = 0
    A = [NA]
    B = [NB]
    C = [NC]
    D = [ND]
    state_cache = {}
    p_arr_cache = {}

    while t < Tmax:
        #print(params)
        state= (A[-1],B[-1],C[-1],D[-1])
        state_key = f"{state}"
        if state_key in state_cache:
            p_state = p_arr_cache[state_key]
        else:
            next_state = calc_next_state(model,params, state, file_path)
            if np.sum(next_state) != 0:
                next_state /= np.sum(next_state)
            next_state = next_state.reshape((MAXTOP, MAXTOP, MAXBOT, MAXBOT))
            p_arr_cache[state_key] = next_state
            state_cache[state_key] = state_key
            p_state = next_state

        NA,NB,NC,ND=faster_function_nopij(p_state)
        t += 1
        if t%100000==0:
            print(t)
        # print(NA, NB, NC, ND, 'a,b,c,d')

        A.append(NA)
        B.append(NB)
        C.append(NC)
        D.append(ND)

    return A,B,C,D







