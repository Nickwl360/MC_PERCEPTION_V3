
import numpy as np
import matplotlib.pyplot as plt
import pyopencl as cl
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
example_file_path = os.path.join(current_dir, 'mc_perception_opencl.cl')
perception_cl_prog = example_file_path

# constants

rng = np.random

def calc_next_state(model,params,current_state, prog_path):
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

def get_Pij(model,params):
    MAXTOP = model.maxtop
    MAXBOT = model.maxbot
    Pij = np.zeros((MAXTOP,MAXTOP,MAXBOT,MAXBOT,MAXTOP,MAXTOP,MAXBOT,MAXBOT),dtype=np.float64)
    for A in range(0,MAXTOP):
        for B in range(0,MAXTOP):
            print(A,B)
            for C in range(0,MAXBOT):
                for D in range(0,MAXBOT):
                    state = (A,B,C,D)
                    P_next = calc_next_state(params, state, perception_cl_prog)
                    P_next/=np.sum(P_next)
                    P_next= P_next.reshape((MAXTOP,MAXTOP,MAXBOT,MAXBOT))
                    Pij[A,B,C,D,:,:,:,:] = P_next

    return Pij
def renormalizepij(Pijkl):
    # Create a matrix to store the normalization factors
    normalizefactors = np.sum(Pijkl, axis=(4, 5, 6, 7))  # Sum over the end state indices
    # Ensure that the denominator is not zero to avoid division by zero
    normalizefactors[normalizefactors == 0] = 1.0
    # Normalize the transition matrix
    Pijkl /= normalizefactors[:, :, :, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    #This works, I double checked with the old method
    return Pijkl


def faster_function(Parr,ns):
    na,nb,nc,nd=ns
    Parr = Parr[na, nb, nc, nd, :, :, :, :]
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

def simulation(Nstart,pmnopnorm,Tmax):
    NA = Nstart[0]
    NB = Nstart[1]
    NC = Nstart[2]
    ND = Nstart[3]
    t = 0
    A = [NA]
    B = [NB]
    C = [NC]
    D = [ND]

    pmnopnormal = pmnopnorm
    while t < Tmax:
        #print(params)
        #NA,NB,NC,ND = nextNs(pmnopnormal,NA,NB,NC,ND)
        NA,NB,NC,ND=faster_function(pmnopnormal,(NA,NB,NC,ND))
        t += 1
        print(t)
        # print(NA, NB, NC, ND, 'a,b,c,d')

        A.append(NA)
        B.append(NB)
        C.append(NC)
        D.append(ND)
        # print("Last A:", A[len(A) - 1])
        # print("Last B:", B[len(B) - 1])
        # print("Last C", C[len(C) - 1])
        # print("Last D:", D[len(D) - 1])
        # print(type(A), type(B), type(C),type(D), 'types A-D')
    return A,B,C,D

def simulation_nopij(model,Nstart,Tmax,params,file_path):
    epsilon1, epsilon2 = 0.0, 0.0
    (hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx) = params
    params = (
    halpha, ha, halpha - epsilon1, ha + epsilon1, hgamma, hc, hgamma - epsilon2, hc + epsilon2, kcoop, kcomp, kdu, kud,
    kx)

    MAXTOP = model.maxtop
    MAXBOT = model.maxbot
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


#
if __name__ == "__main__":
    ########PARAM EDITING#############################
    ### TO REPLACE WITH A SAVING FUNCTION ###
    initial = (0, 0, 0, 0)  # A,B,C,D

    epsilon1 = .0
    epsilon2 = .0
    Itest = 100


    #params = (-10.57359064, -10.74478163, -8.46817699, -3.50334181, 0.81771474, 1.87695072, 2.6787412, 0.12152318, 0.71633869)
    #(hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx) = (-8.39780022, -8.31815575, -6.24283186, -0.62797361, 4.64786633, 2.1348466, 6.06874194, 0.29438665, 1.62159095)
    #(hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx)= (-11.45954105, - 9.81027345, - 10.15358925, - 1.49456199,  0.93641602, 1.79710763, 2.86152824, 0.11585655, 0.56313622)#000   .013
    (hgamma, hc, halpha, ha, kcoop, kcomp, kdu, kud, kx) = (-10.57359064, -10.74478163, -8.46817699, -3.50334181, 0.81771474, 1.87695072, 2.6787412, 0.12152318, 0.71633869)


    params = (halpha, ha, halpha - epsilon1, ha + epsilon1, hgamma, hc, hgamma - epsilon2, hc + epsilon2, kcoop, kcomp, kdu, kud,kx)
    #Pij = get_Pij(params)

    ########################################################

    ##RUNNINGFORWARD################################################
    #As,Bs,Cs,Ds = simulation(initial,Pij,Tmax)
    Tmax = 100000
    As,Bs,Cs,Ds = simulation_nopij_nomodel(initial,Tmax,params,perception_cl_prog)

    total_length = len(As)

    print('halpha, hA,hbeta,hB,hgamma,hC,hdelta,hD,kcoop,kcomp,kdu,kud,kx = ',params)

    ts= np.linspace(0,Tmax,len(As))
    plt.plot(ts,As,linewidth=1,c='b')
    plt.plot(ts,Bs,linewidth=1,c='r')
    plt.xlabel("T",fontsize=15)
    plt.ylabel("# Activated",fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # # #
    plt.figure()
    plt.plot(ts, Cs, linewidth=1, c='b')
    plt.plot(ts, Ds, linewidth=1, c='r')
    plt.xlabel("T" , fontsize=15)
    plt.ylabel("# Activated", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    # # #





