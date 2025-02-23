import numpy as np
import pandas as pd
import os
import scipy.io as sio




def pull_traj_givenSi(model, Si):
    ntraj = model.Nrepeat
    Ltime = model.L_per_traj
    Ltraj = int(Ltime/model.dt)
    #
    E_nli = np.zeros((ntraj, 2, Ltraj), dtype=np.int8)
    R_nli = np.zeros((ntraj, 2, Ltraj), dtype=np.int8)

    ai, bi, ci, di = Si
    longa, longb, longc, longd = model.long_trajectory[0]

    # Convert to numpy arrays for faster computation
    longa, longb, longc, longd = map(lambda x: np.array(x, dtype=np.int8), [longa, longb, longc, longd])
    # Find indices where all conditions are met
    indices = np.nonzero((longa == ai) & (longb == bi) & (longc == ci) & (longd == di))

    # Loop over the indices and extract the trajectories
    for idx, i in enumerate(indices[0][:ntraj]):
        sample = longa[i:i+Ltraj], longb[i:i+Ltraj], longc[i:i+Ltraj], longd[i:i+Ltraj]
        R_nli[idx,0,:] = sample[0]
        R_nli[idx,1,:] = sample[1]
        E_nli[idx,0,:] = sample[2]
        E_nli[idx,1,:] = sample[3]
    del(longa, longb, longc, longd)
    return E_nli, R_nli

def display_dominance_stats(data):
    # Display dominance statistics for each model
    # First display a top row for the column names
    Duration, Variance, CV, Skewness, Sequential, RevCount, Pdouble, Pneither=data
    header = ['I',  'Duration',    'Variance',    'CV',   'Skewness',   'CC1',   'CC2',   'CC3',   'RevCount',    'Pneither (%)']
    print(f"{'   |   '.join(header)}")
    print('-' * 120)

    row1 = [
        '075',
        Duration[0],
        Variance[0],
        CV[0],
        Skewness[0],
        Sequential[0],
        Sequential[1],
        Sequential[2],
        RevCount,
        #model.dominance_statistics["Pdouble"],
        Pneither
    ]
    row2 = [
        '075',
        Duration[1],
        Variance[1],
        CV[1],
        Skewness[1],
        Sequential[0],
        Sequential[1],
        Sequential[2],
        RevCount,
        # model.dominance_statistics["Pdouble"],
        Pneither
    ]
    print(
        f"{'  |  '.join(map(lambda x: f'{x.values[0]:.4f}' if isinstance(x, pd.Series) and isinstance(x.values[0], float) else str(x.values[0]) if isinstance(x, pd.Series) else str(x), row1))}")
    print(
        f"{'  |  '.join(map(lambda x: f'{x.values[0]:.4f}' if isinstance(x, pd.Series) and isinstance(x.values[0], float) else str(x.values[0]) if isinstance(x, pd.Series) else str(x), row2))}")


def extract_dominance_statistics(dt, R_ki, NR):
    Rx_ki = (R_ki == NR)  # Logical indices for times i with R == NR, includes duplicates
    Noriginal = Rx_ki.shape[1]

    # Logical indices for unique, double, and neither dominance
    Ux_ki = Rx_ki[:, np.sum(Rx_ki, axis=0) == 1]
    Dx_ki = Rx_ki[:, np.sum(Rx_ki, axis=0) == 2]
    Nx_ki = Rx_ki[:, np.sum(Rx_ki, axis=0) == 0]

    Nunique = Ux_ki.shape[1]
    Ndouble = Dx_ki.shape[1]
    Nneither = Nx_ki.shape[1]

    # Percentages
    Punique = Nunique * 100 / Noriginal
    Pdouble = Ndouble * 100 / Noriginal
    Pneither = Nneither * 100 / Noriginal

    print(f"Dominance (percent) unique: {Punique:.2f} double: {Pdouble:.2f} neither: {Pneither:.2f}")

    D_i = Ux_ki[0, :] + 2 * Ux_ki[1, :]  # Logical index to unique R == NR
    Ni = len(D_i)
    ti = np.arange(0, Ni * dt, dt)
    dix = np.where(D_i[:-1] != D_i[1:])[0]  # Locate reversals
    print(f"Rx_ki shape: {Rx_ki.shape}")
    print(f"Ux_ki shape: {Ux_ki.shape}")
    print(f"D_i length: {len(D_i)}")
    print(f"ti length: {len(ti)}")
    print(f"dix indices: {dix}")
    if dix.size > 0:
        flag = D_i[0]  # Initial dominance

        # Initialize lists of onsets and offsets
        tD1on, tD1off = ([ti[0]], []) if flag == 1 else ([], [])
        tD2on, tD2off = ([ti[0]], []) if flag == 2 else ([], [])
        tDXon, tDXoff = [ti[0]], []

        List = [flag]  # Initialize list of dominance

        # Loop over all times i
        for i in range(len(D_i) - 1):
            precD, succD = D_i[i], D_i[i + 1]

            if precD != succD:  # If dominance changes from i to i+1
                if precD == 1:
                    tD1off.append(ti[i])
                elif precD == 2:
                    tD2off.append(ti[i])
                tDXoff.append(ti[i])

                if succD == 1:
                    tD1on.append(ti[i])
                    flag = 1
                elif succD == 2:
                    tD2on.append(ti[i])
                    flag = 2
                tDXon.append(ti[i])

                List.append(flag)

        # Finalize lists of onsets and offsets
        if flag == 1:
            tD1off.append(ti[-1])
        elif flag == 2:
            tD2off.append(ti[-1])
        tDXoff.append(ti[-1])

        # Dominance durations
        D1 = np.array(tD1off) - np.array(tD1on)
        D2 = np.array(tD2off) - np.array(tD2on)
        DX = np.array(tDXoff) - np.array(tDXon)

        N1, N2 = len(D1), len(D2)

        # First moments
        Mu1_1, Mu1_2 = np.sum(D1) / N1, np.sum(D2) / N2

        # Second moments
        Mu2_1, Mu2_2 = np.sum((D1 - Mu1_1) ** 2) / N1, np.sum((D2 - Mu1_2) ** 2) / N2

        # Third moments
        Mu3_1, Mu3_2 = np.sum((D1 - Mu1_1) ** 3) / N1, np.sum((D2 - Mu1_2) ** 3) / N2

        # Standard deviations
        Std_1, Std_2 = np.sqrt(Mu2_1), np.sqrt(Mu2_2)

        # Assign output
        RevCount = N1 + N2
        Duration = np.array([Mu1_1, Mu1_2])
        Variance = np.array([Mu2_1, Mu2_2])
        CV = np.array([Std_1, Std_2]) / Duration
        Skewness = np.array([Mu3_1, Mu3_2]) * Duration / Variance ** 2

        # For sequential correlations, z-score periods appropriately
        List = np.array(List)
        DcorrX = DX.copy()
        DcorrX[List == 1] = (DX[List == 1] - Mu1_1) / Std_1
        DcorrX[List == 2] = (DX[List == 2] - Mu1_2) / Std_2

        R11 = np.corrcoef(DcorrX[:-1], DcorrX[1:])[0, 1]
        R12 = np.corrcoef(DcorrX[:-2], DcorrX[2:])[0, 1]
        R13 = np.corrcoef(DcorrX[:-3], DcorrX[3:])[0, 1]

        Sequential = np.array([R11, R12, R13])

    else:
        RevCount = 0
        Duration = np.array([np.inf, np.inf])
        Variance = np.array([0, 0])
        CV = np.array([1, 1])
        Skewness = np.array([1, 1])
        Sequential = np.array([0, 0, 0])

    return Duration, Variance, CV, Skewness, Sequential, RevCount, Pdouble, Pneither

def get_shifted_trajectory_and_space(trajectory_long):

    As,Bs,Cs,Ds = trajectory_long
    As,Bs,Cs,Ds = np.array(As,dtype=np.int8),np.array(Bs,dtype=np.int8),np.array(Cs,dtype=np.int8),np.array(Ds,dtype=np.int8)

    X_i = Cs - Ds
    Xb_i = Cs + Ds
    Y_i = As - Bs
    Yb_i = As + Bs

    shifted_trajectory = [X_i,Y_i,Xb_i,Yb_i]
    del(As,Bs,Cs,Ds)

    uX = np.unique(X_i)
    uY = np.unique(Y_i)
    uXb = np.unique(Xb_i)
    uYb = np.unique(Yb_i)

    u_space = [uX,uY,uXb,uYb]

    return shifted_trajectory, u_space



def compute_average_trajectories(model,   Tper=0.02, Nrepeat=100):
    model.Nrepeat = Nrepeat
    model.L_per_traj = Tper
    shifted_trajectories, u_space = get_shifted_trajectory_and_space(model.long_trajectory[0])
    uX,uY,uXb,uYb = u_space
    model.u_space = u_space

    if model.countXYXb is None:
        model.get_trajectory_density()

    #countXYXb = list(model.countXYXb.values())[0]
    countXYXb = model.countXYXb
    NE = model.maxbot-1
    NR = model.maxtop-1

    print(np.shape(countXYXb))
    # Restrict to dominant states
    countXYXb = countXYXb[:, [0, -1], :]  # Keep only the first and last indices for uY
    uY = uY[[0, -1]]
    N = 2
    total_valid_count = np.sum(countXYXb > 0)
    print(f"Total valid count: {total_valid_count}")

    # Short time axis
    ti = np.arange(0, Tper, model.dt)
    Ni = len(ti)
    print('NUMBER OF STEPS PER TRAJ:',Ni)

    # Allocate arrays for storing long_trajectory
    M = len(uX)
    P = len(uXb)

    X_mnpi = np.full((M, N, P, Ni), np.nan)
    Y_mnpi = np.full((M, N, P, Ni), np.nan)
    Xb_mnpi = np.full((M, N, P, Ni), np.nan)
    Yb_mnpi = np.full((M, N, P, Ni), np.nan)

    X_std_i = np.full((M, N, P, Ni), np.nan)
    Y_std_i = np.full((M, N, P, Ni), np.nan)
    Xb_std_i = np.full((M, N, P, Ni), np.nan)

    print( 'Allocated arrays')
    for n in range(0,2):
        # Initialize inhibitory states
        if n==0:
            R0=0
            R0p=NR
        else:
            R0=NR
            R0p=0

        for m, X in enumerate(uX):

            for p, Xb in enumerate(uXb):
                X, Xb = int(X), int(Xb)
                # Initialize excitatory states
                E0 = (X + Xb) / 2
                E0p = (-1*X + Xb) / 2


                # Check if the location was visited
                if (abs(R0-R0p) >= NR) and countXYXb[m, n, p] > 0:
                    # Generate long_trajectory using the model
                    initial_state = (R0,R0p,E0,E0p)
                    print('calculating trajectory for m,n,p:',X,(R0-R0p),Xb)
                    #E_kli, R_kli = model.generate_multiple_trajectories(initial_state, len(ti), Nrepeat)
                    E_kli, R_kli = pull_traj_givenSi(model, initial_state)
                    #option to extract instead

                                        # Discretize to disambiguate
                    #E_kli = np.round(E_kli.astype(np.uint8)/(model.maxbot-1))
                    #R_kli = np.round(R_kli.astype(np.uint8)/(model.maxbot-1))


                    # Differential coordinates
                    X_ki = (E_kli[:, 0, :] - E_kli[:, 1, :])
                    Xb_ki = (E_kli[:, 0, :] + E_kli[:, 1, :])
                    Y_ki = (R_kli[:, 0, :] - R_kli[:, 1, :])
                    Yb_ki = (R_kli[:, 0, :] + R_kli[:, 1, :])

                    X_ki =np.round(X_ki)/NE
                    Xb_ki = np.round(Xb_ki)/NE
                    Y_ki = np.round(Y_ki)/NR
                    Yb_ki = np.round(Yb_ki)/NR

                    # Average over repetitions
                    X_mnpi[m, n, p, :] = np.squeeze(np.mean(X_ki, axis=0))
                    Y_mnpi[m, n, p, :] = np.squeeze(np.mean(Y_ki, axis=0))
                    Xb_mnpi[m, n, p, :] = np.squeeze(np.mean(Xb_ki, axis=0))
                    Yb_mnpi[m, n, p, :] =np.squeeze(np.mean(Yb_ki, axis=0))

                    #get fluctuations in the trajectories
                    X_std_i[m,n,p,:] = np.squeeze(np.std(X_ki, axis=0))
                    Y_std_i[m,n,p,:] = np.squeeze(np.std(Y_ki, axis=0))
                    Xb_std_i[m,n,p,:] = np.squeeze(np.std(Xb_ki, axis=0))


            print(f"Completed m={m}, n={n}")

    # Package results into a dictionary
    results = {
        "X_mnpi": X_mnpi,
        "Y_mnpi": Y_mnpi,
        "Xb_mnpi": Xb_mnpi,
        "Yb_mnpi": Yb_mnpi,
        "X_std": X_std_i,
        "Y_std": Y_std_i,
        "Xb_std": Xb_std_i,
    }
    model.average_trajectories = results

    return results


def save_analysis(self):

    output_dir = self.analysis_data
    print('saving all things\n')

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save dominance statistics
    if self.dominance_statistics is not None:
        dominance_stats = [
            self.dominance_statistics["RevCount"],
            *self.dominance_statistics["Duration"],
            *self.dominance_statistics["Variance"],
            *self.dominance_statistics["CV"],
            *self.dominance_statistics["Skewness"],
            *self.dominance_statistics["Sequential"],
            self.dominance_statistics["Pneither"],
            self.dominance_statistics["Pdouble"]

        ]
        df = pd.DataFrame([dominance_stats], columns=[
            "RevCount", "Duration1", "Duration2", "Variance1", "Variance2",
            "CV1", "CV2", "Skewness1", "Skewness2", "Sequential1", "Sequential2", "Sequential3","Pneither","Pdouble"
        ])
        df.to_csv(
            os.path.join(output_dir, f"{self.I_test}_HC{self.HC}_VC{self.VC}_dominance_statistics{int(self.Tend)}.csv"),
            index=False
        )
        print("Dominance statistics saved.")

    # Save density counts
    if self.countXYXb is not None:
        np.savez(
            os.path.join(output_dir, f"{self.I_test}_HC{self.HC}_VC{self.VC}_XYXbdensity_counts_{int(self.Tend)}L.npz"),
            countXYXb=self.countXYXb
        )
        np.savez(
            os.path.join(output_dir, f"{self.I_test}_HC{self.HC}_VC{self.VC}_XYYbdensity_counts_{int(self.Tend)}L.npz"),

            countXYYb= self.countXYYb
        )
        print("Density counts saved.")

    # Save average long_trajectory
    if self.average_trajectories is not None:
        np.savez(
            os.path.join(output_dir, f"{self.I_test}_HC{self.HC}_VC{self.VC}_average_trajectories{self.Nrepeat}N_{self.L_per_traj}L.npz"),

            #average_trajectory_data =self.average_trajectories,

            X_mnpi= self.average_trajectories["X_mnpi"],
            Y_mnpi= self.average_trajectories["Y_mnpi"],
            Xb_mnpi= self.average_trajectories["Xb_mnpi"],
            Yb_mnpi= self.average_trajectories["Yb_mnpi"],
            X_std = self.average_trajectories["X_std"],
            Y_std = self.average_trajectories["Y_std"],
            Xb_std = self.average_trajectories["Xb_std"]

        )
        print("Average trajectory data saved.")



def load_analysis(self, file_type):
    dominance_stats_file_path = os.path.join(self.analysis_data, f"{self.I_test}_HC{self.HC}_VC{self.VC}_dominance_statistics{int(self.Tend)}.csv")
    XYXb_file_path = os.path.join(self.analysis_data, f"{self.I_test}_HC{self.HC}_VC{self.VC}_XYXbdensity_counts_{int(self.Tend)}L.npz")
    XYYb_file_path = os.path.join(self.analysis_data, f"{self.I_test}_HC{self.HC}_VC{self.VC}_XYYbdensity_counts_{int(self.Tend)}L.npz")

    if file_type == "dominance_statistics":
        data = pd.read_csv(dominance_stats_file_path)
        self.dominance_statistics = data
        print('loaded dominance stats')
        return data

    elif file_type == "density_countsXYXb":
        data = np.load(str(XYXb_file_path), allow_pickle=True)
        self.countXYXb = data["countXYXb"]
        return self.countXYXb

    elif file_type == "density_countsXYYb":
        data = np.load(str(XYYb_file_path),allow_pickle=True)
        self.countXYYb = data["countXYYb"]
        return self.countXYYb

    elif file_type == "average_trajectories":
        average_trajectories_file_path = os.path.join(self.analysis_data,
                                                      f"{self.I_test}_HC{self.HC}_VC{self.VC}_average_trajectories{self.Nrepeat}N_{self.L_per_traj}L.npz")

        data = np.load(average_trajectories_file_path,allow_pickle=True)

        self.average_trajectories =  data
        return data

    else:
        raise ValueError(f"Unsupported file_type: {file_type}")

if __name__ == "__main__":
    directory = 'Joch_data_given'

    #get dominance_stats for joch data.
    file_name = 'TwoChoiceTrajectoriesSmallStepDensity_075.mat'  #  Replace with your actual .mat file name
    file_path = os.path.join(directory, file_name)

    data = sio.loadmat(file_path)
    print(data.keys())
    #get dominance stats
    R_ki = data['r_li']
    print(repr(R_ki))
    NR = data['NR']
    print(NR)
    dt = data['dt']
    dominance_stats = extract_dominance_statistics(dt, R_ki, NR)
    display_dominance_stats(dominance_stats)