from math import gamma

import numpy as np
import data_handler
import plot_handler


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
def get_shifted_trajectory_densities(shifted_trajectory, u_space):

    X_i, Y_i, Xb_i, Yb_i = shifted_trajectory
    uX, uY, uXb, uYb = u_space
    uX,uY,uXb,uYb = np.array(uX),np.array(uY),np.array(uXb),np.array(uYb)

    countXYXb = np.zeros((len(uX), len(uY), len(uXb)), dtype=int)
    countXYYb = np.zeros((len(uX), len(uY), len(uYb)), dtype=int)

    print('getting densities')
    for i in range(len(X_i)):

        x_idx = np.where(uX == X_i[i])[0][0]
        y_idx = np.where(uY == Y_i[i])[0][0]
        xb_idx = np.where(uXb == Xb_i[i])[0][0]
        yb_idx = np.where(uYb == Yb_i[i])[0][0]

        countXYXb[x_idx, y_idx, xb_idx] += 1
        countXYYb[x_idx, y_idx, yb_idx] += 1

    print('finished counting densities')
    return countXYXb, countXYYb

def pull_traj_givenSi(model, Si):
    ntraj = model.Ntraj
    Ltime = model.Tper
    Ltraj = int(Ltime/model.dt)
    #
    E_nli = np.zeros((ntraj, 2, Ltraj), dtype=np.int8)
    R_nli = np.zeros((ntraj, 2, Ltraj), dtype=np.int8)

    ai, bi, ci, di = Si
    longa, longb, longc, longd = model.long_trajectory

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


class AnalysisHandler:
    def __init__(self, raw_trajectory, model,joch_data=False):
        self.long_trajectory = raw_trajectory[0]
        #in form (As,Bs,Cs,Ds) ALREADY
        self.prefix = model.prefix
        self.joch_data = joch_data

        self.I = model.I
        self.dt = model.dt
        self.maxtop = model.NR
        self.maxbot = model.NE

        self.get_shifted_trajectory_and_space()
        self.dh = data_handler.DataHandler()
        self.plt = plot_handler.PlotHandler(self)

        self.dominance_statistics = None
        self.countXYXb = None
        self.countXYYb = None
        self.average_trajectories = None
        self.gamma_dist, self.lower_dist, self.ac_dist, self.ad_dist = None, None, None, None

    def get_shifted_trajectory_and_space(self):

        (As,Bs,Cs,Ds) = self.long_trajectory
        As,Bs,Cs,Ds = np.array(As,dtype=np.int8),np.array(Bs,dtype=np.int8),np.array(Cs,dtype=np.int8),np.array(Ds,dtype=np.int8)
        X_i = Cs - Ds
        Xb_i = Cs + Ds
        Y_i = As - Bs
        Yb_i = As + Bs
        self.shifted_trajectory = [X_i,Y_i,Xb_i,Yb_i]
        del(As,Bs,Cs,Ds)
        uX = np.unique(X_i)
        uY = np.unique(Y_i)
        uXb = np.unique(Xb_i)
        uYb = np.unique(Yb_i)

        self.u_space = [uX,uY,uXb,uYb]

    def save_all(self):
        if self.dominance_statistics is not None:
            self.dh.save_json('dominance_statistics', f'{self.prefix}dominance_statistics', self.dominance_statistics)
        if self.countXYXb is not None:
            self.dh.save_npz('densities', f'{self.prefix}countsXYXb', counts=self.countXYXb)
        if self.countXYYb is not None:
            self.dh.save_npz('densities', f'{self.prefix}countsXYYb', counts=self.countXYYb)
        if self.average_trajectories is not None:
            self.dh.save_npz('average_trajectories', f'{self.prefix}Tper{self.Tper}Ntraj{self.Ntraj}average_trajectories', **self.average_trajectories)
        if self.gamma_dist is not None:
            self.dh.save_npz('densities', f'{self.prefix}distributions', gamma_dist=self.gamma_dist, lower_dist=self.lower_dist, ac_dist=self.ac_dist, ad_dist=self.ad_dist)


    # # # Trajectory Analysis # # #
    #############################################################################
    def load_or_generate_dominance_statistics(self):

        (As,Bs,Cs,Ds) = self.long_trajectory
        As,Bs,Cs,Ds = np.array(As,dtype=np.int8),np.array(Bs,dtype=np.int8),np.array(Cs,dtype=np.int8),np.array(Ds,dtype=np.int8)
        # check if data is in directory 'trajectory statistics' or not
        try:
            self.dominance_statistics = self.dh.load_json('dominance_statistics', f'{self.prefix}dominance_statistics')
        except FileNotFoundError:
            print('No dominance statistics found, calculating now')
            # convert As,Bs to 2 dd array of dtype int8
            As, Bs = np.array(As, dtype=np.int8), np.array(Bs, dtype=np.int8)

            R_ki = np.array([As, Bs])
            Duration, Variance, CV, Skewness, Sequential, RevCount, Pdouble, Pneither = extract_dominance_statistics(
                self.dt, R_ki, NR=[[self.maxtop - 1]])
            print(
                f"Duration: {Duration}, Variance: {Variance}, CV: {CV}, Skewness: {Skewness}, Sequential: {Sequential}, RevCount: {RevCount}, Pdouble: {Pdouble}, Pneither: {Pneither}")

            self.dominance_statistics = {
                'Duration': Duration,
                'Variance': Variance,
                'CV': CV,
                'Skewness': Skewness,
                'Sequential': Sequential,
                'RevCount': RevCount,
                'Pdouble': Pdouble,
                'Pneither': Pneither
            }
            self.save_all()
    def load_or_generate_trajectory_density(self) -> None:
        try:
            self.countXYXb, self.countXYYb = self.dh.load_npz('densities', f'{self.prefix}countsXYXb'), self.dh.load_npz('densities', f'{self.prefix}countsXYYb')
        except FileNotFoundError:
            print('No trajectory density found, calculating now')
            self.countXYXb, self.countXYYb = get_shifted_trajectory_densities(self.shifted_trajectory, self.u_space)
            self.save_all()
    def load_or_generate_avg_trajectory(self, Tper, Ntraj):
        self.Tper = Tper
        self.Ntraj = Ntraj
        try:
            self.average_trajectories = self.dh.load_npz('average_trajectories', f'{self.prefix}Tper{Tper}Ntraj{Ntraj}average_trajectories')
        except FileNotFoundError:
            print('No average trajectory found, calculating now')
            self.load_or_generate_trajectory_density()
            self.average_trajectories = self.compute_average_trajectories()
            self.save_all()
    def load_or_generate_distributions(self):
        try:
            distributions = self.dh.load_npz('densities', f'{self.prefix}distributions')
            self.gamma_dist, self.lower_dist, self.ac_dist, self.ad_dist = distributions['gamma_dist'], distributions['lower_dist'], distributions['ac_dist'], distributions['ad_dist']
        except FileNotFoundError:
            print('No distributions found, calculating now')
            self.gamma_dist, self.lower_dist, self.ac_dist, self.ad_dist = self.compute_distributions()
            self.save_all()


    def compute_average_trajectories(self):

        uX, uY, uXb, uYb = self.u_space
        # countXYXb = list(model.countXYXb.values())[0]
        countXYXb = self.countXYXb
        NE = self.maxbot - 1
        NR = self.maxtop - 1

        print(np.shape(countXYXb))
        # Restrict to dominant states
        countXYXb = countXYXb[:, [0, -1], :]  # Keep only the first and last indices for uY
        N = 2
        total_valid_count = np.sum(countXYXb > 0)
        print(f"Total valid count: {total_valid_count}")

        # Short time axis
        ti = np.arange(0, self.Tper, self.dt)
        Ni = len(ti)
        print('NUMBER OF STEPS PER TRAJ:', Ni)

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

        print('Allocated arrays')
        for n in range(0, 2):
            # Initialize inhibitory states
            if n == 0:
                R0 = 0
                R0p = NR
            else:
                R0 = NR
                R0p = 0

            for m, X in enumerate(uX):

                for p, Xb in enumerate(uXb):
                    X, Xb = int(X), int(Xb)
                    # Initialize excitatory states
                    E0 = (X + Xb) / 2
                    E0p = (-1 * X + Xb) / 2

                    # Check if the location was visited
                    if (abs(R0 - R0p) >= NR) and countXYXb[m, n, p] > 0:
                        # Generate long_trajectory using the model
                        initial_state = (R0, R0p, E0, E0p)
                        print('calculating trajectory for m,n,p:', X, (R0 - R0p), Xb)
                        # E_kli, R_kli = model.generate_multiple_trajectories(initial_state, len(ti), Nrepeat)
                        E_kli, R_kli = pull_traj_givenSi(self, initial_state)
                        # option to extract instead

                        # Discretize to disambiguate
                        # E_kli = np.round(E_kli.astype(np.uint8)/(model.maxbot-1))
                        # R_kli = np.round(R_kli.astype(np.uint8)/(model.maxbot-1))

                        # Differential coordinates
                        X_ki = (E_kli[:, 0, :] - E_kli[:, 1, :])
                        Xb_ki = (E_kli[:, 0, :] + E_kli[:, 1, :])
                        Y_ki = (R_kli[:, 0, :] - R_kli[:, 1, :])
                        Yb_ki = (R_kli[:, 0, :] + R_kli[:, 1, :])

                        X_ki = np.round(X_ki) / NE
                        Xb_ki = np.round(Xb_ki) / NE
                        Y_ki = np.round(Y_ki) / NR
                        Yb_ki = np.round(Yb_ki) / NR

                        # Average over repetitions
                        X_mnpi[m, n, p, :] = np.squeeze(np.mean(X_ki, axis=0))
                        Y_mnpi[m, n, p, :] = np.squeeze(np.mean(Y_ki, axis=0))
                        Xb_mnpi[m, n, p, :] = np.squeeze(np.mean(Xb_ki, axis=0))
                        Yb_mnpi[m, n, p, :] = np.squeeze(np.mean(Yb_ki, axis=0))

                        # get fluctuations in the trajectories
                        X_std_i[m, n, p, :] = np.squeeze(np.std(X_ki, axis=0))
                        Y_std_i[m, n, p, :] = np.squeeze(np.std(Y_ki, axis=0))
                        Xb_std_i[m, n, p, :] = np.squeeze(np.std(Xb_ki, axis=0))

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

        return results
    def compute_distributions(self):
        dataA, dataB, dataC, dataD = self.long_trajectory
        # Calculate gamma distribution
        fliptimes = []
        time = 0
        if dataA[2] > dataB[2]:
            A, B = 1, 0
        else:
            A, B = 0, 1
        for i in range(3, len(dataA)):
            if A == 1:
                if dataA[i] >= dataB[i]:
                    time += 1
                else:
                    B, A = 1, 0
                    fliptimes.append(time)
                    time = 0
            if B == 1:
                if dataB[i] >= dataA[i]:
                    time += 1
                else:
                    A, B = 1, 0
                    fliptimes.append(time)
                    time = 0
        gamma_dist = np.array(fliptimes)

        # Calculate lower distribution
        lower_dist = np.zeros(self.maxbot)
        for i in range(len(dataC)):
            c = int(dataC[i])
            lower_dist[c] += 1
        lower_dist /= np.sum(lower_dist)

        # Calculate combined distributions
        ac_dist = np.zeros((self.maxtop, self.maxbot))
        ad_dist = np.zeros((self.maxtop, self.maxbot))
        for i in range(len(dataC)):
            a = int(dataA[i])
            c = int(dataC[i])
            d = int(dataD[i])
            ac_dist[a, c] += 1
            ad_dist[a, d] += 1

        ac_dist /= np.sum(ac_dist)
        ad_dist /= np.sum(ad_dist)

        return gamma_dist, lower_dist, ac_dist, ad_dist