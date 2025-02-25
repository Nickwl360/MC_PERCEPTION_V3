
import os
import numpy as np
import pandas as pd
import ast

import data_handler
import maxcal_functions as mc
""" HANDLES THE INFERENCE AND FORWARD MODELING OF THE MAXCAL MODEL """


class BrainModel:

    def __init__(self, I, NR,NE, dt, HC='100', VC='100',joch_data=False):
        self.I = I
        self.NR = NR
        self.NE = NE
        self.HC = HC
        self.VC = VC
        self.dt = dt

        self.ndt = 1#int(self.dt/0.0001)
        print( 'THIS IS NDT',self.ndt)

        self.params = None
        self.Tend = None
        self.countXYXb = None
        self.countXYYb = None
        self.dominance_statistics = None
        self.average_trajectories = None
        self.dh = data_handler.DataHandler()
        self.joch_data = joch_data


        self.long_trajectory = []

        self.prefix = f'I_{self.I}_VC_{self.VC}_HC_{self.HC}_dt_{self.dt}_'
        if self.joch_data:
            self.prefix = f'JOCHDATAI_{self.I}_VC_{self.VC}_HC_{self.HC}_dt_{self.dt}_'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.perception_cl_prog = os.path.join(current_dir, 'mc_perception_opencl.cl')

        # self.inferred_trajectories_path = os.path.join(self.inferred_trajectories, f'I{self.I_test}_L{self.Tend}')

    #inference things
    def load_params(self):
        """
        Load the parameters from a file.
        """
        name = f'{self.prefix}params'
        df = self.dh.load_csv('parameters', name)
        print(df)
        # cell = input("Enter the cell in 'row,col' format: ")
        # row, col = map(int, cell.split(','))
        row,col = 0,1
        self.params = ast.literal_eval(df.iloc[row, col])

    def find_model(self, initial_guess=None):

        self.initial_guess =   (-10.813039700214725, -10.493275942310733, -8.467808763513434, -3.440305398445934, 0.1,
        1., 2., 0.1, 0.1 ) if initial_guess is None else initial_guess

        self.initial_guess=[-10.11114346 , -9.82787052 , -7.80060414 , -2.64612451 ,  0.82227292,
   1.9025744   , 2.72783786  , 0.11921726 ,  0.72995217]
        #check if counts file exists
        counts_path = self.dh._get_path('counts', f'{self.prefix}counts', 'npy')
        if os.path.exists(counts_path):
            count= self.dh.load_npy('counts', f'{self.prefix}counts').item()
            print('LOADED COUNTS FROM FILE')

        else:
            #THIS IS WHERE WE NEED TO ADJUST  LOADING BASED ON WHAT JOCH SENDS ME
            ###############
            ############
            #############
            #self.joch_mat_data = self.dh.load_mat('joch_data', f'TwoChoiceTrajectoriesSmallStepDensity_{self.I}')
            self.joch_mat_data = self.dh.load_mat('joch_data', f'TwoChoiceTrajectoriesDT0002Density_{self.I}')

            # Sample data points based on dt
            a = self.joch_mat_data['r_li'][0][::self.ndt]
            b = self.joch_mat_data['r_li'][1][::self.ndt]
            c = self.joch_mat_data['e_li'][0][::self.ndt]
            d = self.joch_mat_data['e_li'][1][::self.ndt]
            print(len(a))

            count = mc.count_transitions((a, b, c, d))
            self.dh.save_npy('counts', f'{self.prefix}counts', count)

            print('FOUND COUNTS AND SAVED')

        maxparams = mc.maximize_likelyhood(self, count, self.initial_guess)

        print(f'Inferred parameters: {maxparams}')
        self.params = maxparams

        self.dh.save_csv('parameters', f'{self.prefix}params',
                          [['I', 'parameters']] + pd.DataFrame({'I': [self.I], 'params': [','.join(map(str, self.params))]}).values.tolist())
        return

    #forward modelling
    # # # Trajectory Generation # # #
    ############################################################################
    def generate_trajectory(self, initial_state, tmax):
        inferred_trajectory = mc.simulation_nopij(self, initial_state, int(tmax / self.dt), self.params,
                                                  self.perception_cl_prog)

        self.long_trajectory.append(inferred_trajectory)

        return inferred_trajectory
## UNUSED
    # def generate_multiple_trajectories(self, initial_state, tmax, num_trajectories):
    #
    #     E_nli = np.zeros((num_trajectories, 2, tmax))
    #     R_nli = np.zeros((num_trajectories, 2, tmax))
    #
    #     for i in range(num_trajectories):
    #         traj = simulation_nopij(self, initial_state, tmax, self.params, self.perception_cl_prog)
    #         As, Bs, Cs, Ds = traj
    #         E_nli[i, 0, :], E_nli[i, 1, :] = Cs[:tmax], Ds[:tmax]
    #         R_nli[i, 0, :], R_nli[i, 1, :] = As[:tmax], Bs[:tmax]
    #
    #     return E_nli, R_nli

    def save_trajectory(self):

        As, Bs, Cs, Ds = self.long_trajectory[0]
        As, Bs, Cs, Ds = np.array(As).astype(np.int8), np.array(Bs).astype(np.int8), np.array(Cs).astype(
            np.int8), np.array(Ds).astype(np.int8)
        dict_data = {
            "A": As,
            "B": Bs,
            "C": Cs,
            "D": Ds
        }
        self.dh.save_mat("trajectories", f"{self.prefix}trajectory", dict_data, do_compression=True)
        return

    def load_trajectory(self):
        try:
            if self.joch_data:
                self.joch_mat_data = self.dh.load_mat('joch_data', f'TwoChoiceTrajectoriesSmallStepDensity_{self.I}')
                As, Bs, Cs, Ds = self.joch_mat_data['r_li'][0][::self.ndt], self.joch_mat_data['r_li'][1][::self.ndt], self.joch_mat_data['e_li'][0][::self.ndt], self.joch_mat_data['e_li'][1][::self.ndt]
                self.long_trajectory.append((As, Bs, Cs, Ds))
                print(len(As),"DATA LENGTH")
            else:
                traj_data = self.dh.load_mat("trajectories", f"{self.prefix}trajectory")
                As, Bs, Cs, Ds = traj_data["A"], traj_data["B"], traj_data["C"], traj_data["D"]
                print(len(As),"DATA LENGTH")

                self.long_trajectory.append((As[0], Bs[0], Cs[0], Ds[0]))
        except Exception as e:
            print('something broke',e)





