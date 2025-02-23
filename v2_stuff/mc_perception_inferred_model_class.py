from mc_perception_frwd_cl import simulation_nopij
from mc_perception_analyzer_functions import extract_dominance_statistics,compute_average_trajectories,save_analysis,load_analysis,get_shifted_trajectory_densities,get_shifted_trajectory_and_space
from v2_stuff.data_inferred_chunker import chunk_and_save_data, load_chunked_data
from mc_plotter import plot_2d_trajectory,view_trajectory_density_xyxb,view_trajectory_density_xyyb,view_avg_trajectories
from mc_perception_model_finder import maximize_likelyhood,save_inferred_model_csv,count_transitions, load_mat_data
import os
import numpy as np
import pandas as pd
import ast

""" Class for brain model, holds long_trajectory and provides methods for trajectory generation, analysis, and plotting. """

class brain_model:
	
    def __init__(self, params, I_test, maxtop, maxbot,dt,Tend,HC,VC,joch_data=False):
        self.I_test = I_test
        self.params = params
        self.maxtop = maxtop
        self.maxbot = maxbot
        self.HC=HC
        self.VC=VC
        self.Tend = Tend
        self.countXYXb = None
        self.countXYYb = None
        self.dominance_statistics = None
        self.average_trajectories = None
        self.joch_data = joch_data

        self.long_trajectory = []
        self.dt = dt


        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.Joch_data_given = os.path.join(current_dir, 'Joch_data_given')
        self.param_dir = os.path.join(current_dir, 'Infered_parameters')
        self.inferred_trajectories = os.path.join(current_dir, 'inferred_trajectories')
        self.perception_cl_prog = os.path.join(current_dir, 'mc_perception_opencl.cl')
        self.analysis_data= os.path.join(current_dir,'analysis_data')
        self.inferred_trajectories_path = os.path.join(self.inferred_trajectories, f'I{self.I_test}_HC{self.HC}_VC{self.VC}_L{self.Tend}')
        #self.inferred_trajectories_path = os.path.join(self.inferred_trajectories, f'I{self.I_test}_L{self.Tend}')

    def choose_params(self):
        # get user input for file in Infered_parameters:
        print("Available files in Infered_parameters: ")
        for idx,file in enumerate(os.listdir(self.param_dir)):
            print(file,idx)
        choice = input("Enter the file # name of the parameters you would like to use: ")
        file_name = os.listdir(self.param_dir)[int(choice)]
        print(f'Choice {file_name}')
        df = pd.read_csv(os.path.join(self.param_dir, file_name))
        self.params = ast.literal_eval(df.iloc[0, 1])
        return
    def find_model(self, initial_guess):
        directory = 'Joch_data_given'
        counts_file = os.path.join(directory, f'counts_{self.I_test}_VC_{self.VC}_HC_{self.HC}_dt{self.dt}.npy')

        lock_file = r"C:\Users\Nick\AppData\Local\pyopencl\pyopencl\Cache\pyopencl-compiler-cache-v2-py3.12.7.final.0\lock"
        if os.path.exists(lock_file):
            os.remove(lock_file)
            print("Lock file deleted successfully.")
        else:
            print("Lock file not found.")

        if os.path.exists(counts_file):
            count = np.load(counts_file, allow_pickle=True).item()
            print('LOADED COUNTS FROM FILE')
        else:
            #joch_a, joch_b, joch_c, joch_d = load_mat_data(os.path.join(directory,f'TwoChoiceTrajectoriesDensity_{self.I_test}.mat'))
            #joch_a, joch_b, joch_c, joch_d = load_mat_data(os.path.join(directory,f'TwoChoiceAmputatedDensity_{self.I_test}_VC_{self.VC}_HC_{self.HC}.mat'))
            joch_a, joch_b, joch_c, joch_d = load_mat_data(os.path.join(directory,f'TwoChoiceTrajectoriesSmallStepDensity_{self.I_test}.mat'))

            count = count_transitions((joch_a, joch_b, joch_c, joch_d))
            np.save(counts_file, count)
            print('FOUND COUNTS AND SAVED')

        maxparams=maximize_likelyhood(self,count,initial_guess)
        print(f'Inferred parameters: {maxparams}')
        self.params = maxparams
        #save_inferred_model_csv(self.I_test,f'VC_{self.VC}_HC_{self.HC}',self.params)
        save_inferred_model_csv(self.I_test,f'VC_{self.VC}_HC_{self.HC}V4',self.params)


        return

    # # # Trajectory Generation # # #
############################################################################
    def generate_trajectory(self, initial_state, tmax):
        inferred_trajectory = simulation_nopij(self,initial_state, int(tmax/self.dt), self.params, self.perception_cl_prog)
        self.long_trajectory.append(inferred_trajectory)
		
        return inferred_trajectory
    def generate_multiple_trajectories(self, initial_state, tmax, num_trajectories):

        E_nli = np.zeros((num_trajectories, 2, tmax))
        R_nli = np.zeros((num_trajectories, 2, tmax))

        for i in range(num_trajectories):
            traj = simulation_nopij(self,initial_state, tmax, self.params, self.perception_cl_prog)
            As,Bs,Cs,Ds = traj
            E_nli[i, 0, :], E_nli[i, 1, :] = Cs[:tmax], Ds[:tmax]
            R_nli[i, 0, :], R_nli[i, 1, :] = As[:tmax], Bs[:tmax]

        return E_nli, R_nli

	
    def save_trajectory(self):
	
        if not os.path.exists(self.inferred_trajectories):
            os.makedirs(self.inferred_trajectories)

        As, Bs, Cs, Ds = self.long_trajectory[0]

        chunk_and_save_data(self, As,'A')
        chunk_and_save_data(self, Bs,'B')
        chunk_and_save_data(self, Cs,'C')
        chunk_and_save_data(self, Ds,'D')

        print(f'{self.I_test}Trajectories saved to {self.inferred_trajectories_path}')

        return
    def load_trajectory(self,short = False):
        """
	Load long_trajectory from a file.
        """
        if not self.joch_data:
            As = load_chunked_data(self,  'A', short = short)
            Bs = load_chunked_data(self, 'B', short = short)
            Cs = load_chunked_data(self,  'C', short = short)
            Ds = load_chunked_data(self,  'D', short = short)

            self.long_trajectory.append((As, Bs, Cs, Ds))
            self.shifted_trajectory,self.u_space= get_shifted_trajectory_and_space(self.long_trajectory[0])
            print('loaded and converted to shifted trajectory')

        else:
            a,b,c,d = load_mat_data(os.path.join(self.Joch_data_given,f'TwoChoiceTrajectoriesSmallStepDensity_{self.I_test}.mat'))
            self.long_trajectory.append((a,b,c,d))

            self.shifted_trajectory,self.u_space= get_shifted_trajectory_and_space((a,b,c,d))
            self.countXYXb,self.countXYYb = get_shifted_trajectory_densities(self.shifted_trajectory,self.u_space)

            print('loaded joch data')
        return

    # # # Trajectory Analysis # # #
#############################################################################
    def dominance_statistics_trajectories(self):
        """
	Analyze the loaded or generated long_trajectory.
	"""
        traj = self.long_trajectory[0]
        As, Bs, Cs, Ds = traj

        #check if data is in directory 'trajectory statistics' or not
        try:
            self.dominance_statistics = self.load_analysis("dominance_statistics")
        except:
            print('No dominance statistics found, calculating now')
            #convert As,Bs to 2 dd array of dtype int8
            As,Bs = np.array(As,dtype=np.int8),np.array(Bs,dtype=np.int8)
            R_ki = np.array([As,Bs])
            Duration, Variance, CV, Skewness, Sequential, RevCount, Pdouble, Pneither = extract_dominance_statistics(self.dt,R_ki, NR=[[self.maxtop-1]])
            print(f"Duration: {Duration}, Variance: {Variance}, CV: {CV}, Skewness: {Skewness}, Sequential: {Sequential}, RevCount: {RevCount}, Pdouble: {Pdouble}, Pneither: {Pneither}")

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
            self.save_analysis()
    def get_trajectory_density(self):
        if self.long_trajectory == []:
            self.load_trajectory()
        traj = self.long_trajectory[0]

        try:
            self.countXYXb, self.countXYYb = self.load_analysis("density_countsXYXb"),self.load_analysis("density_countsXYYb")
            print('Trajectory density found and loaded\n')

        except:
            print('No trajectory density found, calculating now\n')
            self.shifted_trajectory, self.u_space = get_shifted_trajectory_and_space(traj)
            self.countXYXb,self.countXYYb = get_shifted_trajectory_densities( self.shifted_trajectory,self.u_space)
            self.save_analysis()

        return self.countXYXb,self.countXYYb
    def get_avg_trajectory_flow(self,Tper,Nrepeat):
        self.L_per_traj = Tper
        self.Nrepeat = Nrepeat
        if self.joch_data:
            self.average_trajectories= compute_average_trajectories(self,Tper,Nrepeat)
            return

        try:
            self.average_trajectories = self.load_analysis("average_trajectories")
            print('Average trajectory found and loaded\n')

        except:
            print('No average trajectory found, calculating now')
            self.average_trajectories = compute_average_trajectories(self,  Tper, Nrepeat)
            self.save_analysis()




    def save_analysis(self):
        """
        Save the analysis to a file.
        """
        save_analysis(self)
        return
    def load_analysis(self,file_type):
        """
        Load the analysis from a file.
        """
        data = load_analysis(self,file_type)
        return data
	


    # # # Plotting # # #
############################################################################
    def plot_trajectory(self,  sample_size):
        """
        Plot a given trajectory.
        """
        print(np.shape(self.long_trajectory[0]))
        plot_2d_trajectory(self.long_trajectory[0], sample_size)

        return
    def plot_trajectory_density(self, thresh=400):
        """
        Plot the trajectory density.
        """
        if self.countXYXb is not None:
            view_trajectory_density_xyxb(self, thresh)
            view_trajectory_density_xyyb(self, thresh)
        else:
            self.countXYXb,self.countYXYb = self.get_trajectory_density()
            view_trajectory_density_xyxb(self, thresh)
            view_trajectory_density_xyyb(self, thresh)

        return
    def plot_avg_trajectory(self):
        """
        Plot the average trajectory.
        """
        if self.average_trajectories is not None:
            view_avg_trajectories(self)

        else:
            self.get_avg_trajectory_flow(.001,10)
            print('getting average trajectory for tper, Nrepeat',.001,10)
            view_avg_trajectories(self)


        return
    def save_plot(self, plot, file_path):
        """
        Save the plot to a file.
        """
        pass
