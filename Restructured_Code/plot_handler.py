import numpy as np
import data_handler
import os
from matplotlib import pyplot as plt
import pandas as pd




def get_color_index( X, Y, num_colors ):
    angle = np.arctan2( Y, X)
    cix   = round( (angle + np.pi) / (2 * np.pi) * (num_colors - 1) )
    return cix


class PlotHandler:
    """Handle plotting operations for analysis results"""

    def __init__(self,Analysis):
        """Initialize PlotHandler with analysis handler instance"""
        self.ah = Analysis
        self.dh = data_handler.DataHandler()
        self.prefix = self.ah.prefix

    def display_dominance_statistics(self):
        header = ['I', 'Duration', 'Variance', 'CV', 'Skewness', 'CC1', 'CC2', 'CC3', 'RevCount', 'Pneither (%)']
        print(f"{'   |   '.join(header)}")
        print('-' * 120)
        row1 = [
            self.ah.I,
            self.ah.dominance_statistics["Duration1"],
            self.ah.dominance_statistics["Variance1"],
            self.ah.dominance_statistics["CV1"],
            self.ah.dominance_statistics["Skewness1"],
            self.ah.dominance_statistics["Sequential1"],
            self.ah.dominance_statistics["Sequential2"],
            self.ah.dominance_statistics["Sequential3"],
            self.ah.dominance_statistics["RevCount"],
            # model.dominance_statistics["Pdouble"],
            self.ah.dominance_statistics["Pneither"]
        ]
        row2 = [
            self.ah.I_test,
            self.ah.dominance_statistics["Duration2"],
            self.ah.dominance_statistics["Variance2"],
            self.ah.dominance_statistics["CV2"],
            self.ah.dominance_statistics["Skewness2"],
            self.ah.dominance_statistics["Sequential1"],
            self.ah.dominance_statistics["Sequential2"],
            self.ah.dominance_statistics["Sequential3"],
            self.ah.dominance_statistics["RevCount"],
            # model.dominance_statistics["Pdouble"],
            self.ah.dominance_statistics["Pneither"]
        ]
        print(
            f"{'  |  '.join(map(lambda x: f'{x.values[0]:.4f}' if isinstance(x, pd.Series) and isinstance(x.values[0], float) else str(x.values[0]) if isinstance(x, pd.Series) else str(x), row1))}")
        print(
            f"{'  |  '.join(map(lambda x: f'{x.values[0]:.4f}' if isinstance(x, pd.Series) and isinstance(x.values[0], float) else str(x.values[0]) if isinstance(x, pd.Series) else str(x), row2))}")
    def plot_2d_trajectory(self, tmax=-1):
        if hasattr(self.ah,'long_trajectory'):
            As, Bs, Cs, Ds = self.ah.long_trajectory
            Ts = range(len(As))
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

            ax1.plot(Ts[:tmax], As[:tmax], label='A')
            ax1.plot(Ts[:tmax], Bs[:tmax], label='B')
            ax1.legend()

            ax2.plot(Ts[:tmax], Cs[:tmax], label='C')
            ax2.plot(Ts[:tmax], Ds[:tmax], label='D')
            ax2.legend()

            self.dh.save_fig('plots', f'{self.prefix}TrajectoryL{len(As)}', plt)
            plt.show()
            return
    def plot_density_XYXb(self, thresh=400):
        """Plot trajectory density"""
        if hasattr(self.ah, 'countXYXb'):
            fs = 14

            u_space = self.ah.u_space
            uX, uY, uXb, uYb = u_space
            uX, uY, uXb, uYb = np.array(uX), np.array(uY), np.array(uXb), np.array(uYb)
            NE, NR = self.ah.maxbot - 1, self.ah.maxtop - 1
            maxuX, maxuY, maxuXb, maxuYb = len(uX), len(uY), len(uXb), len(uYb)

            # Count occurrences of unique combinations
            # countXYXb = list(model.countXYXb.values())[0]
            countXYXb = np.array(self.ah.countXYXb)

            # Threshold
            total_count = np.sum(countXYXb > 0)
            superthresh_count = np.sum(countXYXb > thresh)
            subthresh_fraction_xyxb = (total_count - superthresh_count) / total_count
            print(f"Sub-threshold fraction: {subthresh_fraction_xyxb:.4f}")

            countXYXb[countXYXb < thresh] = 0

            # Histogram of log(counts)
            logcount = np.log10(countXYXb[countXYXb > 0])
            edges = np.linspace(2, 5.5, 21)

            # plt.figure()
            # plt.hist(logcount, bins=edges)
            # plt.xticks([2, 3, 4, 5], ['1e2', '1e3', '1e4', '1e5'])
            # plt.title('Log Count Histogram')
            # plt.xlabel('Log10(Counts)')
            # plt.ylabel('Frequency')
            # plt.show()

            # 3D Scatter plot of long_trajectory
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            clrmp = plt.cm.jet
            Nclr = clrmp.N

            for m, X_val in enumerate(uX):
                for n, Y_val in enumerate(uY):
                    # Color based on normalized Y and X values
                    cix = get_color_index(Y_val / maxuY, X_val / maxuX, Nclr)
                    clri = clrmp(cix)

                    for p, Xb_val in enumerate(uXb):
                        cnt = countXYXb[m, n, p]
                        if cnt > 0:
                            # Marker size based on log(count)
                            marker_size = 2 * np.log(cnt) + 1
                            ax.scatter(
                                X_val / NE, Y_val / NR, Xb_val / NE,
                                color=clri, s=marker_size, label='_nolegend_'
                            )

            ax.set_xlim([-0.45, 0.45])
            ax.set_ylim([-1.0, 1.0])
            ax.set_zlim([0.0, 0.85])
            ax.set_xticks([-0.25, 0.0, 0.25])
            ax.set_yticks([-1, 0, 1])
            ax.set_zticks([0.0, 0.5])
            ax.set_box_aspect([1, 1, 1])  # Aspect ratio
            ax.set_xlabel('e-e\'', fontsize=fs)
            ax.set_ylabel('r-r\'', fontsize=fs)
            ax.set_zlabel('e+e\'', fontsize=fs)

            plt.title('Trajectory Density in 3D', fontsize=fs)
            self.dh.save_fig('plots', f'{self.prefix}DensityXYXb', plt)

            plt.show()

            return
    def plot_density_XYYb(self, thresh=400):
        """Plot trajectory density"""
        if hasattr(self.ah, 'countXYYb'):
            fs = 14

            u_space = self.ah.u_space
            uX, uY, uXb, uYb = u_space
            uX, uY, uXb, uYb = np.array(uX), np.array(uY), np.array(uXb), np.array(uYb)
            NE, NR = self.ah.maxbot - 1, self.ah.maxtop - 1
            maxuX, maxuY, maxuXb, maxuYb = len(uX), len(uY), len(uXb), len(uYb)

            # Count occurrences of unique combinations
            # countXYXb = list(model.countXYXb.values())[0]
            countXYYb = np.array(self.ah.countXYYb)

            # Threshold
            total_count = np.sum(countXYYb > 0)
            superthresh_count = np.sum(countXYYb > thresh)
            subthresh_fraction_xyyb = (total_count - superthresh_count) / total_count
            print(f"Sub-threshold fraction: {subthresh_fraction_xyyb:.4f}")

            countXYYb[countXYYb < thresh] = 0

            # Histogram of log(counts)
            logcount = np.log10(countXYYb[countXYYb > 0])
            edges = np.linspace(2, 5.5, 21)

            plt.figure()
            plt.hist(logcount, bins=edges)
            plt.xticks([2, 3, 4, 5], ['1e2', '1e3', '1e4', '1e5'])
            plt.title('Log Count Histogram')
            plt.xlabel('Log10(Counts)')
            plt.ylabel('Frequency')
            plt.show()

            # 3D Scatter plot of long_trajectory
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            clrmp = plt.cm.jet
            Nclr = clrmp.N

            for m, X_val in enumerate(uX):
                for n, Y_val in enumerate(uY):
                    # Color based on normalized Y and X values
                    cix = get_color_index(Y_val / maxuY, X_val / maxuX, Nclr)
                    clri = clrmp(cix)

                    for p, Yb_val in enumerate(uYb):
                        cnt = countXYYb[m, n, p]
                        if cnt > 0:
                            # Marker size based on log(count)
                            marker_size = 2 * np.log(cnt) + 1
                            ax.scatter(
                                X_val / NE, Y_val / NR, Yb_val / NE,
                                color=clri, s=marker_size, label='_nolegend_'
                            )

            ax.set_xlim([-0.45, 0.45])
            ax.set_ylim([-1.0, 1.0])
            ax.set_zlim([0.0, 0.85])
            ax.set_xticks([-0.25, 0.0, 0.25])
            ax.set_yticks([-1, 0, 1])
            ax.set_zticks([0.0, 0.5])
            ax.set_box_aspect([1, 1, 1])  # Aspect ratio
            ax.set_xlabel('e-e\'', fontsize=fs)
            ax.set_ylabel('r-r\'', fontsize=fs)
            ax.set_zlabel('e+e\'', fontsize=fs)

            plt.title('Trajectory Density in 3D', fontsize=fs)
            self.dh.save_fig('plots', f'{self.prefix}DensityXYYb', plt)

            plt.show()

            return
    def plot_avg_trajectories(self):
        """
        Plot the trajectory in 3D space with colored lines based on input data.

        Parameters:
        uX, uY, uXb: 1D arrays of X, Y, and Xb values.
        X_mnpi, Y_mnpi, Xb_mnpi: 4D arrays of trajectory data.
        NE, NR: Normalizing factors for X and Y axes.
        """
        uX, uY, uXb, _ = self.ah.u_space
        data = self.ah.average_trajectories
        # print(np.shape(data))
        # print(data[0])

        X_mnpi = data["X_mnpi"]
        Y_mnpi = data["Y_mnpi"]
        Xb_mnpi = data["Xb_mnpi"]

        X_std = data["X_std"]
        Y_std = data["Y_std"]
        Xb_std = data["Xb_std"]

        fs = 10  # Font size for labels and ticks

        M = len(uX)
        P = len(uXb)

        # Create the figure
        clrmp = plt.get_cmap('coolwarm')
        # clrmp = plt.get_cmap('jet')

        # num_colors = 256
        num_colors = 256
        clrmp_array = clrmp(np.linspace(0, 1, num_colors))
        maxuX = np.max(uX)
        maxuY = np.max(uY)

        fig = plt.figure(figsize=(15, 7))

        # First subplot

        ax1 = fig.add_subplot(1, 2, 1, projection='3d', position=[0.05, 0.1, 0.6, 0.8])
        ax2 = fig.add_subplot(1, 2, 2, aspect='auto', position=[0.7, 0.1, 0.25, 0.8])

        ax1.set_xlim([-0.5, 0.5])
        ax1.set_ylim([-1.1, 1.1])
        ax1.set_zlim([0.0, 0.85])

        ax2.set_xlim([-0.5, 0.5])
        ax2.set_ylim([-1.1, 1.1])

        # Now get the axis limits to calculate the scale factors
        xlim = ax1.get_xlim()
        ylim = ax1.get_ylim()
        zlim = ax1.get_zlim()

        # Calculate the data range for each axis
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        z_range = zlim[1] - zlim[0]

        # Scale factors based on axis ranges
        x_scale = 1 / x_range
        y_scale = 1 / y_range
        z_scale = 1 / z_range

        for m in range(0, M):
            nX = uX[m]
            for n in range(0, 2):

                if n == 0:
                    nY = min(uY)
                else:
                    nY = max(uY)

                cix = get_color_index(nX / maxuX, nY / maxuY, num_colors)
                for p in range(0, P):

                    X_i = np.squeeze(X_mnpi[m, n, p, :])
                    L = len(X_i)
                    Y_i = np.squeeze(Y_mnpi[m, n, p, :])
                    Xb_i = np.squeeze(Xb_mnpi[m, n, p, :])

                    x_std_i = np.squeeze(X_std[m, n, p, :])
                    y_std_i = np.squeeze(Y_std[m, n, p, :])
                    xb_std_i = np.squeeze(Xb_std[m, n, p, :])

                    if abs(Y_i[0]) > 0.95:
                        # ax1
                        num_points = int(L)
                        ax1.scatter(X_i, Y_i, Xb_i, marker='.', color=clrmp_array[cix, :], linewidth=.6, zorder=1,
                                    alpha=0.95)
                        # thin lines between points plotted
                        initial_width = 0.3
                        max_value = 10  # Set the maximum value to cap the linewidth growth
                        initial_width * (1 + x_std_i + y_std_i + xb_std_i) / max_value
                        ax1.plot(X_i, Y_i, Xb_i, color=clrmp_array[cix, :] * .7, linewidth=0.3, zorder=2, alpha=0.9)

                        # add arrows
                        for i in range(0, num_points - 2, int(num_points / 4)):
                            direction = np.array([X_i[i + 1] - X_i[i], Y_i[i + 1] - Y_i[i], Xb_i[i + 1] - Xb_i[i]])
                            direction = np.array(
                                [direction[0] * x_scale, direction[1] * y_scale, direction[2] * z_scale])
                            norm = np.linalg.norm(direction)
                            if norm > 0:
                                direction /= norm
                            length = .08

                            arrow_V = length * direction
                            if abs(Y_i[i]) <= 0.9:
                                ax1.quiver(X_i[i], Y_i[i], Xb_i[i], arrow_V[0] / x_scale, arrow_V[1] / y_scale,
                                           arrow_V[2] / z_scale, color=clrmp_array[cix, :], linewidth=2.5, zorder=5)
                                # ax1.quiver(X_i[i], Y_i[i], Xb_i[i], arrow_V[0]/x_scale, arrow_V[1]/y_scale, arrow_V[2]/z_scale, color='black', linewidth=2.5, zorder=5, alpha=0.8)

                        ihalf = int(num_points / 2)

                        ## DIRECTION OF ARROWS
                        direction2 = np.array([X_i[ihalf + 1] - X_i[ihalf], Y_i[ihalf + 1] - Y_i[ihalf]])
                        direction2 = np.array([direction2[0] * x_scale, direction2[1] * y_scale])

                        norm2 = np.linalg.norm(direction2)
                        if norm2 > 0:
                            direction2 /= norm2

                        length = .08
                        arrow_V2 = length * direction2
                        ##
                        # PLOT 2
                        if (Xb_i[ihalf]) >= 0.4:
                            # add a thin line between scatter points
                            ax2.plot(X_i, Y_i, color=clrmp_array[cix, :], linewidth=1.3, zorder=2, alpha=0.9)
                            ax2.scatter(X_i, Y_i, marker='.', color=clrmp_array[cix, :], linewidth=1.3)
                            ax2.quiver(X_i[ihalf], Y_i[ihalf], arrow_V2[0] / x_scale, arrow_V2[1] / y_scale,
                                       color='black', linewidth=.9, zorder=5, alpha=0.8)

        ax1.set_xlabel('X', fontsize=fs)
        ax1.set_ylabel('Y', fontsize=fs)
        ax1.set_zlabel('X bar', fontsize=fs)
        ax1.set_title('Trajectory flow', fontsize=fs)

        ax2.set_xlabel('X', fontsize=fs)
        ax2.set_ylabel('Y', fontsize=fs)
        ax1.view_init(elev=38, azim=-41)

        # save plot to 'trajectory_plots' dir


        self.dh.save_fig('plots', f'{self.prefix}Tper{self.ah.Tper}Ntraj{self.ah.Ntraj}avg_traj', plt)
        # fix start view angle
        plt.show()
        self.close_plots()
    def plot_gamma_distribution(self, axs):
        gamma_dist = self.ah.gamma_dist
        axs[0, 0].hist(gamma_dist, bins=120, color='blue', alpha=0.7)
        axs[0, 0].set_title('Gamma Distribution')
        axs[0, 0].set_xlabel('Value')
        axs[0, 0].set_ylabel('Frequency')

    def plot_lower_distribution(self, axs):
        lower_dist = self.ah.lower_dist
        axs[0, 1].bar(range(len(lower_dist)), lower_dist, color='green', alpha=0.7)
        axs[0, 1].set_title('Lower Distribution')
        axs[0, 1].set_xlabel('Index')
        axs[0, 1].set_ylabel('Probability')

    def plot_ac_distribution(self, axs, fig):
        ac_dist = self.ah.ac_dist
        cax1 = axs[1, 0].imshow(ac_dist, aspect='auto', cmap='viridis')
        fig.colorbar(cax1, ax=axs[1, 0])
        axs[1, 0].set_title('AC Distribution')
        axs[1, 0].set_xlabel('C Index')
        axs[1, 0].set_ylabel('A Index')

    def plot_ad_distribution(self, axs, fig):
        ad_dist = self.ah.ad_dist
        cax2 = axs[1, 1].imshow(ad_dist, aspect='auto', cmap='viridis')
        fig.colorbar(cax2, ax=axs[1, 1])
        axs[1, 1].set_title('AD Distribution')
        axs[1, 1].set_xlabel('D Index')
        axs[1, 1].set_ylabel('A Index')



    def close_plots(self):
        plt.close('all')


class AnalysisPlot:
    def __init__(self,nrows,ncols,figsize = (12,9)):
        self.fig, self.axs = plt.subplots(nrows,ncols,figsize=figsize)
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = figsize
        self.subplots = []

        if nrows == 1 and ncols == 1:
            self.axs = np.array([[self.axs]])
        elif nrows ==1 or ncols ==1:
            self.axs = self.axs.reshape(nrows,ncols)

    def add_subplot(self,subplot):
        self.subplots.append(subplot)

    def show(self,title = '',save=False,saveName='plot'):
        for subplot in self.subplots:
            projection = '3d' if subplot.plot_type in ['average_trajectory3d','density'] else None
            ax = self.axs[subplot.row,subplot.col]
            if projection:
                self.fig.delaxes(ax)
                if self.ncols==1:
                    ax = self.fig.add_subplot(self.nrows,self.ncols,subplot.row*subplot.col+subplot.row + 1,projection='3d')
                else:
                    ax = self.fig.add_subplot(self.nrows,self.ncols,subplot.row*subplot.col+subplot.col + 1,projection='3d')
            subplot.apply_plot(ax)
        self.fig.suptitle(title,fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(right=0.95,wspace=0.3, hspace=0.3)
        if save:
            self.save(saveName)
        plt.show()

    def save(self, filename,dir='data/plots'):
        os.makedirs(dir, exist_ok=True)
        self.fig.savefig(os.path.join(dir, filename))

class SubPlot:
    def __init__(self, position, plot_type,data,uspace=None):
        self.row = position[0]
        self.col = position[1]
        self.data = data
        self.plot_type = plot_type
        self.uspace=uspace

    def display_dominance_statistics(self):
     data = self.data
     header = ['Duration', 'Variance', 'CV', 'Skewness', 'CC1', 'CC2', 'CC3', 'RevCount', 'Pdouble', 'Pneither']
     print(f"{'  |  '.join(header)}")
     print('-' * (len(header) * 10))

     for key in data.keys():
         if isinstance(data[key], str) and data[key].startswith('[') and data[key].endswith(']'):
             data[key] = np.array(data[key].replace('[', '').replace(']', '').split()).astype(float)

     row1 = [
         data["Duration"][0],
         data["Variance"][0],
         data["CV"][0],
         data["Skewness"][0],
         data["Sequential"][0],
         data["Sequential"][1],
         data["Sequential"][2],
         data["RevCount"],
         data["Pdouble"],
         data["Pneither"]
     ]
     row2 = [
         data["Duration"][1],
         data["Variance"][1],
         data["CV"][1],
         data["Skewness"][1],
         data["Sequential"][0],
         data["Sequential"][1],
         data["Sequential"][2],
         data["RevCount"],
         data["Pdouble"],
         data["Pneither"]
     ]

     row_format = "{:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}"
     print(row_format.format(*[f"{x:.4f}" if isinstance(x, float) else str(x) for x in row1]))
     print(row_format.format(*[f"{x:.4f}" if isinstance(x, float) else str(x) for x in row2]))


    def apply_plot(self, ax):

       plot_methods = {
           'gamma': self.plot_gamma_distribution,
           'lower': self.plot_lower_distribution,
           'ac': self.plot_ACAD_distribution,
           'ad': self.plot_ACAD_distribution,
           'average_trajectory3d': self.plot_avg_trajectories3d,
           'average_trajectorytopdown': self.plot_avg_trajectoriestopdown,
           'density': lambda ax, data: None,
           'basic':self.plot_basic_trajectory
       }

       plot_method = plot_methods.get(self.plot_type)
       if plot_method:
          if self.plot_type in ['average_trajectory3d', 'average_trajectorytopdown']:
              plot_method(ax, self.data, self.uspace)
          else:
              plot_method(ax, self.data)

    def plot_gamma_distribution(self,ax,data):
        ax.hist(data, bins=120, color='blue', alpha=0.7)
        if self.row ==0:
            ax.set_title('Gamma Distribution')
        ax.set_xlabel('Flipping Time')
        ax.set_ylabel('Frequency')
        ax.set_xlim([0,9e4])
        ax.set_ylim([0,200])

    def plot_lower_distribution(self,ax,data):
        ax.bar(range(len(data)), data, color='green', alpha=0.7)
        if self.row == 0:
            ax.set_title('Lower Distribution')
        ax.set_xlabel('Index')
        ax.set_ylabel('Probability')
    def plot_ACAD_distribution(self,ax,data):

        ax.imshow(data, aspect='auto', cmap='viridis')
        if self.row == 0:
            ax.set_title(f'{self.plot_type} Distribution')
        if self.plot_type == 'ac':
            ax.set_xlabel('C Index')
        elif self.plot_type == 'ad':
            ax.set_xlabel('D Index')
        ax.set_ylabel('A Index')


    def plot_avg_trajectoriestopdown(self,ax,data,uspace):
        """
        Plot the trajectory in 3D space with colored lines based on input data.

        Parameters:
        uX, uY, uXb: 1D arrays of X, Y, and Xb values.
        X_mnpi, Y_mnpi, Xb_mnpi: 4D arrays of trajectory data.
        NE, NR: Normalizing factors for X and Y axes.
        """
        uX, uY, uXb, _ = uspace


        X_mnpi = data["X_mnpi"]
        Y_mnpi = data["Y_mnpi"]
        Xb_mnpi = data["Xb_mnpi"]

        X_std = data["X_std"]
        Y_std = data["Y_std"]
        Xb_std = data["Xb_std"]

        fs = 10  # Font size for labels and ticks

        M = len(uX)
        P = len(uXb)

        # Create the figure
        clrmp = plt.get_cmap('coolwarm')
        # clrmp = plt.get_cmap('jet')

        # num_colors = 256
        num_colors = 256
        clrmp_array = clrmp(np.linspace(0, 1, num_colors))
        maxuX = np.max(uX)
        maxuY = np.max(uY)

        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-1.1, 1.1])

        # Now get the axis limits to calculate the scale factors
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Calculate the data range for each axis
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        # Scale factors based on axis ranges
        x_scale = 1 / x_range
        y_scale = 1 / y_range
        for m in range(0, M):
            nX = uX[m]
            for n in range(0, 2):

                if n == 0:
                    nY = min(uY)
                else:
                    nY = max(uY)

                cix = get_color_index(nX / maxuX, nY / maxuY, num_colors)
                for p in range(0, P):

                    X_i = np.squeeze(X_mnpi[m, n, p, :])
                    L = len(X_i)
                    Y_i = np.squeeze(Y_mnpi[m, n, p, :])
                    Xb_i = np.squeeze(Xb_mnpi[m, n, p, :])

                    x_std_i = np.squeeze(X_std[m, n, p, :])
                    y_std_i = np.squeeze(Y_std[m, n, p, :])
                    xb_std_i = np.squeeze(Xb_std[m, n, p, :])

                    if abs(Y_i[0]) > 0.95:
                        # ax1
                        num_points = int(L)
                        ihalf = int(num_points / 2)

                        ## DIRECTION OF ARROWS
                        direction2 = np.array([X_i[ihalf + 1] - X_i[ihalf], Y_i[ihalf + 1] - Y_i[ihalf]])
                        direction2 = np.array([direction2[0] * x_scale, direction2[1] * y_scale])

                        norm2 = np.linalg.norm(direction2)
                        if norm2 > 0:
                            direction2 /= norm2

                        length = .08
                        arrow_V2 = length * direction2
                        ##
                        # PLOT 2
                        if (Xb_i[ihalf]) >= 0.4:
                            # add a thin line between scatter points
                            ax.plot(X_i, Y_i, color=clrmp_array[cix, :], linewidth=1.3, zorder=2, alpha=0.9)
                            ax.scatter(X_i, Y_i, marker='.', color=clrmp_array[cix, :], linewidth=1.3)
                            ax.quiver(X_i[ihalf], Y_i[ihalf], arrow_V2[0] / x_scale, arrow_V2[1] / y_scale,
                                       color='black', linewidth=.6, zorder=5, alpha=0.7)



        ax.set_xlabel('X', fontsize=fs)
        ax.set_ylabel('Y', fontsize=fs)
        if self.row==0:
            ax.set_title('Trajectory flow (top down)', fontsize=fs)
        # save plot to 'trajectory_plots' dir
    def plot_avg_trajectories3d(self,ax,data,uspace):


        uX, uY, uXb, _ = uspace
        # print(np.shape(data))
        # print(data[0])

        X_mnpi = data["X_mnpi"]
        Y_mnpi = data["Y_mnpi"]
        Xb_mnpi = data["Xb_mnpi"]

        X_std = data["X_std"]
        Y_std = data["Y_std"]
        Xb_std = data["Xb_std"]

        fs = 10  # Font size for labels and ticks

        M = len(uX)
        P = len(uXb)

        # Create the figure
        clrmp = plt.get_cmap('coolwarm')
        # clrmp = plt.get_cmap('jet')

        # num_colors = 256
        num_colors = 256
        clrmp_array = clrmp(np.linspace(0, 1, num_colors))
        maxuX = np.max(uX)
        maxuY = np.max(uY)


        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-1.1, 1.1])
        ax.set_zlim([0.0, 0.85])


        # Now get the axis limits to calculate the scale factors
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()

        # Calculate the data range for each axis
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        z_range = zlim[1] - zlim[0]

        # Scale factors based on axis ranges
        x_scale = 1 / x_range
        y_scale = 1 / y_range
        z_scale = 1 / z_range

        for m in range(0, M):
            nX = uX[m]
            for n in range(0, 2):

                if n == 0:
                    nY = min(uY)
                else:
                    nY = max(uY)

                cix = get_color_index(nX / maxuX, nY / maxuY, num_colors)
                for p in range(0, P):

                    X_i = np.squeeze(X_mnpi[m, n, p, :])
                    L = len(X_i)
                    Y_i = np.squeeze(Y_mnpi[m, n, p, :])
                    Xb_i = np.squeeze(Xb_mnpi[m, n, p, :])

                    x_std_i = np.squeeze(X_std[m, n, p, :])
                    y_std_i = np.squeeze(Y_std[m, n, p, :])
                    xb_std_i = np.squeeze(Xb_std[m, n, p, :])

                    if abs(Y_i[0]) > 0.95:
                        # ax1
                        num_points = int(L)
                        ax.scatter(X_i, Y_i, Xb_i, marker='.', color=clrmp_array[cix, :], linewidth=.6, zorder=1,
                                    alpha=0.95)
                        # thin lines between points plotted
                        initial_width = 0.3
                        max_value = 10  # Set the maximum value to cap the linewidth growth
                        initial_width * (1 + x_std_i + y_std_i + xb_std_i) / max_value
                        ax.plot(X_i, Y_i, Xb_i, color=clrmp_array[cix, :] * .7, linewidth=0.3, zorder=2, alpha=0.9)

                        # add arrows
                        for i in range(0, num_points - 2, int(num_points / 4)):
                            direction = np.array([X_i[i + 1] - X_i[i], Y_i[i + 1] - Y_i[i], Xb_i[i + 1] - Xb_i[i]])
                            direction = np.array(
                                [direction[0] * x_scale, direction[1] * y_scale, direction[2] * z_scale])
                            norm = np.linalg.norm(direction)
                            if norm > 0:
                                direction /= norm
                            length = .08

                            arrow_V = length * direction
                            if abs(Y_i[i]) <= 0.9:
                                ax.quiver(X_i[i], Y_i[i], Xb_i[i], arrow_V[0] / x_scale, arrow_V[1] / y_scale,
                                           arrow_V[2] / z_scale, color=clrmp_array[cix, :], linewidth=2.5, zorder=5)
                                # ax1.quiver(X_i[i], Y_i[i], Xb_i[i], arrow_V[0]/x_scale, arrow_V[1]/y_scale, arrow_V[2]/z_scale, color='black', linewidth=2.5, zorder=5, alpha=0.8)

        ax.set_xlabel('X', fontsize=fs)
        ax.set_ylabel('Y', fontsize=fs)
        ax.set_zlabel('X bar', fontsize=fs)
        if self.row == 0:
            ax.set_title('Trajectory flow', fontsize=fs)
        ax.view_init(elev=38, azim=-41)
    def plot_basic_trajectory(self,ax,data):

        As, Bs, Cs, Ds = data
        Ts = range(len(As))
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        tmax = -1
        ax1.plot(Ts[:tmax], As[:tmax], label='A')
        ax1.plot(Ts[:tmax], Bs[:tmax], label='B')
        ax1.legend()

        ax2.plot(Ts[:tmax], Cs[:tmax], label='C')
        ax2.plot(Ts[:tmax], Ds[:tmax], label='D')
        ax2.legend()
        plt.show()





