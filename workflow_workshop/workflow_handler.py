import os
import importlib
import json
from typing import Dict, Any, Optional, List, Union


class ConfigurationManager:
    """Manages loading and saving of configuration files."""

    def __init__(self, config_dir: str = 'predefined_configurations'):
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)

    def list_configurations(self) -> List[str]:
        """List all available configuration files."""
        return os.listdir(self.config_dir)

    def load_configuration(self, filename: str) -> Dict[str, Any]:
        """Load a configuration from a file."""
        file_path = os.path.join(self.config_dir, filename)
        with open(file_path, 'r') as f:
            return json.load(f)

    def save_configuration(self, config: Dict[str, Any], filename: str) -> None:
        """Save a configuration to a file."""
        file_path = os.path.join(self.config_dir, filename)
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=4)

    def get_user_config_choice(self,
                               prompt: str = "Enter the index of the configuration you want to use (or 'm' for manual): ") -> \
    Union[str, int]:
        """Let the user choose a configuration or enter manual mode."""
        config_list = self.list_configurations()
        for index, config in enumerate(config_list):
            print(f"{index}: {config}")

        choice = input(prompt)
        if choice.lower() == 'm':
            return 'm'
        else:
            return int(choice)


class WorkflowManager:
    """Manages the entire workflow process including model setup, analysis, and plotting."""

    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.model = None
        self.analysis_handler = None
        self.plot_handler = None
        self.settings = {}
        self.workflow = {}

    def initialize(self):
        """Initialize the workflow by loading settings and workflow options."""
        self.load_settings()
        self.load_workflow()

    def load_settings(self) -> Dict[str, Any]:
        """Load model settings from configuration or manual input."""
        choice = self.config_manager.get_user_config_choice()

        if choice == 'm':
            self.settings = {
                "I": input("Enter I value: "),
                "NR": int(input("Enter NR value: ")),
                "NE": int(input("Enter NE value: ")),
                "dt": float(input("Enter dt value: ")),
                "joch_data": input("Use Joch data? (true/false): ").strip().lower() == "true",
            }
        else:
            config_list = self.config_manager.list_configurations()
            config = self.config_manager.load_configuration(config_list[choice])
            self.settings = config.get('settings', {})

        return self.settings

    def load_workflow(self) -> Dict[str, Any]:
        """Load workflow options from configuration or manual input."""
        choice = self.config_manager.get_user_config_choice(
            "Enter the index of the workflow configuration (or 'm' for manual): ")

        if choice == 'm':
            self.workflow = {
                "parameters": input("Find parameters? (infer/load/none): ").strip().lower(),
                "trajectory": input("Generate or load trajectory? (generate/load/none): ").strip().lower(),
                "densities": input("Generate or load density? (generate/load/none): ").strip().lower(),
                "average_distributions": input(
                    "Generate or load average distributions? (generate/load/none): ").strip().lower(),
                "average_trajectory": input(
                    "Generate or load average trajectory flows? (generate/load/none): ").strip().lower(),
                "generate_plots": input("Generate plots? (yes/no): ").strip().lower() == "yes",
                "comparison": input("Perform comparison with another model? (yes/no): ").strip().lower() == "yes",
                "save_plots": input("Save plots? (yes/no): ").strip().lower() == "yes"
            }
        else:
            config_list = self.config_manager.list_configurations()
            config = self.config_manager.load_configuration(config_list[choice])
            self.workflow = config.get('workflow', {})

        return self.workflow

    def create_model(self):
        """Create and initialize the model based on settings."""
        # Import here to avoid circular imports
        import maxcal_model_handler as mc

        self.model = mc.BrainModel(
            self.settings["I"],
            self.settings["NR"],
            self.settings["NE"],
            self.settings["dt"],
            joch_data=self.settings.get("joch_data", False)
        )

        return self.model

    def initialize_analysis(self):
        """Initialize the analysis handler with the current model."""
        # Import here to avoid circular imports
        from analysis_handler import AnalysisHandler

        if self.model is None:
            self.create_model()

        self.analysis_handler = AnalysisHandler(self.model)
        return self.analysis_handler

    def initialize_plotting(self):
        """Initialize the plot handler."""
        # Import here to avoid circular imports
        from plot_handler import PlotHandler

        self.plot_handler = PlotHandler()
        return self.plot_handler

    def save_current_configuration(self, filename: str):
        """Save the current configuration for future use."""
        config = {
            'settings': self.settings,
            'workflow': self.workflow
        }
        self.config_manager.save_configuration(config, filename)
        print(f"Configuration saved as {filename}")

    def run_workflow(self):
        """Execute the workflow based on the current settings and workflow options."""
        if not self.settings or not self.workflow:
            self.initialize()

        # Create model if not already created
        if self.model is None:
            self.create_model()

        # Handle parameters
        if self.workflow.get('parameters') == 'infer':
            print("Inferring parameters...")
            self.model.find_model()
        elif self.workflow.get('parameters') == 'load':
            print("Loading parameters...")
            self.model.load_params()
            print(f"Loaded parameters: {self.model.params}")

        # Handle trajectory
        if self.workflow.get('trajectory') == 'generate':
            print("Generating trajectory...")
            initial_state = tuple(
                map(float, input("Enter initial state (comma-separated, e.g., 0,0,0,0): ").split(',')))
            duration = float(input("Enter trajectory duration: "))
            self.model.generate_trajectory(initial_state, duration)

            if input("Save trajectory? (yes/no): ").strip().lower() == 'yes':
                self.model.save_trajectory()
        elif self.workflow.get('trajectory') == 'load':
            print("Loading trajectory...")
            self.model.load_trajectory()

        # Initialize analysis if needed
        if self.analysis_handler is None:
            self.initialize_analysis()

        # Handle densities and distributions
        if self.workflow.get('densities') == 'generate':
            print("Generating trajectory density...")
            self.analysis_handler.load_or_generate_trajectory_density()

        if self.workflow.get('average_distributions') == 'generate':
            print("Generating average distributions...")
            self.analysis_handler.load_or_generate_distributions()

        if self.workflow.get('average_trajectory') == 'generate':
            print("Generating average trajectory flows...")
            tper = float(input("Enter Tper value (default: 0.03): ") or "0.03")
            ntraj = int(input("Enter Ntraj value (default: 10000): ") or "10000")
            self.analysis_handler.load_or_generate_avg_trajectory(Tper=tper, Ntraj=ntraj)

        # Generate plots if requested
        if self.workflow.get('generate_plots', False):
            self.generate_plots(save=self.workflow.get('save_plots', False))

        # Handle comparison if requested
        if self.workflow.get('comparison', False):
            self.run_comparison()

    def generate_plots(self, save=False):
        """Generate plots based on current analysis results."""
        if self.analysis_handler is None:
            print("No analysis results available. Run analysis first.")
            return

        # Import here to avoid circular imports
        from plot_handler import AnalysisPlot, SubPlot

        # Show dominance statistics
        p = SubPlot([0, 0], 'dominance', self.analysis_handler.dominance_statistics)
        p.display_dominance_statistics()

        # Show distributions
        if hasattr(self.analysis_handler, 'gamma_dist') and hasattr(self.analysis_handler, 'lower_dist'):
            plot = AnalysisPlot(1, 2)
            gamma_plot = SubPlot([0, 0], 'gamma', self.analysis_handler.gamma_dist)
            lower_plot = SubPlot([0, 1], 'lower', self.analysis_handler.lower_dist)
            plot.add_subplot(gamma_plot)
            plot.add_subplot(lower_plot)
            plot.show(title='Gamma and Lower Distributions',
                      save=save,
                      saveName=f'Distributions_{self.analysis_handler.prefix}.png')

        # Show AC/AD distributions
        if hasattr(self.analysis_handler, 'ac_dist') and hasattr(self.analysis_handler, 'ad_dist'):
            plot = AnalysisPlot(1, 2)
            ac_plot = SubPlot([0, 0], 'ac', self.analysis_handler.ac_dist)
            ad_plot = SubPlot([0, 1], 'ad', self.analysis_handler.ad_dist)
            plot.add_subplot(ac_plot)
            plot.add_subplot(ad_plot)
            plot.show(title='AC and AD Distributions',
                      save=save,
                      saveName=f'ACAD_{self.analysis_handler.prefix}.png')

        # Show average trajectories
        if hasattr(self.analysis_handler, 'average_trajectories') and hasattr(self.analysis_handler, 'u_space'):
            plot_3d = AnalysisPlot(1, 1, figsize=(10, 8))
            avg_traj = SubPlot([0, 0], 'average_trajectory3d',
                               self.analysis_handler.average_trajectories,
                               self.analysis_handler.u_space)
            plot_3d.add_subplot(avg_traj)
            suffix = f'Ntraj{self.analysis_handler.Ntraj}_Tper{self.analysis_handler.Tper}'
            plot_3d.show(title=f'Average Trajectory 3D - {suffix}',
                         save=save,
                         saveName=f'AvgTraj3D_{self.analysis_handler.prefix}_{suffix}.png')

            plot_td = AnalysisPlot(1, 1, figsize=(10, 8))
            topdown = SubPlot([0, 0], 'average_trajectorytopdown',
                              self.analysis_handler.average_trajectories,
                              self.analysis_handler.u_space)
            plot_td.add_subplot(topdown)
            plot_td.show(title=f'Top-Down View - {suffix}',
                         save=save,
                         saveName=f'TopDown_{self.analysis_handler.prefix}_{suffix}.png')

    def run_comparison(self):
        """Run a comparison workflow between current model and another model."""
        print("\n=== Comparison Workflow ===")
        print("Setting up comparison model...")

        # Import needed modules
        import maxcal_model_handler as mc
        from analysis_handler import AnalysisHandler
        from plot_handler import AnalysisPlot, SubPlot

        # Get comparison model settings
        comparison_settings = {}
        choice = self.config_manager.get_user_config_choice(
            "Enter the index of the comparison configuration (or 'm' for manual): ")

        if choice == 'm':
            comparison_settings = {
                "I": input("Enter I value for comparison model: "),
                "NR": int(input("Enter NR value for comparison model: ")),
                "NE": int(input("Enter NE value for comparison model: ")),
                "dt": float(input("Enter dt value for comparison model: ")),
                "joch_data": input("Use Joch data for comparison model? (true/false): ").strip().lower() == "true",
            }
        else:
            config_list = self.config_manager.list_configurations()
            config = self.config_manager.load_configuration(config_list[choice])
            comparison_settings = config.get('settings', {})

        # Create and initialize comparison model
        comparison_model = mc.BrainModel(
            comparison_settings["I"],
            comparison_settings["NR"],
            comparison_settings["NE"],
            comparison_settings["dt"],
            joch_data=comparison_settings.get("joch_data", False)
        )

        # Load or generate comparison trajectory
        if input("Load comparison trajectory? (yes/no): ").strip().lower() == 'yes':
            comparison_model.load_trajectory()
        else:
            comparison_model.find_model() or comparison_model.load_params()
            initial_state = tuple(
                map(float, input("Enter initial state (comma-separated, e.g., 0,0,0,0): ").split(',')))
            duration = float(input("Enter trajectory duration: "))
            comparison_model.generate_trajectory(initial_state, duration)
            if input("Save comparison trajectory? (yes/no): ").strip().lower() == 'yes':
                comparison_model.save_trajectory()

        # Create analysis handler for comparison model
        comparison_analysis = AnalysisHandler(comparison_model)

        # Generate needed analysis results
        print("Generating analysis results for comparison...")
        comparison_analysis.load_or_generate_dominance_statistics()
        comparison_analysis.load_or_generate_trajectory_density()
        comparison_analysis.load_or_generate_avg_trajectory(Tper=0.03, Ntraj=10000)
        comparison_analysis.load_or_generate_distributions()

        # Make sure current model has all analysis results as well
        if self.analysis_handler is None:
            self.initialize_analysis()
        self.analysis_handler.load_or_generate_dominance_statistics()
        self.analysis_handler.load_or_generate_trajectory_density()
        self.analysis_handler.load_or_generate_avg_trajectory(Tper=0.03, Ntraj=10000)
        self.analysis_handler.load_or_generate_distributions()

        # Create and show comparison plots
        save_plots = input("Save comparison plots? (yes/no): ").strip().lower() == 'yes'

        # Compare dominance statistics
        print("\nComparing dominance statistics...")
        p1 = SubPlot([0, 0], 'dominance', self.analysis_handler.dominance_statistics)
        p2 = SubPlot([0, 1], 'dominance', comparison_analysis.dominance_statistics)
        p1.display_dominance_statistics()
        p2.display_dominance_statistics()

        # Compare 3D plots
        print("Comparing 3D plots...")
        plot_3d = AnalysisPlot(1, 2, figsize=(14, 8))
        avg_traj1 = SubPlot([0, 0], 'average_trajectory3d',
                            self.analysis_handler.average_trajectories,
                            self.analysis_handler.u_space)
        avg_traj2 = SubPlot([0, 1], 'average_trajectory3d',
                            comparison_analysis.average_trajectories,
                            comparison_analysis.u_space)
        plot_3d.add_subplot(avg_traj1)
        plot_3d.add_subplot(avg_traj2)
        suffix = f'Ntraj{self.analysis_handler.Ntraj}_Tper{self.analysis_handler.Tper}'
        model1_name = "Model1" if not self.settings.get("joch_data") else "Joch"
        model2_name = "Model2" if not comparison_settings.get("joch_data") else "Joch"
        plot_3d.show(
            title=f'{model1_name}(Left) vs. {model2_name}(Right) N={self.analysis_handler.Ntraj}, T={self.analysis_handler.Tper}',
            save=save_plots,
            saveName=f'Compare3D_{model1_name}vs{model2_name}_{suffix}.png'
        )

        # Compare top-down flows
        print("Comparing top-down flows...")
        plot_td = AnalysisPlot(1, 2)
        topdown1 = SubPlot([0, 0], 'average_trajectorytopdown',
                           self.analysis_handler.average_trajectories,
                           self.analysis_handler.u_space)
        topdown2 = SubPlot([0, 1], 'average_trajectorytopdown',
                           comparison_analysis.average_trajectories,
                           comparison_analysis.u_space)
        plot_td.add_subplot(topdown1)
        plot_td.add_subplot(topdown2)
        plot_td.show(
            title=f'{model1_name}(Left) vs. {model2_name}(Right) N={self.analysis_handler.Ntraj}, T={self.analysis_handler.Tper}',
            save=save_plots,
            saveName=f'CompareTopDown_{model1_name}vs{model2_name}_{suffix}.png'
        )

        # Compare AC/AD distributions
        print("Comparing AC/AD distributions...")
        plot_acad = AnalysisPlot(2, 2)
        ac1 = SubPlot([0, 0], 'ac', self.analysis_handler.ac_dist)
        ac2 = SubPlot([1, 0], 'ac', comparison_analysis.ac_dist)
        ad1 = SubPlot([0, 1], 'ad', self.analysis_handler.ad_dist)
        ad2 = SubPlot([1, 1], 'ad', comparison_analysis.ad_dist)
        plot_acad.add_subplot(ac1)
        plot_acad.add_subplot(ac2)
        plot_acad.add_subplot(ad1)
        plot_acad.add_subplot(ad2)
        plot_acad.show(
            title=f'{model1_name}(Top) vs. {model2_name}(Bottom) AC/AD Distributions',
            save=save_plots,
            saveName=f'CompareACAD_{model1_name}vs{model2_name}.png'
        )

        # Compare gamma and lower distributions
        print("Comparing gamma and lower distributions...")
        plot_gl = AnalysisPlot(2, 2)
        gamma1 = SubPlot([0, 0], 'gamma', self.analysis_handler.gamma_dist)
        gamma2 = SubPlot([1, 0], 'gamma', comparison_analysis.gamma_dist)
        lower1 = SubPlot([0, 1], 'lower', self.analysis_handler.lower_dist)
        lower2 = SubPlot([1, 1], 'lower', comparison_analysis.lower_dist)
        plot_gl.add_subplot(gamma1)
        plot_gl.add_subplot(gamma2)
        plot_gl.add_subplot(lower1)
        plot_gl.add_subplot(lower2)
        plot_gl.show(
            title=f'{model1_name}(Top) vs. {model2_name}(Bottom) Gamma & Lower Distributions',
            save=save_plots,
            saveName=f'CompareGammaLower_{model1_name}vs{model2_name}.png'
        )

        print("Comparison completed.")