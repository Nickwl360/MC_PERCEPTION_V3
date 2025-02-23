import os
import configuration_manager as cm
import maxcal_model_handler as mc
from analysis_handler import AnalysisHandler
from plot_handler import PlotHandler
import pandas as pd


############WORK ON PREDEFINE CONFIGS/manual LATER######################

def load_settings():
    config_list = os.listdir('predefined_configurations')
    for index, config in enumerate(config_list):
        print(f"{index}: {config}")

    choice = input("Enter the index of the configuration(s) you would like to use (or m for manual): ")
    if choice.lower() == 'm':
        return {
            "NR": int(input("Enter NR (size + 1) for top layer: ")),
            "NE": int(input("Enter NE (size + 1) for bottom layer: ")),
            "dt": float(input("Enter dt: ")),
            "joch_data": input("Use Joch data? (true/false): ").strip().lower() == "true",
            }

    else:
        file_path = os.path.join('predefined_configurations', config_list[int(choice)])
        config = cm(file_path)
        return config.get('settings', {})
def load_or_prompt_workflow():
    config_list = os.listdir('predefined_configurations')
    for index, config in enumerate(config_list):
        print(f"{index}: {config}")

    choice = input("Enter the index of the configuration(s) you would like to use (or m for manual): ")
    if choice.lower() == 'm':
        return {
            "parameters": input("Find parameters? (infer/load/none): ").strip().lower(),
            "trajectory": input("Generate or load trajectory? (generate/load/none): ").strip().lower(),
            "densities": input("Generate or load density? (generate/load/none): ").strip().lower(),
            "average_distributions": input(
                "Generate or load average distributions? (generate/load/none): ").strip().lower(),
            "average_trajectory": input(
                "Generate or load average trajectory flows? (generate/load/none): ").strip().lower(),
            "generate_plots": input("Generate plots? (yes/no): ").strip().lower() == "yes"
        }

    else:
        file_path = os.path.join('predefined_configurations', config_list[int(choice)])
        config = cm(file_path)
        return config.get('workflow',{})
def run_workflow(settings, workflow_options):
    model =  mc.BrainModel( settings["I"], settings["NR"], settings["NE"], settings["dt"])
    if workflow_options.get('parameters') == 'infer':
        initial_guess = workflow_options.get('parameters').get('initial_guess', None)

        print("Inferring parameters")

    if workflow_options.get('trajectory') == 'generate':
        print("Generating trajectory")
    elif workflow_options.get('trajectory') == 'load':
        print("Loading trajectory")

    if workflow_options.get('densities') == 'generate':
        print("Generating densities")
    elif workflow_options.get('densities') == 'load':
        print("Loading densities")

    if workflow_options.get('average_distributions') == 'generate':
        print("Generating average distributions")
    elif workflow_options.get('average_distributions') == 'load':

        print("Loading average distributions")
    if workflow_options.get('average_trajectory') == 'generate':
        print("Generating average trajectory flows")
    elif workflow_options.get('average_trajectory') == 'load':
        print("Loading average trajectory flows")

    if workflow_options.get('generate_plots'):
        print("Generating plots")
    else:
        print("No plots generated")



def analyze_all(traj,model):

    analysis = AnalysisHandler(traj, model)
    analysis.load_or_generate_dominance_statistics()
    analysis.load_or_generate_trajectory_density()
    analysis.load_or_generate_avg_trajectory(Tper=.03, Ntraj=10000)
    analysis.load_or_generate_distributions()

    plotter = PlotHandler(analysis)
    plotter.plot_avg_trajectories()
    plotter.plot_distributions()


def comparitive_workflow():
    control_model_mc = mc.BrainModel('075', 11, 26, 0.0001)
    control_model_mc.load_params()
    control_model_mc.load_trajectory()
    traj = control_model_mc.long_trajectory
    analyze_all(traj,control_model_mc)

    print('FINISHED CONTROL MODEL')

    joch_control  = mc.BrainModel('075', 11, 26, 0.0001, joch_data=True)
    joch_control.load_trajectory()
    traj = joch_control.long_trajectory
    analyze_all(traj,joch_control)

    print('FINISHED JOCH CONTROL')

    control_model_mcdt2 = mc.BrainModel('075', 11, 26, 0.0002)
    control_model_mcdt2.load_params()
    control_model_mcdt2.generate_trajectory((0,0,0,0), 8000) #will  give 40mil
    control_model_mcdt2.save_trajectory()
    traj = control_model_mcdt2.long_trajectory
    analyze_all(traj,control_model_mcdt2)

    print('FINISHED CONTROL MODEL DT2')

    joch_control_dt2 = mc.BrainModel('075', 11, 26, 0.0002, joch_data=True)
    joch_control_dt2.ndt =2
    joch_control_dt2.load_trajectory()
    traj = joch_control_dt2.long_trajectory
    analyze_all(traj,joch_control_dt2)

    print('FINISHED JOCH CONTROL DT2')



if __name__ == '__main__':
    #testing
    #model = mc.BrainModel('075', 11, 26, 0.0002)
    #model.find_model()
   #  parameters v1=  [-11.21441075  ,-9.17350676, -12.56330495 ,  0.41337435 , 19.89311771, #wack ass kcoop
   # 7.8307939  ,-18.78255287  ,-0.07484337 , -8.09972307]
    #model.find_model()
    #model.load_params()
    #print(model.params)

    #if good:
    comparitive_workflow()



