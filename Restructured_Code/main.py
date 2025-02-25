import os
import configuration_manager as cm
import maxcal_model_handler as mc
from analysis_handler import AnalysisHandler
from plot_handler import PlotHandler
import pandas as pd
from plot_handler import AnalysisPlot,SubPlot


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


    control_model_mcdt2 = mc.BrainModel('075', 11, 26, 0.0002)
    #control_model_mcdt2.find_model()
    control_model_mcdt2.load_params()
    control_model_mcdt2.generate_trajectory((0,0,0,0), 20) #will  give 40mil
    p = SubPlot([0,0],'basic',control_model_mcdt2.long_trajectory)

    #if data looks good generate super long and do below
    #control_model_mcdt2.save_trajectory()

    #
    # analysis = AnalysisHandler(control_model_mcdt2)
    # analysis.load_or_generate_dominance_statistics()
    # analysis.load_or_generate_trajectory_density()
    # analysis.load_or_generate_avg_trajectory(Tper=.03, Ntraj=10000)
    # analysis.load_or_generate_distributions()


    print('FINISHED CONTROL MODEL DT2')






def compare_gammaandlower(MCanalysis,Jochanalysis,save=False):
    d1Analyze = MCanalysis
    jd1Analyze = Jochanalysis
    Plot = AnalysisPlot(2,2)
    gammaplotMC = SubPlot([0,0],'gamma',d1Analyze.gamma_dist)
    gammaplotJoch = SubPlot([1,0],'gamma',jd1Analyze.gamma_dist)
    lowerDistMC = SubPlot([0,1],'lower',d1Analyze.lower_dist)
    lowerDistJoch = SubPlot([1,1],'lower',jd1Analyze.lower_dist)
    Plot.add_subplot(gammaplotMC)
    Plot.add_subplot(gammaplotJoch)
    Plot.add_subplot(lowerDistMC)
    Plot.add_subplot(lowerDistJoch)
    Plot.show(title='MaxCal(Top) vs. Jochen Distributions(Bottom)',save=save,saveName=f'MCvsJochGamma&Lower{d1Analyze.prefix}.png')
def compareACADDistributions(MCanalysis,Jochanalysis,save=False):
    d1Analyze = MCanalysis
    jd1Analyze = Jochanalysis
    Plot = AnalysisPlot(2,2)
    acMC = SubPlot([0, 0], 'ac', d1Analyze.ac_dist)
    acJoch = SubPlot([1, 0], 'ac', jd1Analyze.ac_dist)
    adMC = SubPlot([0, 1], 'ad', d1Analyze.ad_dist)
    adJoch = SubPlot([1, 1], 'ad', jd1Analyze.ad_dist)
    Plot.add_subplot(acMC)
    Plot.add_subplot(acJoch)
    Plot.add_subplot(adMC)
    Plot.add_subplot(adJoch)
    Plot.show(title='MaxCal(Top) vs. Jochen Distributions(Bottom)',save=save,saveName=f'MCvsJochACADDistributions{d1Analyze.prefix}.png')
def compareTopDownFlows(MCanalysis,Jochanalysis,save=False):
    d1Analyze = MCanalysis
    jd1Analyze = Jochanalysis
    d1Analyze.load_or_generate_avg_trajectory(Tper=.03, Ntraj=10000)
    jd1Analyze.load_or_generate_avg_trajectory(Tper=.03, Ntraj=10000)

    PlotTopDown = AnalysisPlot(1,2)
    topdownMC = SubPlot([0,0],'average_trajectorytopdown',d1Analyze.average_trajectories,d1Analyze.u_space)
    topdownJoch = SubPlot([0,1],'average_trajectorytopdown',jd1Analyze.average_trajectories,jd1Analyze.u_space)
    PlotTopDown.add_subplot(topdownMC)
    PlotTopDown.add_subplot(topdownJoch)
    suffix = f'Ntraj{d1Analyze.Ntraj}_Tper{d1Analyze.Tper}'
    PlotTopDown.show(title=f'MaxCal(Left) vs. Jochen Flows(Right)  N={d1Analyze.Ntraj}, T={d1Analyze.Tper}',save=save,saveName=f'MCvsJochTopDownFlow{d1Analyze.prefix}{suffix}.png')
def compare3dPlots(MCanalysis,Jochanalysis,save=False):
    d1Analyze = MCanalysis
    jd1Analyze = Jochanalysis
    d1Analyze.load_or_generate_avg_trajectory(Tper=.03, Ntraj=10000)
    jd1Analyze.load_or_generate_avg_trajectory(Tper=.03, Ntraj=10000)

    Plot3d = AnalysisPlot(1,2,figsize=(12,8))
    avgtrajMC = SubPlot([0,0],'average_trajectory3d',d1Analyze.average_trajectories,d1Analyze.u_space)
    avgtrajJoch = SubPlot([0,1],'average_trajectory3d',jd1Analyze.average_trajectories,jd1Analyze.u_space)
    Plot3d.add_subplot(avgtrajMC)
    Plot3d.add_subplot(avgtrajJoch)
    suffix = f'Ntraj{d1Analyze.Ntraj}_Tper{d1Analyze.Tper}'

    Plot3d.show(title=f'MaxCal(Left) vs. Jochen Flows(Right)  N={d1Analyze.Ntraj}, T={d1Analyze.Tper}',save=save,saveName=f'MCvsJoch3d{d1Analyze.prefix}{suffix}.png')
def compareDominanceStatistics(MCanalysis,Jochanalysis):
    d1Analyze = MCanalysis
    jd1Analyze = Jochanalysis
    d1Analyze.load_or_generate_dominance_statistics()
    jd1Analyze.load_or_generate_dominance_statistics()
    p1 = SubPlot([0,0],'dominance',d1Analyze.dominance_statistics)
    p2 = SubPlot([0,1],'dominance',jd1Analyze.dominance_statistics)
    p1.display_dominance_statistics()
    p2.display_dominance_statistics()


def plotting_tester(save=False):
    d1 = mc.BrainModel('075', 11, 26, 0.0001)
    jd1 = mc.BrainModel('075', 11, 26, 0.0001, joch_data=True)
    d1.load_trajectory()
    jd1.load_trajectory()
    d1Analyze = AnalysisHandler(d1)
    jd1Analyze = AnalysisHandler(jd1)
    d1Analyze.load_or_generate_distributions()
    jd1Analyze.load_or_generate_distributions()

    compareDominanceStatistics(d1Analyze, jd1Analyze)
    #compare3dPlots(d1Analyze, jd1Analyze, save=save)
    #compareTopDownFlows(d1Analyze, jd1Analyze, save=save)
    #compareACADDistributions(d1Analyze, jd1Analyze, save=save)
    #compare_gammaandlower(d1Analyze, jd1Analyze, save=save)

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
    #comparitive_workflow()
    plotting_tester(save=False)


