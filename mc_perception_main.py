import os
import pandas as pd
import ast
from mc_perception_inferred_model_class import brain_model

MAXTOP = 11
MAXBOT = 26
M_step = 1


def get_params(Itest,VC,HC):
    dir_path = 'Infered_parameters'
    file_name = f'{Itest}_inferred_params'
    file_name = f'{Itest}_VC_{VC}_HC_{HC}_inferred_paramsV2'

    file_path = os.path.join(dir_path, file_name+'.csv')
    df = pd.read_csv(file_path)

    df['Values'] = df.iloc[:, 1].apply(ast.literal_eval)
    params = df['Values'].values[0]
    return params
def display_dominance_stats(model_list):
    # Display dominance statistics for each model
    # First display a top row for the column names
    header = ['I',  'Duration',    'Variance',    'CV',   'Skewness',   'CC1',   'CC2',   'CC3',   'RevCount',    'Pneither (%)']
    print(f"{'   |   '.join(header)}")
    print('-' * 120)
    for model in model_list:
        row1 = [
            model.I_test,
            model.dominance_statistics["Duration1"],
            model.dominance_statistics["Variance1"],
            model.dominance_statistics["CV1"],
            model.dominance_statistics["Skewness1"],
            model.dominance_statistics["Sequential1"],
            model.dominance_statistics["Sequential2"],
            model.dominance_statistics["Sequential3"],
            model.dominance_statistics["RevCount"],
            #model.dominance_statistics["Pdouble"],
            model.dominance_statistics["Pneither"]
        ]
        row2 = [
            model.I_test,
            model.dominance_statistics["Duration2"],
            model.dominance_statistics["Variance2"],
            model.dominance_statistics["CV2"],
            model.dominance_statistics["Skewness2"],
            model.dominance_statistics["Sequential1"],
            model.dominance_statistics["Sequential2"],
            model.dominance_statistics["Sequential3"],
            model.dominance_statistics["RevCount"],
            #model.dominance_statistics["Pdouble"],
            model.dominance_statistics["Pneither"]
        ]
        print(
            f"{'  |  '.join(map(lambda x: f'{x.values[0]:.4f}' if isinstance(x, pd.Series) and isinstance(x.values[0], float) else str(x.values[0]) if isinstance(x, pd.Series) else str(x), row1))}")
        print(
            f"{'  |  '.join(map(lambda x: f'{x.values[0]:.4f}' if isinstance(x, pd.Series) and isinstance(x.values[0], float) else str(x.values[0]) if isinstance(x, pd.Series) else str(x), row2))}")




if __name__ == "__main__":
    Itest = ['000','025','050','075','100']
    tmax = [40_000_000, 30_000_000, 25_000_000, 20_000_000,15_000_000]
    model_list = []

    I_control = '075'
    HC,VC = 100,100
    Tend = 8000
    dt = 0.0001
    initial_guess= (
        -10.813039700214725, -10.493275942310733, -8.467808763513434, -3.440305398445934, 0.8102093637779607,
        1.8747257423529846, 2.686201028042257, 0.11667559074547738, 0.7218451091627106 )
    # params_control = get_params(I_control,VC,HC)
    I_control_model = brain_model(None, I_control, MAXTOP, MAXBOT, dt=dt, Tend=Tend,HC=HC,VC=VC)
    I_control_model.choose_params()
    model_list.append(I_control_model)
    #I_control_model.find_model(initial_guess=initial_guess)
    #print(I_control_model.params)

    try:
        I_control_model.load_trajectory(short=False)
        print('loaded trajectory\n')
    except:
        print('No trajectory found, generating now')
        I_control_model.generate_trajectory((0, 0, 0, 0), Tend)
        I_control_model.save_trajectory()
    I_control_model.dominance_statistics_trajectories()
    display_dominance_stats([I_control_model])
    #need to fix displayer
    #I_control_model.get_trajectory_density()
    #I_control_model.plot_trajectory_density()
    #
    joch_model_control = brain_model(None, I_control, MAXTOP, MAXBOT, dt=dt, Tend=Tend,HC=HC,VC=VC,joch_data=True)
    joch_model_control.load_trajectory()

    joch_model_control.dominance_statistics_trajectories()
    display_dominance_stats([joch_model_control])
    #joch_model_control.get_avg_trajectory_flow(.03,10000)
    #joch_model_control.plot_avg_trajectory()

    #I_control_model.get_avg_trajectory_flow(.03,10000)
    #I_control_model.plot_avg_trajectory()



    #display_dominance_stats(model_list)




