from utils import data_handler as dh
import numpy as np

dh = dh.DataHandler()
df = dh.load_mat('trajectories', 'I_075_VC_100_HC_100_dt_0.0001_trajectory')

As, Bs, Cs, Ds = df['A'].astype('int8'), df['B'].astype('int8'), df['C'].astype('int8'), df['D'].astype('int8')
#As, Bs, Cs, Ds = df['A'][0].astype('int8'), df['B'][0].astype('int8'), df['C'][0].astype('int8'), df['D'][0].astype('int8')

print(np.shape(As))
#As, Bs, Cs, Ds = As.astype('int8'), Bs.astype('int8'), Cs.astype('int8'), Ds.astype('int8')
dict_data = {
    "A": As,
    "B": Bs,
    "C": Cs,
    "D": Ds
}
print(np.dtype(As[0]))
#dh.save_mat("trajectories", "I_075_VC_100_HC_100_dt_0.0001_trajectory", dict_data, do_compression=True)