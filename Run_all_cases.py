import numpy as np
import graph_generator as gg
import kuramoto_runfile as kr
import dmd_runfile as dmd
import pickle

nds = 800
omgs = np.random.randn(nds)
tf = 800.
dt = 5e-2
kval = 10.
sgm = 0.

univ_dict = {'number_of_nodes': nds,
             'kval': kval,
             'frequencies': omgs,
             'tf': tf,
             'time_step': dt
             }

pickling_on = open("universal_info.pickle", "wb")
pickle.dump(univ_dict, pickling_on)
pickling_on.close()

tag_schedule = ['sf', 'sw']
frq_schedule = [.1, 1., 10.]

for tag in tag_schedule:
    gg.graph_run(nds, tag)
    for frq in frq_schedule:
        kr.kuramoto_run(tf, dt, kval, nds, omgs, sgm, frq, tag)
        dmd.dmd_run(tf, dt, nds, frq, tag)

