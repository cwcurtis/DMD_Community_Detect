import numpy as np
from kuramoto import kuramoto
import time
import pickle


def kuramoto_run(tf, dt, kval, nds, omgs, sgm, frq, tag):

    nt = np.int(tf / dt) + 1

    fstr = str(frq)
    fstr = fstr.replace('.', 'pt')

    pickle_off = open("graph_info_" + tag + ".pickle", "rb")
    graph_dict = pickle.load(pickle_off)

    amat = graph_dict['adjacency_matrix']

    xt, dxt = kuramoto(amat, frq*omgs, nds, sgm, kval, tf, dt)

    soln_dict = {'time_steps': nt,
                 'dist_spread': frq,
                 'soln': xt,
                 'deriv': dxt}

    pickling_on = open("solution_info_" + tag + "_" + fstr + ".pickle", "wb")
    pickle.dump(soln_dict, pickling_on)
    pickling_on.close()
