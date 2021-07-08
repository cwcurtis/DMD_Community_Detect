import numpy as np
import graph_generator as gg
import kuramoto_runfile as kr
import dmd_runfile as dmd
import pickle
import multiprocessing as mp
from time import time


def inner_loop(pair, nds, tf, dt, kval, omgs, sgm):
    tag = pair[0]
    frq = pair[1]
    kr.kuramoto_run(tf, dt, kval, nds, omgs, sgm, frq, tag)
    dmd.dmd_run(tf, dt, nds, frq, tag)


if __name__ == '__main__':

    nds = 400
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

    full_schedule = [('sf', .1), ('sf', 1.), ('sf', 10.), ('sw', .1), ('sw', 1.), ('sw', 10.)]
    gg.graph_run(nds, 'sf')
    gg.graph_run(nds, 'sw')

    pool = mp.Pool(mp.cpu_count())
    start = time()
    pool.starmap(inner_loop, [(pair, nds, tf, dt, kval, omgs, sgm) for pair in full_schedule])
    end = time()
    pool.close()
    print("Elapsed time is: %f" % (end-start))