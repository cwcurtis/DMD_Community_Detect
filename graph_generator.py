import numpy as np
from graph_maker import graph_build_sf, graph_build_sw, graph_build_er_block, max_eval_cmp
import pickle


def graph_run(nds, tag):

    if tag == 'er':
        # Block Erdos-Renyi
        p, q, r = .75, .01, 5
        params = [p, q, r]
        amat = graph_build_er_block(p, q, r, nds)
    elif tag == 'sf':
        # Scale Free, or Preferential Attachment
        m0, m = 10, 4
        params = [m0, m]
        amat = graph_build_sf(m0, m, nds)
    elif tag == 'sw':
        # Small world
        kavg, beta = 20, .6
        params = [kavg, beta]
        amat = graph_build_sw(nds, kavg, beta)

    lam_max = max_eval_cmp(amat)
    print(lam_max)
    print("Critical Gamma is : %1.2f" % ((np.sqrt(np.pi)*nds*lam_max)/2.))

    graph_dict = {'adjacency_matrix': amat,
              'max_eval': lam_max,
              'tag': tag,
              'parameters': params
              }

    pickling_on = open("graph_info_"+tag+".pickle", "wb")
    pickle.dump(graph_dict, pickling_on)
    pickling_on.close()
