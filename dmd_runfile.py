import numpy as np
import pickle
from numba_kernel_dmd import dmd_cmp, krn_dmd_cmp, mode_err_cmp, recon_err_cmp


def dmd_run(tf, dt, nds, frq, tag):
    fstr = str(frq)
    fstr = fstr.replace('.', 'pt')

    pickle_off = open("solution_info_" + tag + "_" + fstr + ".pickle", "rb")
    soln_dict = pickle.load(pickle_off)

    xt = soln_dict['soln']
    dxt = soln_dict['deriv']

    tstind = np.int(np.floor((tf-40.)/dt))
    xtkp = xt[:, tstind:]
    dxtkp = dxt[:, tstind:]
    taxis = dt*tstind + dt*np.arange(np.shape(xtkp)[1])
    xtot = np.zeros((2*nds, np.shape(xtkp)[1]), dtype=np.float64)
    xtot[:nds, :] = np.cos(xtkp)
    xtot[nds:, :] = np.sin(xtkp)

    phs_ord_param = np.mean(np.exp(1j*xtkp), 0)
    rfac1 = np.abs(phs_ord_param)
    psifac1 = np.angle(phs_ord_param)

    skp = 1

    xred = xtot[:, ::skp]
    dmd_thrshhld = 4
    # nterms = 20
    dmd_evls, dmd_phim, dmd_kmodes = dmd_cmp(xred, dmd_thrshhld)
    # dmd_evls, dmd_phim, dmd_kmodes = krn_dmd_cmp(xred, nterms, dmd_thrshhld)
    dmd_err = mode_err_cmp(dmd_evls, dmd_phim)
    recon_err_cmp(xred, dmd_evls, dmd_kmodes, dmd_phim)

    data_dict = {'skip': skp,
             'time_axis': taxis,
             'velocities': dxtkp,
             'radial_parameter1': rfac1,
             'angle_parameter1': psifac1,
             'dmd_evls': dmd_evls,
             'dmd_phim': dmd_phim,
             'dmd_kmodes': dmd_kmodes,
             'dmd_error': dmd_err
             }

    pickling_on = open("Data_Collection_" + tag + "_" + fstr + ".pickle", "wb")
    pickle.dump(data_dict, pickling_on)
    pickling_on.close()


def dmd_window_run(ti, tf, dt, nds, cxt, sxt):

    indf = np.int(np.floor(tf/dt))
    indi = np.int(np.floor(ti/dt))
    indt = indf - indi
    xtot = np.zeros((2*nds, indt), dtype=np.float64)
    xtot[:nds, :] = cxt[:, indi:indf]
    xtot[nds:, :] = sxt[:, indi:indf]

    dmd_thrshhld = 3
    dmd_evls, dmd_phim, dmd_kmodes = dmd_cmp(xtot, dmd_thrshhld)
    dmd_err = mode_err_cmp(dmd_evls, dmd_phim)

    return dmd_evls, dmd_phim, dmd_kmodes, dmd_err
