import numpy as np


def kuramoto(amat, omgs, n, sgm, kval, tf, dt):
    nstep = np.int(tf / dt)
    x0 = np.zeros((n, 1), dtype=np.float64)
    xt = np.zeros((n, nstep + 1), dtype=np.float64)
    dxt = np.zeros((n, 2), dtype=np.float64)

    x0[:, 0] = 2. * np.pi * np.arange(n) / n
    xt[:, 0] = np.squeeze(x0)
    dxt[:, 0] = np.squeeze(kuramoto_nl(omgs, amat, kval, x0))
    for jj in range(nstep):
        k1 = dt*kuramoto_nl(omgs, amat, kval, x0)
        k2 = dt*kuramoto_nl(omgs, amat, kval, x0 + k1)
        x0 += .5*(k1 + k2)
        xt[:, jj + 1] = np.squeeze(x0)
    dxt[:, 1] = np.squeeze(kuramoto_nl(omgs, amat, kval, x0))
    return xt, dxt


def kuramoto_nl(omgs, amat, kval, lhs):
    nds = lhs.shape[0]
    rhs = np.zeros((nds, 1), dtype=np.float64)
    #lmat = np.tile(lhs, (1, nds))
    difs = np.sin(np.tile(lhs, (1, nds)) - np.tile(lhs.T, (nds, 1)))
    #difs = np.sin(lmat - lmat.T)
    rhs[:, 0] = omgs + np.squeeze(kval*np.mean(amat*difs, 0))
    return rhs
