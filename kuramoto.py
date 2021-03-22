import numpy as np


def kuramoto(amat, omgs, n, sgm, kval, tf, dt):
    nstep = np.int(tf / dt)
    Pi = np.pi
    x0 = 2.*Pi*np.arange(n)/n
    # x0 = np.zeros(n, dtype=np.float64)
    xt = np.zeros((n, nstep + 1), dtype=np.float64)
    dxt = np.zeros((n, nstep + 1), dtype=np.float64)
    xt[:, 0] = x0
    lhs = x0
    for jj in range(nstep):
        dxt[:, jj] = kuramoto_nl(omgs, amat, kval, lhs)
        k1 = dt*dxt[:, jj]
        k2 = dt*kuramoto_nl(omgs, amat, kval, lhs + k1)
        lhs += .5*(k1 + k2)
        xt[:, jj + 1] = lhs
    dxt[:, nstep] = kuramoto_nl(omgs, amat, kval, lhs)
    return xt, dxt


def kuramoto_nl(omgs, amat, kval, lhs):
    nds = lhs.size
    lvec = np.zeros((nds, 1), dtype=np.float64)
    lvec[:, 0] = lhs
    difs = np.sin(np.tile(lvec, (1, nds)) - np.tile(lvec.T, (nds, 1)))
    rhs = omgs + kval*np.mean(amat*difs, 0)
    return rhs


def kuramoto_nl_alt(omgs, amat, kval, lhs):
    nds = lhs.size
    dmat = np.zeros((nds, nds), dtype=np.float64)
    for jj in range(0, nds-1):
        diffvec = np.sin(lhs[(jj+1):] - lhs[:nds-1-jj])
        dmat += np.diag(diffvec, jj+1)

    dmat += -dmat.T
    rhs = omgs + kval*np.mean(amat*dmat, 0)
    return rhs