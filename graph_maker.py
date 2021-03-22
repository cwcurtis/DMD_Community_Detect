import numpy as np


def max_eval_cmp(A):
    tol = 1e-3
    n, n = np.shape(A)
    nmat = np.zeros((n, n), dtype=np.int64)
    nmat[:, :] = A@A
    cnt = 1
    cest = 1.
    nest = np.log(np.trace(nmat))/(cnt+1)
    err = np.abs(nest - cest)
    while err > tol and cnt <= 100:
        cest = nest
        cnt += 1
        nmat = A @ nmat
        nest = np.log(np.trace(nmat))/(cnt + 1)
        err = np.abs(nest - cest)
    print("Final eigenvalue error: %1.2e" % err)
    return np.exp(nest)


def graph_build_sf(n0, m, n):
    amat = np.zeros((n, n), dtype=np.int)
    # initialize with ER graph among n0 nodes
    per = .75
    for ii in range(1, n0):
        for jj in range(ii):
            pval = np.random.uniform(0., 1., 1)
            if pval < per:
                amat[ii, jj] = 1
                amat[jj, ii] = 1

    amat = amat - np.diag(np.diag(amat))
    probs = np.zeros(n, dtype=np.float64)
    for jj in range(n0, n):
        degs = np.sum(amat, 1)
        totdeg = np.sum(degs)
        probs[:jj] = degs[:jj]/totdeg
        dist = np.ones(jj, dtype=np.float64)
        for ll in range(1, jj):
            dist[ll-1] = np.sum(probs[:ll])
        inds = np.arange(jj)
        for ll in range(m):
            pval = np.random.uniform(0., 1., 1)
            indschse = dist > pval
            indnxt = np.min(inds[indschse])
            amat[jj, indnxt] = 1
            amat[indnxt, jj] = 1

    # remove any self connections
    amat = amat - np.diag(np.diag(amat))
    return amat


def graph_build_sw(n, kavg, beta):
    kah = np.int(kavg/2)
    amat = np.zeros((n, n), dtype=np.int)
    shft = n - 1 - kah
    inds = np.arange(n)
    # build balanced circular network
    for ii in range(n):
        for jj in range(n):
            dif = np.abs(ii-jj)
            if np.mod(dif, shft) <= kah and dif > 0:
                amat[ii, jj] = 1
                amat[jj, ii] = 1

    # rewire
    for ii in range(n):
        for jj in range(ii + 1, ii + kah + 1):
            jjs = np.mod(jj, n)
            pdf = np.zeros(n, dtype=np.float64)
            pdf[:ii] = 1./(n-1.)
            pdf[ii+1:] = 1./(n-1.)
            pdist = np.ones(n, dtype=np.float64)
            for ll in range(1, n-1):
                pdist[ll-1] = np.sum(pdf[:ll])
            pval = np.random.uniform(0., 1., 1)
            if pval < beta:
                chsvl = np.random.uniform(0., 1., 1)
                indskp = pdist > chsvl
                indrw = np.min(inds[indskp])
                if amat[ii, indrw] == 0:
                    amat[ii, jjs] = 0
                    amat[jjs, ii] = 0

                    amat[ii, indrw] = 1
                    amat[indrw, ii] = 1

    return amat


def pmat_build(p, q, r):
    pmat = q * np.ones((r, r))
    pmat -= q * np.eye(r)
    pmat += p * np.eye(r)
    return pmat


def graph_build_er_block(p, q, r, n):
    amat = np.zeros((n, n), dtype=np.int)
    shuffle = np.random.permutation(np.arange(2, n))
    part = np.sort(shuffle[:r])
    pmat = pmat_build(p, q, r + 1)
    rinds = np.arange(r + 1)

    for ii in range(n):
        for jj in range(ii, n):
            rnval = np.random.rand(1)
            # print(np.array((ii >= part)*(ii <= part), dtype=np.bool))

            if ii < part[0]:
                rvalii = 0
            elif ii >= part[-1]:
                rvalii = r
            else:
                for ll in range(r):
                    if part[ll] <= ii < part[ll + 1]:
                        rvalii = rinds[ll]

            if jj < part[0]:
                rvaljj = 0
            elif jj >= part[-1]:
                rvaljj = r
            else:
                for ll in range(r):
                    if part[ll] <= jj < part[ll + 1]:
                        rvaljj = rinds[ll]

            if pmat[rvalii, rvaljj] > rnval:
                amat[ii, jj] = 1
                amat[jj, ii] = 1

    return amat