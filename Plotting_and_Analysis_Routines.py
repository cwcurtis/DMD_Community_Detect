import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dmd_runfile import dmd_window_run


# How to generate examples of the random graphs
def graph_plotter(cmats, shfts, pstr, qstr, rstr, fstr, thrshhld, im_dir, ftag):
    fig1, f1_axes = plt.subplots(ncols=1, nrows=3, constrained_layout=True, figsize=(10, 8))
    ax = f1_axes.flatten()
    nmds = np.shape(cmats[0])[0]
    for ii in range(len(cmats)):
        amat = np.ones((nmds, nmds), dtype=np.int)
        indsskp = np.abs(cmats[ii]) < thrshhld
        amat[indsskp] = 0
        nx.draw_circular(nx.from_numpy_matrix(amat), node_size=20, width=1., ax=ax[ii])
        ax[ii].set_title(r"$\tau$=%d" % shfts[ii])
    plt.savefig(im_dir + "graph_example" + ftag, dpi=150)


def oscillator_plotter(xt, nt_dmd, nt, im_dir, ftag):
    ntr = nt - nt_dmd
    cxt = np.cos(xt[:, ntr:])
    sxt = np.sin(xt[:, ntr:])
    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal')
    ax1.scatter(cxt[:, 0],sxt[:, 0],c='b',s=10,label="$t=16$")
    ax1.scatter(1.1*cxt[:, np.int(nt_dmd/2.)],1.1*sxt[:, np.int(nt_dmd/2.)],c='r',s=10,label="$t=18$")
    ax1.scatter(1.2*cxt[:, nt_dmd-1],1.2*sxt[:, nt_dmd-1],c='k',s=10,label="$t=20$")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.legend()
    plt.savefig(im_dir + "Oscillator_pos" + ftag, dpi=150)


def kmat_comp(dmd_kmodesred, dmd_evls_red, dmd_totmodes, nds):
    indspos = dmd_evls_red.imag > 0
    md_inds = np.arange(dmd_totmodes)
    indskp = md_inds[indspos]
    dmd_rl_kmodes = dmd_kmodesred[:nds, indskp]
    scfacs = np.linalg.norm(dmd_rl_kmodes, 2, 1)
    dmd_rl_kmodes = np.diag(1. / scfacs) @ dmd_rl_kmodes
    kcormat = np.zeros((nds, nds), dtype=np.float64)
    for jj in range(nds):
        for kk in range(jj + 1):
            kcormat[jj, kk] = np.abs(np.sum(dmd_rl_kmodes[jj, :] * np.conj(dmd_rl_kmodes[kk, :])))
    dk = np.diag(np.diag(kcormat))
    zdk = kcormat - dk
    kcormat = dk + zdk + zdk.T
    return kcormat


def graph_comp(kcormat, xt, ntr, thrshhld):
    nds, nds = np.shape(kcormat)
    amat_corr = np.zeros((nds, nds), dtype=np.int)
    indskp = np.abs(kcormat) > thrshhld
    amat_corr[indskp] = 1
    fin_graph = nx.from_numpy_matrix(amat_corr)
    comps = nx.connected_components(fin_graph)
    ndlst = [list(c) for c in sorted(comps, key=len, reverse=True)]
    cntlst = [len(cmp) for cmp in ndlst]
    ncomps = len(ndlst)
    print("Number of components: %d" % ncomps)
    fin_modes = [None] * ncomps
    cnt = 0
    for cmp in ndlst:
        fin_modes[cnt] = xt[cmp, ntr:]
        cnt += 1
    ndprmlst = [item for cmp in ndlst for item in cmp]

    return fin_graph, ndlst, cntlst, fin_modes, ndprmlst


def graph_comp_alt(kcormat, thrshhld):
    nds, nds = np.shape(kcormat)
    amat_corr = np.zeros((nds, nds), dtype=np.int)
    indskp = np.abs(kcormat) > thrshhld
    amat_corr[indskp] = 1
    fin_graph = nx.from_numpy_matrix(amat_corr)
    comps = nx.connected_components(fin_graph)
    ndlst = [list(c) for c in sorted(comps, key=len, reverse=True)]

    return ndlst


def dmd_dendrite_tree(xt, ti, tf, dt, skp, nwin, thrshhld):
    nds, nt = np.shape(xt[:, ::skp])
    dt = dt * skp

    ntw = np.int(np.floor((tf - ti) / dt))
    nstep = np.int(np.floor(ntw / nwin))
    nhlf = np.int(np.floor(nstep / 2))
    taxis = np.linspace(ti, tf, ntw + 1)
    taxpls = taxis[::nstep]
    taxmns = taxis[nhlf:ntw + 1 - nhlf:nstep]
    cxt = np.cos(xt[:, ::skp])
    sxt = np.sin(xt[:, ::skp])
    nds_levels = [None] * (2 * nwin - 1)
    tree_cnts = np.zeros(2 * nwin - 1, dtype=np.int)
    for jj in range(2 * nwin - 1):
        if jj % 2 == 0:
            jref = np.int(jj / 2)
            tr = taxpls[jref + 1]
            tl = taxpls[jref]
        else:
            jref = np.int((jj - 1) / 2)
            tr = taxmns[jref + 1]
            tl = taxmns[jref]
        dmd_evls, dmd_phim, dmd_kmodes, dmd_err = dmd_window_run(tl, tr, dt, nds, cxt, sxt)
        dmd_modecut = np.ma.log10(dmd_err) < -2
        dmd_totmodes = np.sum(dmd_modecut)
        dmd_err_red = dmd_err[dmd_modecut]
        dmd_evls_red = np.log(dmd_evls[dmd_modecut]) / dt
        dmd_phired = dmd_phim[:, dmd_modecut]
        dmd_kmodesred = dmd_kmodes[:, dmd_modecut]
        kcormat = kmat_comp(dmd_kmodesred, dmd_evls_red, dmd_totmodes, nds)
        ndlst = graph_comp_alt(kcormat, thrshhld)
        nds_levels[jj] = ndlst
        tree_cnts[jj] = len(ndlst)

    print(tree_cnts)
    nleaves = np.sum(tree_cnts)
    tree_amat = np.zeros((nleaves, nleaves), dtype=np.int)
    rcnt = 0
    ccnt = tree_cnts[0]
    for jj in range(1, 2 * nwin - 1):
        plvl = nds_levels[jj - 1]
        clvl = nds_levels[jj]
        plvl_cnts = tree_cnts[jj - 1]
        clvl_cnts = tree_cnts[jj]
        lcl_tr = np.zeros((plvl_cnts, clvl_cnts), dtype=np.int)
        for kk in range(plvl_cnts):
            for ll in range(clvl_cnts):
                if set(plvl[kk]).issubset(set(clvl[ll])):
                    lcl_tr[kk, ll] = 1
        tree_amat[rcnt:(rcnt + plvl_cnts), ccnt:(ccnt + clvl_cnts)] = lcl_tr
        rcnt += plvl_cnts
        ccnt += clvl_cnts
    dtamat = np.diag(np.diag(tree_amat))
    tree_amat -= dtamat
    tree_amat += (dtamat + tree_amat.T)
    return tree_amat