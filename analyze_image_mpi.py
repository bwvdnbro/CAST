#!/usr/bin/env python3

# Simple MPI version
# Each rank (process) takes an equal number of files
# all processes communicate to one process (rank 0) at the end through two collective communications

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def sersic(r, I0, rs, n):
    return I0 * np.exp(-((r / rs) ** (1.0 / n)))


def fit_sersic(file):
    data = np.load(file)
    img = data["img"]
    imgx = data["x"]
    imgy = data["y"]
    r = np.sqrt(imgx**2 + imgy**2)
    params, _ = curve_fit(
        sersic,
        r.flatten(),
        img.flatten(),
        p0=(5.0, 1.0, 2.0),
        maxfev=10000,
    )
    n = params[2]
    nexp = data["n"]
    rs = params[1]
    rsexp = data["rs"]
    ndiff = (n - nexp) / nexp
    rsdiff = (rs - rsexp) / rsexp
    xi2 = ndiff**2 + rsdiff**2
    return params[2], params[1]


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("input", nargs="+")
    argparser.add_argument("output")
    args = argparser.parse_args()

    nfile = len(args.input)
    nchunk = int(nfile / size) + (
        nfile % size > 0
    )  # avoid missing files because of round down
    sfile = rank * nchunk
    efile = min((rank + 1) * nchunk, nfile)

    ns = []
    rss = []
    for file in args.input[sfile:efile]:
        n, rs = fit_sersic(file)
        print(rank, file, n, rs)
        ns.append(n)
        rss.append(rs)

    print(rank, len(ns), len(rss))
    allns = comm.gather(ns, root=0)
    allrss = comm.gather(rss, root=0)
    if rank == 0:
        ns = [e for es in allns for e in es]
        rss = [e for es in allrss for e in es]
        pl.plot(ns, rss, ".")
        pl.savefig(args.output, dpi=300)
