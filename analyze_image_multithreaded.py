#!/usr/bin/env python3

# Multithreaded version
# Severly handicaped because of Global Interpreter Lock, which introduces a lock into the Python interpreter
# itself at a very deep level

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit
from concurrent.futures import ThreadPoolExecutor


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
    print(f"xi2: {xi2}, ndiff: {ndiff}, rsdiff: {rsdiff}")
    return params[2], params[1]


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("input", nargs="+")
    argparser.add_argument("output")
    args = argparser.parse_args()

    with ThreadPoolExecutor(16) as e:
        nrs = list(e.map(fit_sersic, args.input))
    ns = [e[0] for e in nrs]
    rss = [e[1] for e in nrs]

    pl.plot(ns, rss, ".")
    pl.savefig(args.output, dpi=300)
