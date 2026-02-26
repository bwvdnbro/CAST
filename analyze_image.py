#!/usr/bin/env python3

# Clean, serial version
# This version is easy to parallelize because the core logic of the algorithm
# (the part that can be done in parallel)
# is clearly separated into a standalone function

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit

from astropy import wcs
from astropy.io import fits
import astropy.units as u


def sersic(r, I0, rs, n):
    return I0 * np.exp(-((r / rs) ** (1.0 / n)))


def read_fits(input_name):
    hdul = fits.open(input_name)
    w = wcs.WCS(hdul[0].header)
    distance = hdul[0].header["DISTANCE"] * u.Mpc

    p = w.pixel_to_world([[0, 0], [1, 1]], [0, 0])
    dangle = p[1][0].ra - p[0][0].ra
    dx = ((dangle / u.radian) * distance).to(u.kpc)

    imgcoord = np.linspace(0.0, dx * w.array_shape[0], w.array_shape[0])
    imgcoord -= 0.5 * dx * w.array_shape[0]
    imgx, imgy = np.meshgrid(imgcoord, imgcoord)

    return imgx, imgy, hdul[0].data


def fit_sersic(file):
    imgx, imgy, img = read_fits(file)
    r = np.sqrt(imgx**2 + imgy**2)
    params, _ = curve_fit(
        sersic,
        r.flatten(),
        img.flatten(),
        p0=(5.0, 1.0, 2.0),
        maxfev=10000,
    )
    return params[2], params[1]


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("input", nargs="+")
    argparser.add_argument("output")
    args = argparser.parse_args()

    ns = []
    rss = []
    for file in args.input:
        n, rs = fit_sersic(file)
        print(file, n, rs)
        ns.append(n)
        rss.append(rs)

    pl.plot(ns, rss, ".")
    pl.savefig(args.output, dpi=300)
