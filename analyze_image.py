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
    dax = w.wcs.cdelt[0] * u.degree
    day = w.wcs.cdelt[1] * u.degree
    dx = (dax.to(u.radian) * distance).to(u.kpc, equivalencies=u.dimensionless_angles())
    dy = (day.to(u.radian) * distance).to(u.kpc, equivalencies=u.dimensionless_angles())

    ny, nx = hdul[0].data.shape
    imgcoordx = np.linspace(0.0, dx * nx, nx) - 0.5 * dx * nx
    imgcoordy = np.linspace(0.0, dy * ny, ny) - 0.5 * dy * ny
    imgx, imgy = np.meshgrid(imgcoordx, imgcoordy)

    return imgx, imgy, hdul[0].data


counter = 0


def fit_sersic(file):
    global counter
    imgx, imgy, img = read_fits(file)
    r = np.sqrt(imgx**2 + imgy**2)
    if False:
        data = np.load(file.replace(".fits", ".npz"))
        imgx2 = data["x"]
        imgy2 = data["y"]
        pl.plot(imgx2.flatten(), imgx.flatten(), ".")
        pl.plot(
            [imgx2.flatten().min(), imgx2.flatten().max()],
            [imgx2.flatten().min(), imgx2.flatten().max()],
            "-",
        )
        pl.plot(imgy2.flatten(), imgy.flatten(), ".")
        pl.plot(
            [imgy2.flatten().min(), imgy2.flatten().max()],
            [imgy2.flatten().min(), imgy2.flatten().max()],
            "-",
        )
        pl.savefig(f"test_{counter:03d}.png", dpi=300)
        counter += 1
        pl.close()
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
