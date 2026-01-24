#!/usr/bin/env python3

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl
from scipy.optimize import curve_fit


def sersic(r, I0, rs, n):
    return I0 * np.exp(-((r / rs) ** (1.0 / n)))


def logsersic(r, I0, rs, n):
    return np.log(I0) - (r / rs) ** (1.0 / n)


def logjac(r, I0, rs, n):
    dI0 = np.zeros(r.shape) + 1.0 / I0
    drs = -(1.0 / n) * (r / rs) ** (1.0 / n - 1.0) * (-r / rs**2)
    dn = (r / rs) ** (1.0 / n) * np.log(r / rs) / n**2
    jac = np.zeros((r.shape[0], 3))
    jac[:, 0] = dI0
    jac[:, 1] = drs
    jac[:, 2] = dn
    return jac


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("input")
    argparser.add_argument("output")
    args = argparser.parse_args()

    data = np.load(args.input)

    coords = data["coords"]  # this should be stored in a different, more confusing way
    img = data["img"]

    I0s = data["I0s"]
    ns = data["ns"]
    rss = data["rss"]

    maxx = 1000.0  # this should be stored in metadata
    Nx, Ny = img.shape
    index = 0
    results = []
    for (xc, yc), I0, n, rs in zip(coords, I0s, ns, rss):
        ix = int(xc * Nx / maxx)
        iy = int(yc * Ny / maxx)
        xlim = [max(0, ix - 10), min(Nx, ix + 10)]
        ylim = [max(0, iy - 10), min(Ny, iy + 10)]
        tile = img[ylim[0] : ylim[1], xlim[0] : xlim[1]]
        #    pl.imshow(tile)
        #    pl.gca().axis("off")
        x = np.linspace(
            (xlim[0] + 0.5) * maxx / Nx, (xlim[1] + 0.5) * maxx / Nx, tile.shape[1]
        )
        y = np.linspace(
            (ylim[0] + 0.5) * maxx / Ny, (ylim[1] + 0.5) * maxx / Ny, tile.shape[0]
        )
        xs, ys = np.meshgrid(x, y)
        #    pl.contourf(x, y, tile)
        r = np.sqrt((xs - xc) ** 2 + (ys - yc) ** 2)
        pl.plot(r, tile, "k.")
        rref = np.linspace(0.0, r.max(), 100)
        pl.plot(rref, sersic(rref, I0, rs, n), "-", color="C0")
        params, _ = curve_fit(
            logsersic,
            r.flatten(),
            np.maximum(np.zeros(tile.shape) - 100.0, np.log(tile)).flatten(),
            p0=(1.0, 1.0, 4.0),
            #            bounds=[[0.1, 0.5, 0.3], [3., 2., 10.0]],
            jac=logjac,
            ftol=1.0e-10,
        )
        print(params)
        results.append((params[1], params[2]))
        pl.plot(rref, sersic(rref, *params), "-", color="C1")
        pl.savefig(f"{args.output}_{index:03d}.png", dpi=300, bbox_inches="tight")
        pl.close()
        index += 1
    results = np.array(results)
    pl.plot(results[:, 0], results[:, 1], ".")
    pl.plot(rss, ns, ".")
    pl.savefig(f"{args.output}_result.png", dpi=300, bbox_inches="tight")
