#!/usr/bin/env python3

"""
Comprehensive Artiverse Survey Toolkit (CAST)

Creates a projection image for all 1000 images in the CAS, and performs
a Sersic fit. Collects all images onto a single web page, and produces
a scatter plot of the Sersic parameters.
"""

import numpy as np
import glob
from astropy.io import fits
import astropy.units as u
import astropy.wcs as wcs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as pl
import scipy.optimize as opt

data_path = "Data"
output_path = "Output"

def sersic(r, I0, rs, n):
    return I0 * np.exp(-((r / rs) ** (1.0 / n)))
    
if __name__ == "__main__":
  images = []
  minmax = None # for plotting
  for image in sorted(glob.glob(f"{data_path}/*.fits")):
    hdul = fits.open(image)
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
    images.append((imgx, imgy, hdul[0].data))
    if minmax is None:
      minmax = [hdul[0].data.min(), hdul[0].data.max()]
    else:
      minmax[0] = min(minmax[0], hdul[0].data.min())
      minmax[1] = max(minmax[1], hdul[0].data.max())
  
  htmlfile = open(f"{output_path}/CAS.html", "w")
  htmlfile.write("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>CAS</title></head><body>")
  htmlfile.write("<h1>CAS galaxies</h1>")
  ns = []
  rss = []
  for i, (x, y, img) in enumerate(images):
    fig, ax = pl.subplots(1, 2, figsize=(10.,5.))
    
    ax[0].contourf(x, y, img, levels=np.linspace(minmax[0], minmax[1], 1000), cmap="nipy_spectral")
    ax[0].set_title(f"CAS {i:03d}")
    ax[0].set_xlabel("x [kpc]")
    ax[0].set_ylabel("y [kpc]")
    
    r = np.sqrt(x**2 + y**2)
    params, _ = opt.curve_fit(
        sersic,
        r.flatten(),
        img.flatten(),
        p0=(5.0, 1.0, 2.0),
        maxfev=10000)
    ns.append(params[2])
    rss.append(params[1])
    ax[1].plot(r.flatten(), img.flatten(), ".")
    rref = np.linspace(r.min(), r.max(), 100)
    ax[1].plot(rref, sersic(rref.value, *params), "-", label=f"n = {params[2]:.2f}, rs = {params[1]:.2f}")
    ax[1].set_xlabel("r [kpc]")
    ax[1].set_ylabel("I [EM]")
    ax[1].legend(loc="upper right")
    
    pl.tight_layout()
    pl.savefig(f"{output_path}/img_{i:03d}.png", dpi=100)
    pl.close()
    
    htmlfile.write(f"<h2>CAS {i:03d}</h2><p><img src=\"img_{i:03d}.png\"/></p>")
  
  fig, ax = pl.subplots(1,1, figsize=(5.,5.))
  ax.plot(ns, rss, ".")
  ax.set_xlabel("Sersic n")
  ax.set_ylabel("Seric rs")
  pl.tight_layout()
  pl.savefig(f"{output_path}/sersic.png", dpi=100)
  htmlfile.write("<h1>Sersic parameters</h1>")
  htmlfile.write("<p><img src=\"sersic.png\"/></p>")
  
  htmlfile.write("</body></html>")
  htmlfile.close()
