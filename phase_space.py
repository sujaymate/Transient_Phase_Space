#!/usr/bin/env python

from pathlib import Path

from matplotlib import transforms
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def L_Tb(nuW:np.array, Tb:float) -> np.array:
    """ Function to return lines of constant brightness temp.
    Copied for the gnuplot code.
    
    T_B = L/(2k(nu.W)**2)
        = (3.621e+22)*L/(nu.W)**2
        = (         )*L/x**2
     --> L = T_B*x**2*(2.761e-23)    Watts/Hz
           = [ ]*(1.05025e-11)  Jy,kpc^2
 
    NB. 1 W.Hz^{-1} == 1.05026*10^{-11} Jy.kpc^2

    Args:
        nuW (np.array): Fiducial width
        Tb (float): Brightness temperature

    Returns:
        np.array: Luminosity at give Tb and nuW
    """
    
    return Tb*2.761*1.05025e-18*nuW**2


data_path = Path(__file__).parent.joinpath("gach_rud")

# set font
mpl.rcParams["font.family"] = "serif"
#mpl.rc("text", usetex=True)
mpl.rcParams["font.weight"] = 500
mpl.rcParams['mathtext.default'] = 'regular'

# Create figure and set limits, ticks, labels
fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
ax1.set_xscale('log')
ax1.set_yscale('log')

# create abs. luminosity axis
ax2 = ax1.twinx()
ax2.set_yscale('log')

# set limits for all axis
ax1.set_xlim(1e-10, 1e10)
ax1.set_ylim(1e-10, 1e16)
ax2.set_ylim(1e-10, 1e16)

# fix ticks and labels
xticks = np.array([1e-10, 1e-5, 1, 1e5, 1e10])
yticks = np.array([1e-10, 1e-5, 1, 1e5, 1e10, 1e15])
xticklabels = ['10$^{-10}$', '10$^{-5}$', '1.0', '10$^{5}$', '10$^{10}$']
yticklabels = ['10$^{-10}$', '10$^{-5}$', '1.0', '10$^{5}$', '10$^{10}$', '10$^{15}$']
ax1.xaxis.set_ticks(xticks)
ax1.yaxis.set_ticks(yticks)
ax1.yaxis.set_ticklabels(yticklabels, va='bottom', ha='right')
ax1.xaxis.set_ticklabels(xticklabels)

ax2.yaxis.set_ticks(yticks)  # set same ticks as ax1
labels = ['10$^{10}$', '10$^{15}$', '10$^{20}$', '10$^{25}$', '10$^{30}$', '10$^{35}$']
ax2.yaxis.set_ticklabels(labels, va='bottom')

# set labels and fontsizes
ax1.tick_params('both', labelsize=11.5)
ax2.tick_params('y', labelsize=11.5)
ax1.set_xlabel("$\\nu \cdot $W (GHz s)", fontsize=15, labelpad=-2)
ax1.set_ylabel("L$_{\\nu}$ (Jy kpc$^2$)", fontsize=15,labelpad=-10)
ax2.set_ylabel("L$_{\\nu}$ (ergs s$^{-1}$ Hz$^{-1}$)", fontsize=15, labelpad=5)
fig.subplots_adjust(0.09, 0.08, 0.9, .98)

# plot temp lines
trans = ax1.transAxes
rot = np.rad2deg(np.arctan(2)) # somehow rotation equal to aspect ratio works
rot = ax1.transData.transform_angles([rot], np.array([1, 1])[None, :])[0]

Tbs = np.geomspace(1e4, 1e40, 10, endpoint=True)
Tb_text = np.isin(Tbs, [1e4, 1e12, 1e20, 1e28, 1e36])
x_text = [5e8, 5e8, 5e3, 10, 0.065]
j = 0  # iterator for text
for i, Tb in enumerate(Tbs):
    if Tb < 1e13:
        zorder=1
    else:
        zorder=0
    ax1.plot(xticks, L_Tb(xticks, Tb), ls='--', lw=.8, c='#808080', alpha=0.5, dashes=(2, 2), zorder=zorder)
    if Tb_text[i]:
        # temp name
        ax1.text(x_text[j], L_Tb(x_text[j], Tb), f"10$^{{{np.log10(Tb):2.0f}}}$ K",
                 fontsize=12, rotation=rot, va='top', ha='center', zorder=2)
        j+=1

# uncertainty principle region    
ax1.axvspan(1e-10, 1e-9, color='#808080', alpha=0.9, zorder=-1)
ax1.text(0.015, 0.5, "Uncertainty Principle", fontsize=13, rotation=90, va='center', transform=trans)

# coherent/incoherent region
ax1.fill_between(xticks, L_Tb(xticks, 1e12), color='#87CEFA', zorder=0)
ax1.text(0.25e5, 2.5e4, "Coherent Emission", fontsize=13, rotation=rot, va='center', ha='center')
ax1.text(1e5, 0.25e4, "Incoherent Emission", fontsize=13, rotation=rot, va='center', ha='center')

# add arrows
x1 = 0.76; y1 = 0.52
x2 = 0.71; y2 = 0.56
dx = 0.125
ax1.annotate("", xy=(x1+dx, y1 - dx ), xytext=(x1, y1),
             arrowprops=dict(arrowstyle="->, head_length=1,head_width=0.25"), xycoords=trans)
ax1.annotate("", xy=(x2-dx, y2 + dx ), xytext=(x2, y2),
             arrowprops=dict(arrowstyle="->, head_length=1,head_width=0.25"), xycoords=trans)

# set marker and fontsize for source points and labels
ms=5
fs=12

# plot pulsars
psr = np.loadtxt(data_path.joinpath("psrs_2"), usecols=(4,5))
ax1.scatter(*psr.T, c="#0000FF", s=ms)
ax1.text(.25, 0.2, "Pulsars", c='#0000FF', fontsize=fs, va='center', ha='center', transform=trans)

# plot pulsar GRPs
psr_grp = np.loadtxt(data_path.joinpath("GRPs_vals"), usecols=(6,7), skiprows=1)
ax1.scatter(*psr_grp.T, c="#6A5ACD", s=ms)
ax1.text(0.3, 0.51, "Pulsars GRPs", c='#6A5ACD', fontsize=fs, va='center', ha='center', transform=trans)

# plot crab nanoshots
crab_ns = np.loadtxt(data_path.joinpath("crab_nanogiant"), usecols=(1, 0))
ax1.scatter(crab_ns[1], crab_ns[0], c="#656C7F", s=ms)
ax1.text(0.25*0.25, 0.5, "Crab\nnanoshots", c='#656C7F', fontsize=fs, va='top', ha='left', transform=trans)

# plot RRATs
rrat = np.loadtxt(data_path.joinpath("rrats_nohead"), usecols=(4,5))
ax1.scatter(*rrat.T, c="#FF0000", s=ms)
med = np.median(rrat, 0)
ax1.text(0.275, 0.4, "RRATs", c='#FF0000', fontsize=fs, va='center', ha='center', transform=trans)

# plot FRBs
frb = np.loadtxt(data_path.joinpath("frbs_vals_to_plot"), usecols=(1,0), skiprows=1)
ax1.scatter(*frb.T, c="#F08080", s=ms)
ax1.text(0.3, 0.85, "FRBs", c='#F08080', fontsize=fs, va='center', ha='center', transform=trans)

# plot SGR 1935+2154
sgr = np.loadtxt(data_path.joinpath("SGR1935+2154"))
ax1.errorbar(sgr[2]*sgr[3], sgr[0]*sgr[1]**2, yerr=sgr[0]*sgr[1]**3,
             lolims=True, fmt=".", ms=ms, lw=1, c='#166461')
ax1.text(0.35, 0.65, "SGR 1935+2154", c='#166461', fontsize=fs, va='center', ha='center', transform=trans)

# plot GLEAM-X
gx = np.loadtxt(data_path.joinpath("luminosity_nuW.txt"), usecols=(1, 0), skiprows=1)
ax1.scatter(*gx.T, c="#231F20", s=ms)
ax1.text(0.525, 0.39, "GLEAM-X", c='#231F20', fontsize=fs, va='center', ha='center', transform=trans)

# plot AGNs/Blazars/QSO
agns = np.loadtxt(data_path.joinpath("Gosia_AGN_QSO_Blazar_TDE2"), usecols=(1, 6, 8), skiprows=1)
ax1.scatter(agns[:, 0]*86400*agns[:, 2], agns[:, 1]*1.05026e-20, c="#0000BB", s=ms)
ax1.text(0.75, 0.91, "AGNs/Blazars/QSO", c='#0000BB', fontsize=fs, va='center', ha='center', transform=trans)

# GRBs
grbs = np.loadtxt(data_path.joinpath("Gosia_GRB2"), usecols=(1, 6, 8))
ax1.scatter(grbs[:, 0]*86400*grbs[:, 2], grbs[:, 1]*1.05026e-20, c="#d208cc", s=ms)
ax1.text(0.8, 0.8, "GRBs", c='#d208cc', fontsize=fs, va='center', ha='center', transform=trans)

# GW170817
gw = np.loadtxt(data_path.joinpath("gw170817"))
ax1.scatter(*gw, c="#d208cc", s=ms)
ax1.text(0.91, 0.55, "GRB170817", c='#d208cc', fontsize=fs, va='center', ha='center', transform=trans)

# SNs
SN = np.loadtxt(data_path.joinpath("Gosia_SN2"), usecols=(1, 6, 8))
ax1.scatter(SN[:, 0]*86400*SN[:, 2], SN[:, 1]*1.05026e-20, c="#6b4730", s=ms)
ax1.text(0.95, 0.65, "Supernovae", c='#6b4730', fontsize=fs, va='center', ha='center', transform=trans)

# Novae
novae = np.loadtxt(data_path.joinpath("Gosia_Novae2"), usecols=(1, 6, 8))
ax1.scatter(novae[:, 0]*86400*novae[:, 2], novae[:, 1]*1.05026e-20, c="#01748e", s=ms)
ax1.text(0.95, 0.4, "Novae", c='#01748e', fontsize=fs, va='center', ha='center', transform=trans)

# XRBs
xrbs = np.loadtxt(data_path.joinpath("Gosia_XRB2"), usecols=(1, 6, 8))
ax1.scatter(xrbs[:, 0]*86400*xrbs[:, 2], xrbs[:, 1]*1.05026e-20, c="#CD853F", s=ms)
ax1.text(0.76, 0.36, "XRBs", c='#CD853F', fontsize=fs, va='center', ha='center', transform=trans)

# misc (Jupiter DAM and GCRT)
misc = np.loadtxt(data_path.joinpath("misc"), usecols=(0, 1))
ax1.scatter(*misc.T, c="#9d6f46", s=ms)
ax1.text(0.375, 0.04, "Jupiter DAM", c='#9d6f46', fontsize=fs, va='center', ha='center', transform=trans)
ax1.text(0.58, 0.49, "GCRT 1745", c='#9d6f46', fontsize=fs, va='center', ha='center', transform=trans)

# MKT J1704 
mkt = np.loadtxt(data_path.joinpath("flarey_boi"), usecols=(0, 1))
ax1.scatter(*mkt.T, c="#87a922", s=ms)
ax1.text(0.91, 0.348, "MKT J1704", c='#87a922', fontsize=fs, va='center', ha='center', transform=trans)

# RSCVn
rscv = np.loadtxt(data_path.joinpath("Gosia_RSCVn2"), usecols=(1, 6, 8))
ax1.scatter(rscv[:, 0]*86400*rscv[:, 2], rscv[:, 1]*1.05026e-20, c="#293432", s=ms)
ax1.text(0.85, 0.275, "RSCVn", c='#293432', fontsize=fs, va='center', ha='center', transform=trans)

# Magnetic CV
magcv = np.loadtxt(data_path.joinpath("Gosia_MagCV2"), usecols=(1, 6, 8), skiprows=1)
ax1.scatter(magcv[:, 0]*86400*magcv[:, 2], magcv[:, 1]*1.05026e-20, c="#228B22", s=ms)
ax1.text(0.7, 0.22, "Magnetic CV", c='#228B22', fontsize=fs, va='center', ha='center', transform=trans)

# Solar flares
solar = np.loadtxt(data_path.joinpath("solar_vals"), usecols=(4, 5), skiprows=1)
ax1.scatter(*solar.T, c="#ff8b00", s=ms)
ax1.text(0.5, 0.175, "Solar Bursts", c='#ff8b00', fontsize=fs, va='center', ha='center', transform=trans)

# flaring stars
flstars = np.loadtxt(data_path.joinpath("Gosia_flare_stars2"), usecols=(1, 6, 8), skiprows=1)
ax1.scatter(flstars[:, 0]*86400*flstars[:, 2], flstars[:, 1]*1.05026e-20, c="#9d6f46", s=ms)
ax1.text(0.75, 0.07, "Flaring Stars/Brown Dwarves", c='#9d6f46', fontsize=fs, va='center', ha='center', transform=trans)


fig.savefig("phase_space_py.png", dpi=150)
plt.show()