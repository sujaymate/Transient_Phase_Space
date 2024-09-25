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
           = [ ]*(1.05025e-13)  Jy,kpc^2
 
    NB. 1 W.Hz^{-1} == 1.05026*10^{-13} Jy.kpc^2

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

Tbs = np.geomspace(1e4, 1e40, 10, endpoint=True)  # sample Temp values
Tb_text = np.isin(Tbs, [1e4, 1e12, 1e20, 1e28, 1e36])  # only print temp value at these temp
x_text = [5e8, 5e8, 5e3, 10, 0.000065]  # xvals to calc. temp value text position
j = 0  # iterator for text
for i, Tb in enumerate(Tbs):
    ax1.plot(xticks, L_Tb(xticks, Tb), ls='--', lw=.8, c='#808080', alpha=0.5, dashes=(2, 2), zorder=1)
    if Tb_text[i]:
        # print temp value text
        ax1.text(x_text[j], L_Tb(x_text[j], Tb), f"10$^{{{np.log10(Tb):2.0f}}}$ K",
                 fontsize=12, rotation=rot, va='top', ha='center', zorder=2)
        j+=1

# uncertainty principle region    
ax1.axvspan(1e-10, 1e-9, color='#808080', alpha=0.9, zorder=-1)
ax1.text(0.015, 0.5, "Uncertainty Principle", fontsize=13, rotation=90, va='center', transform=trans)

# coherent/incoherent region
ax1.fill_between(xticks, L_Tb(xticks, 1e12), color='#87CEFA', zorder=0)
ax1.text(0.25e5*10, 7.5e4*10, "Coherent Emission", fontsize=13, rotation=rot, va='center', ha='center')
ax1.text(1e5*10, 2.0e4*10, "Incoherent Emission", fontsize=13, rotation=rot, va='center', ha='center')

# add arrows
x1 = 0.77; y1 = 0.5
x2 = 0.71; y2 = 0.56
dx = 0.125
ax1.annotate("", xy=(x1+dx, y1 - dx ), xytext=(x1, y1),
             arrowprops=dict(arrowstyle="->, head_length=1,head_width=0.25"), xycoords=trans)
ax1.annotate("", xy=(x2-dx, y2 + dx ), xytext=(x2, y2),
             arrowprops=dict(arrowstyle="->, head_length=1,head_width=0.25"), xycoords=trans)

# Add CHIME/Slow and CHIME/FRB range
ax1.axvspan(1e-3*.6, 100e-3*.6, color='#445172', lw=1.2, alpha=0.4, zorder=0)
ax1.text(.375, 0.02, "CHIME/FRB", fontsize=13, rotation=90, ha='left', transform=trans)

ax1.axvspan(16e-3*.6, 5*.6, color='#ffcc33', ls='--', lw=1.2, alpha=0.4, zorder=0)
ax1.text(0.45, 0.02, "CHIME/Slow", fontsize=13, rotation=90, ha='left', transform=trans)

# set marker and fontsize for source points and labels
ms=10
fs=11

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
ax1.scatter(*frb.T, c="#841b2d", s=ms)
ax1.text(0.3, 0.85, "FRBs", c='#841b2d', fontsize=fs, va='center', ha='center', transform=trans)

# chime slow new
chime_slow = np.loadtxt(data_path.joinpath("chime_slow_detections_new.dat"), usecols=(1, 2, 3, 4), skiprows=1)
nu_w_slow = chime_slow[0] * chime_slow[1] / 1000
lum_slow = 4*np.pi*(chime_slow[3]*1000)**2*chime_slow[2]  # Jy kpc2
ax1.scatter(nu_w_slow, lum_slow, s=15, c='#10622b')
ax1.errorbar(nu_w_slow, lum_slow, yerr=lum_slow**1.1,
             lolims=True, fmt=".", ms=10, lw=1.1, c='#10622b')

# chime slow r117
chime_slow = np.loadtxt(data_path.joinpath("chime_slow_detections_r117.dat"), usecols=(1, 2, 3, 4), skiprows=1)
nu_w_slow = chime_slow[:, 0] * chime_slow[:, 1] / 1000
lum_slow = 4*np.pi*(chime_slow[:, 3]*1000)**2*chime_slow[:, 2]  # Jy kpc2
ax1.scatter(nu_w_slow, lum_slow, s=15, c='#1e9627')


# plot SGR 1935+2154
sgr = np.loadtxt(data_path.joinpath("SGR1935+2154"))
ax1.errorbar(sgr[2]*sgr[3], sgr[0]*sgr[1]**2, yerr=sgr[0]*sgr[1]**3,
             lolims=True, fmt=".", ms=ms, lw=1, c='#841b2d')
ax1.text(0.35, 0.65, "SGR 1935+2154", c='#841b2d', fontsize=fs, va='center', ha='center', transform=trans)

# plot GLEAM-X
gx = np.loadtxt(data_path.joinpath("luminosity_nuW.txt"), usecols=(1, 0), skiprows=1)
ax1.scatter(*gx.T, c="#231F20", s=ms)
ax1.text(0.47, 0.46, "GLEAM-X", c='#231F20', fontsize=fs, va='center', ha='center', transform=trans)

# plot MWA LPTs
# plot GPM 1839-10
# load data (freq, flux @ freq, flux @ 1GHz, fluence @ freq)
gpm_data = np.loadtxt(data_path.joinpath("GPM1839-10_pulse_table.csv"), delimiter=",", usecols=(4, 5, 6, 7))
gpm_data = gpm_data[gpm_data[:, 1] != 0]
gpm_dist = 5.7 #* 3.0857e21  # only mean dist. taken, errors are ignored
nu_w = gpm_data[:, 0]*1e-3 * gpm_data[:, 3] / gpm_data[:, 1]  # in sec, approximated as fluence / peak flux
nonzero_nu_w = nu_w > 0
nu_w = nu_w[nonzero_nu_w]

# Luminosity of each pulse determined using Eqn 4 in Methods section of Hurley-Walker et al 2023. Omega_1GHz is set to 1
alpha = -3.17  # only the mean value take, errors ignored
beta = -0.26
q = -0.56  # only the mean value take, errors ignored
L0 = 4 * np.pi * gpm_dist**2 * gpm_data[:, 2]*1e-3 * np.sqrt(np.pi/abs(q)) * np.exp(-(alpha + beta + 1)**2/(4*abs(q))) # Jy kp^2
L0 = L0[nonzero_nu_w] #* 1.05e-43
ax1.scatter(nu_w, L0, c="#6A760D", alpha=0.8, s=ms)
ax1.text(0.63, 0.41, "GPM\n1839-10", c='#6A760D', fontsize=fs, va='center', ha='center', transform=trans)

# plot MeerKAT LPT (code from Iris De Ruiter)
psrJ0901_peak_flux = np.array([13.4e-3, 25.4e-3]) # in Jy (Lband, Uband) in image
psrJ0901_dist = np.mean([328e-3, 467e-3]) # average of ymw16 and ne2001 in kpc 

psrJ0901_widths = np.array([2, 2]) # in secs (Lband, Uband) in image
psrJ0901_freq = np.array([1.284, 0.816]) # in GHz

psrJ0901_lum = 4*np.pi*psrJ0901_peak_flux*psrJ0901_dist**2

#[mtp0013_peak_fluxdensities[i]*mtp0013_dist**2 
#               for i in range(len(mtp0013_peak_fluxdensities))] # in Jy kpc^2

ax1.scatter(psrJ0901_freq*psrJ0901_widths, psrJ0901_lum, c="#c60000", s=ms)
ax1.text(0.52, 0.28, "PSR\nJ0901-4046", c='#c60000', fontsize=fs, va='center', ha='center', transform=trans)

# plot ASKAP source (code from Iris De Ruiter, data from Manisha Caleb)
askap_1935_dist = np.mean([4.3, 5.4])  # Mean of NE2001 and YMW16 distances in kpc
askap_1935_peak_flux = np.loadtxt(data_path.joinpath("askap_1935_2148.csv"), usecols=(3), delimiter=',')

tint_spans = np.array([1, 1, 5, 3, 2, 1, 1, 1, 2, 1, 2, 4, 2, 6, 2, 1]) # in number of integrations
askap_freq = 887.5*1e-3 # in GHz

askap_1935_lum = 4*np.pi*askap_1935_peak_flux*1e-3*askap_1935_dist**2  # Jy kpc

plt.scatter(askap_freq*tint_spans*10, askap_1935_lum, c="#cc6600", s=ms)
ax1.text(0.6, 0.49, "ASKAP\n1935+2148", c='#cc6600', fontsize=fs, va='center', ha='center', transform=trans)

# plot LOFAR ILT J1101 + 5521 (code and data from Iris De Ruiter)
ilt_J1101_flux = np.array([68, 78, 256, 46, 93, 123, 41])/1000 # in Jy from Overleaf

LOFAR_freq = 0.144 # GHz
ilt_J1101_duration = np.array([8, 5, 8, 4, 7, 6, 9])*8 #seconds
dist_ilt_J1101 = 0.504  # kpc

ilt_J1101_lum = 4*np.pi*ilt_J1101_flux*dist_ilt_J1101**2  # Jy kpc
plt.scatter(LOFAR_freq*ilt_J1101_duration, ilt_J1101_lum, c="#6a329f", s=ms)
ax1.text(0.63, 0.35, "ILT\nJ2202+5521", c='#6a329f', fontsize=fs, va='center', ha='center', transform=trans)

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
ax1.scatter(SN[:, 0]*86400*SN[:, 2], SN[:, 1]*1.05026e-20, c="#6b4730", alpha=0.7, s=ms)
ax1.text(0.95, 0.65, "Supernovae", c='#6b4730', fontsize=fs, va='center', ha='center', transform=trans)

# Novae
novae = np.loadtxt(data_path.joinpath("Gosia_Novae2"), usecols=(1, 6, 8))
ax1.scatter(novae[:, 0]*86400*novae[:, 2], novae[:, 1]*1.05026e-20, c="#01748e", s=ms)
ax1.text(0.95, 0.4, "Novae", c='#01748e', fontsize=fs, va='center', ha='center', transform=trans)

# XRBs
xrbs = np.loadtxt(data_path.joinpath("Gosia_XRB2"), usecols=(1, 6, 8))
ax1.scatter(xrbs[:, 0]*86400*xrbs[:, 2], xrbs[:, 1]*1.05026e-20, c="#CD853F", s=ms)
ax1.text(0.76, 0.36, "XRBs", c='#CD853F', fontsize=fs, va='center', ha='center', transform=trans)

fig.savefig(data_path.parent.joinpath("phase_space_py.png"), dpi=150)
fig.savefig(data_path.parent.joinpath("phase_space_py.pdf"))
plt.show()
