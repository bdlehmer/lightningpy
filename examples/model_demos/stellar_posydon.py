

import numpy as np
from scipy.interpolate import interp1d
from lightning.stellar import BPASSModelA24, POSYDON, PEGASEModelA24
from lightning.sfh import PiecewiseConstSFH
import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb


wave_grid = np.logspace(np.log10(0.01),
                        np.log10(10),
                        1000)
filter_labels = ['SDSS_u', 'SDSS_g', 'SDSS_r', 'SDSS_i', 'SDSS_z',
                 'MOIRCS_J', 'MOIRCS_H', 'MOIRCS_Ks',
                 'IRAC_CH1', 'IRAC_CH2', 'IRAC_CH3', 'IRAC_CH4']
redshift = 0.0
#age = [0.0] + list(np.logspace(7, np.log10(13.0e9), 7))
age = np.array([0,3e6,1e7,5e7,1e8])

#pos_sin = POSYDON(filter_labels,
#                age=age,
#                redshift=redshift,
#                wave_grid=wave_grid,
#                binaries=False,
#                nebular_effects=False)

pos_bin = POSYDON(filter_labels,
                age=age,
                redshift=redshift,
                wave_grid=wave_grid,
                binaries=True,
                nebular_effects=False)

bpass_bin = BPASSModelA24(filter_labels,
                 age=age,
                 redshift=redshift,
                 wave_grid=wave_grid,
                 binaries=True,
                 nebular_effects=False)

peg_star = PEGASEModelA24(filter_labels,
                 age=age,
                 redshift=redshift,
                 wave_grid=wave_grid,
                 nebular_effects=False)

bpass_neb = BPASSModelA24(filter_labels,
                age=age,
                redshift=redshift,
                wave_grid=wave_grid,
                binaries=True,
                nebular_effects=True)

pos_neb = POSYDON(filter_labels,
                age=age,
                redshift=redshift,
                wave_grid=wave_grid,
                binaries=True,
                nebular_effects=True)

peg_neb = PEGASEModelA24(filter_labels,
                 age=age,
                 redshift=redshift,
                 wave_grid=wave_grid,
                 nebular_effects=True)

Nmod = len(age) - 1

cm_peg = mpl.colormaps['Greens']
colors_peg = cm_peg(np.linspace(0.2, 0.9, Nmod))[::-1]

cm_pos = mpl.colormaps['Reds']
colors_pos = cm_pos(np.linspace(0.2, 0.9, Nmod))[::-1]

cm_bpass = mpl.colormaps['Purples']
colors_bpass = cm_bpass(np.linspace(0.2, 0.9, Nmod))[::-1]


# Plot Spectra without Nebula
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Panels: upper-left -> age 0-3 Myr, upper-right -> 3-10 Myr,
# lower-left -> 10-50 Myr (combined), lower-right -> 50-100 Myr
panels = [
    {"type": "single", "idx": 0},
    {"type": "single", "idx": 1},
    {"type": "single", "idx": 2},
    {"type": "single", "idx": 3},
]

titles = ["Age = 0-3 Myr", "Age = 3-10 Myr", "Age = 10-50 Myr", "Age = 50-100 Myr"]

f1 = 1
for ax, panel, title in zip(axes, panels, titles):
    i = panel["idx"]
    peg_spec = peg_star.nu_grid_obs * peg_star.Lnu_obs[i, -3, :] / peg_star.mstar[i,-3]*1e6
    bpass_spec = bpass_bin.nu_grid_obs * bpass_bin.Lnu_obs[i, -3, :] / bpass_bin.mstar[i,-3]*1e6
    pos_spec = pos_bin.nu_grid_obs * pos_bin.Lnu_obs[i, 0, :] / pos_bin.mstar[i] * 1e6

    ax.plot(peg_star.wave_grid_rest, peg_spec, color=colors_peg[i], label='PEGASE')
    ax.plot(bpass_bin.wave_grid_rest, bpass_spec, color=colors_bpass[i], label='BPASS')
    ax.plot(pos_bin.wave_grid_rest, pos_spec, color=colors_pos[i], label='POSYDON (binaries)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e4, 1e10)
    ax.axvline(0.0912, color='dimgray', linestyle='--')
    ax.set_xlabel('Rest-Frame Wavelength [µm]')
    ax.set_ylabel(r'$\nu L_\nu\ [L_{\odot} \, (10^{6} M_{\odot})^{-1}]$')
    ax.legend()
    ax.set_title(title)

plt.tight_layout()
fig.savefig('posydon_bin_sfh_seds_stars.pdf')

# Plot Spectra with Nebula
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

f1 = 1
for ax, panel, title in zip(axes, panels, titles):
    i = panel["idx"]
    peg_spec = peg_neb.nu_grid_obs * peg_neb.Lnu_obs[i, -3, 1, :] / peg_neb.mstar[i,-3]*1e6  # use log U = -2
    bpass_spec = bpass_neb.nu_grid_obs * bpass_neb.Lnu_obs[i, -3, 1, :] / bpass_neb.mstar[i,-3]*1e6 # use log U = -2
    pos_spec = pos_neb.nu_grid_obs * pos_neb.Lnu_obs[i, 0, 1,:] / pos_neb.mstar[i] * 1e6

    ax.plot(peg_neb.wave_grid_rest, peg_spec, color=colors_peg[i], label='PEGASE')
    ax.plot(bpass_neb.wave_grid_rest, bpass_spec, color=colors_bpass[i], label='BPASS')
    ax.plot(pos_neb.wave_grid_rest, pos_spec, color=colors_pos[i], label='POSYDON (binaries)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e4, 1e10)
    ax.axvline(0.0912, color='dimgray', linestyle='--')
    ax.set_xlabel('Rest-Frame Wavelength [µm]')
    ax.set_ylabel(r'$\nu L_\nu\ [L_{\odot} \, (10^{6} M_{\odot})^{-1}]$')
    ax.legend()
    ax.set_title(title)

plt.tight_layout()
fig.savefig('posydon_bin_sfh_seds_nebula.pdf')


breakpoint()
