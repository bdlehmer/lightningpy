'''
posydon.py

Stellar population modeling scaffold for POSYDON.

This module provides a POSYDON class that follows the interface and style of
bpass.py / pegase.py. It is a lightweight scaffold: the _construct_model method
provides the canonical signature and performs basic modeldir checks; actual
POSYDON file parsing should be implemented where indicated.
'''

import numpy as np
import h5py

import astropy.units as u
import astropy.constants as const
from astropy.table import Table 
from astropy.io import ascii
import pdb

from scipy.interpolate import interp1d, interpn
from scipy.integrate import trapezoid as trapz

from ..base import BaseEmissionModel

__all__ = ['POSYDON']


class POSYDON(BaseEmissionModel):
    """Stellar emission models generated using POSYDON.

    This class is a drop-in scaffold modeled after BPASSModel/PEGASEModel and
    exposes the same construction signature so it can be wired into
    lightning.py similarly. The actual POSYDON parsing and gridding logic
    needs to be implemented based on the file formats in
    lightning/data/models/POSYDON/.
    """

    model_name = 'POSYDON-Stellar'
    model_type = 'Stellar-Emission'
    gridded = False

    def _construct_model(self, age=None, lognH=2.0, step=True, Z_met=0.020,
                         wave_grid=None, cosmology=None, binaries=True,
                         nebular_effects=True, line_labels=None, dust_grains=False):
        """Canonical constructor signature matching other stellar model classes.

        Parameters mirror the conventions used in bpass.py and pegase.py so
        instances of this class can be interchanged with minimal changes.

        NOTE: This implementation currently only performs basic directory
        checks and sets up placeholders. Implement actual model reading and
        interpolation to populate attributes like self.age, self.mstar,
        self.Lbol, self.q0, self.line_lum, and the spectral arrays.
        """

        # Set model directory to the POSYDON subfolder of the configured modeldir
        self.modeldir = self.modeldir.joinpath('POSYDON')

        if not self.modeldir.exists():
            raise FileNotFoundError(f"POSYDON model directory not found: {self.modeldir}")

        # Choose the single/binary fullgrid HDF5 file provided with the models.
        if binaries:
            fname = 'POSYDON_fullgrid_g_binary.h5'
        else:
            fname = 'POSYDON_fullgrid_g_single.h5'

        fpath = self.modeldir.joinpath(fname)
        if not fpath.exists():
            raise FileNotFoundError(f"Expected POSYDON HDF5 file not found: {fpath}")

        # Open HDF5 and read canonical datasets inspired by the BPASS loader.
        f = h5py.File(fpath.open('rb'))

        # Basic grids
        wave_model = f['wave'][:].astype(float)            # Angstroms
        nu_model = f['nu'][:].astype(float)
        time = f['age'][:].astype(float)
        Zstars = np.atleast_1d(f['Zstars'][:].astype(float))
        # Read logU if present (always store on self even if nebular_effects is False)
        logU = np.array(f['logU'][:]).astype(float) if 'logU' in f else None
        self.logU = logU

        # Common metadata
        self.Zmet = list(Zstars)
        self.line_names = None
        if 'lines' in f and 'names' in f['lines']:
            # HDF5 stores bytes; decode to strings
            try:
                self.line_names = [n.decode('utf-8') for n in f['lines/names'][:]]
            except Exception:
                self.line_names = list(f['lines/names'][:])

        # Stellar quantities
        mstar = np.array(f['mstar'][:]).astype(float)   # shape: (ntime, nZ)
        Lbol = np.array(f['Lbol'][:]).astype(float)
        # logq0 or nion
        if 'logq0' in f:
            logq0 = np.array(f['logq0'][:]).astype(float)
            # convert to q0 if needed later; store as-is
        else:
            logq0 = None


        # Spectral datasets
        # 'spec/noneb' expected shape (ntime, nZ, nwave)
        # 'spec/neb' may have shape (ntime, nZ, nlogU, Nextra, nwave)
        has_neb = ('spec' in f) and ('neb' in f['spec'])
        if has_neb and nebular_effects:
            spec_neb = np.array(f['spec/neb'][:]).astype(float)
            # Collapse any extraneous singleton axes to match (ntime, nZ, nlogU, nwave)
            if spec_neb.ndim == 5:
                # At the moment this is collapsing the nH dimension to 10^2 cm^-2
                # In the future, we will want nH to be a variable
                # choose first index of the extra axis if present
                # resulting shape -> (ntime, nZ, nlogU, nwave)
                spec_neb = spec_neb[..., 0, :]
            self.logU = logU
            lnu_model = spec_neb
            self.nebular = True
            self.Nparams = 2
            self.param_names = ['Zmet', 'logU']
            self.param_descr = ['Metallicity (mass fraction, where solar = 0.020)',
                                'log10 of the ionization parameter']
            self.param_names_fncy = [r'$Z$', r'$\log \mathcal{U}$']
            self.param_bounds = np.array([[1e-5, 0.040],
                                          [-4, -1]])
        else:
            spec_noneb = np.array(f['spec/noneb'][:]).astype(float)
            # ensure shape (ntime, nZ, nwave)
            if spec_noneb.ndim == 3:
                lnu_model = spec_noneb
            elif spec_noneb.ndim == 4:
                # possible extra axes: collapse last-but-one if singleton
                lnu_model = spec_noneb[..., 0, :]
            else:
                raise ValueError('Unexpected spec/noneb shape in POSYDON file: %s' % (spec_noneb.shape,))
            self.nebular = False
            self.Nparams = 1
            self.param_names = ['Zmet']
            self.param_descr = ['Metallicity (mass fraction, where solar = 0.020)']
            self.param_names_fncy = [r'$Z$']
            self.param_bounds = np.array([1e-5, 0.040]).reshape(1,2)

        # Lines luminosities (if present)
        if 'lines' in f and 'lum' in f['lines']:
            lines_lum = np.array(f['lines/lum'][:]).astype(float)
            # Canonical targets:
            #   nebular -> (ntime, nZ, nlogU, Nlines)
            #   non-nebular -> (ntime, nZ, Nlines)
            ntime = len(time)
            nZ = len(Zstars)
            nlogU = len(self.logU) if getattr(self, 'logU', None) is not None else 0
            Nlines = len(self.line_names) if self.line_names is not None else None

            # Collapse an extra axis if present (e.g., Nextra)
            if lines_lum.ndim == 5:
                # assume format (ntime, nZ, nlogU, Nextra, Nlines) in some order
                # prefer to take index 0 of the penultimate axis
                lines_lum = lines_lum[..., 0, :]

            # Deterministic canonicalization: map axes to (ntime, nZ, nlogU?, Nlines)
            arr = np.array(lines_lum)
            arr = np.squeeze(arr)
            shape = arr.shape
            # target sizes
            Nlines = len(self.line_names) if self.line_names is not None else None

            def find_axis_by_size(sz, used=()):
                if sz is None:
                    return None
                for i,s in enumerate(shape):
                    if i in used:
                        continue
                    if s == sz:
                        return i
                return None

            time_ax = find_axis_by_size(ntime)
            z_ax = find_axis_by_size(nZ, used=() )
            logu_ax = find_axis_by_size(nlogU) if nlogU > 0 else None
            lines_ax = find_axis_by_size(Nlines)

            # If lines axis not found, assume last axis is lines
            if lines_ax is None:
                lines_ax = len(shape) - 1

            # If time axis not found, assume axis 0
            if time_ax is None:
                time_ax = 0

            # Build permutation: start with time
            perm = [time_ax]
            # add z axis if present, otherwise will insert singleton
            if z_ax is not None and z_ax not in perm:
                perm.append(z_ax)
            # add logu axis if present and nebular_effects True
            if nebular_effects and (logu_ax is not None) and (logu_ax not in perm):
                perm.append(logu_ax)
            # ensure lines axis is last in perm; we'll move it later
            remaining = [i for i in range(len(shape)) if i not in perm and i != lines_ax]
            perm += remaining
            perm.append(lines_ax)

            try:
                arr = np.transpose(arr, perm)
            except Exception:
                arr = np.squeeze(lines_lum)

            # Now arr axes are something like (time, maybe Zmet, maybe logU, ..., lines)
            # If arr has 3 dims and looks like (ntime, nlogU, Nlines) but nZ==1, insert singleton metallicity axis
            if arr.ndim == 3:
                if (arr.shape[0] == ntime and nlogU > 0 and arr.shape[1] == nlogU and (Nlines is None or arr.shape[2] == Nlines)):
                    # make (ntime, 1, nlogU, Nlines) so subsequent processing is consistent
                    arr = arr[:, np.newaxis, :, :]
            # If non-nebular, collapse any logU axis (take index 0)
            if not nebular_effects:
                if nlogU > 0:
                    # find axis equal to nlogU and take first index
                    for ax in range(arr.ndim - 1):
                        if arr.shape[ax] == nlogU:
                            arr = np.take(arr, 0, axis=ax)
                            break
            # Ensure arr has at least three axes (ntime, nZ, Nlines)
            if arr.ndim == 1:
                arr = arr[:, np.newaxis, np.newaxis]
            if arr.ndim == 2:
                arr = arr[:, np.newaxis, :]
            # If nebular_effects True and we expect a logU axis but it's missing, insert singleton
            if nebular_effects and (nlogU > 0) and arr.ndim == 3:
                arr = arr[:, :, np.newaxis, :]

            lines_lum = arr
        else:
            lines_lum = None

        f.close()

        # Basic setup similar to BPASSModel: set age grid, check against Universe age
        if cosmology is None:
            from astropy.cosmology import FlatLambdaCDM
            cosmology = FlatLambdaCDM(H0=70, Om0=0.3)

        univ_age = cosmology.age(self.redshift).value * 1e9

        if (age is None):
            if (not step):
                # Truncate model time grid to the age of the Universe
                self.age = np.array(list(time[time <= univ_age]) + [univ_age])
            else:
                raise ValueError('For piecewise SFH, age bins for stellar models must be specified.')
        else:
            self.age = age

        assert (~np.any(self.age > univ_age)), 'The provided ages cannot exceed the age of the Universe at z.'
        self.step = step

        # Store basic scalar arrays
        self.metallicity = Z_met

        # Arrange arrays to match interface: lnu_model should be (ntime, nZ, [...], nwave)
        # For the continuous-age interface, interpolate along time axis to requested ages
        # lnu_model currently has time as axis 0
        # If nebular, lnu_model shape -> (ntime, nZ, nlogU, nwave)

        # Build observed-frame arrays
        wave_model_obs = wave_model * (1 + self.redshift)
        nu_model_obs = nu_model / (1 + self.redshift)

        lnu_obs = lnu_model * (1 + self.redshift)

        if (self.step):
            # Piecewise (binned) SFH: integrate the source model over each age bin.
            Nbins = len(self.age) - 1
            dt_bins = np.array(self.age[1:]) - np.array(self.age[:-1])
            # Estimate the native time resolution of the source model
            src_dt = np.min(np.diff(time)) if len(time) > 1 else 0.0
            if np.any(dt_bins < src_dt):
                raise ValueError('The minimum age bin width is %.3e years; this is set by the time resolution of the source models.' % (src_dt,))
            self.Nages = Nbins

            # Build source bin edges from model time centres
            if len(time) > 1:
                mid = 0.5 * (time[:-1] + time[1:])
                edges = np.hstack([time[0] - (time[1] - time[0]) / 2.0, mid, time[-1] + (time[-1] - time[-2]) / 2.0])
            else:
                # Single time point: treat it as a single bin spanning itself
                edges = np.array([time[0] - 0.5, time[0] + 0.5])
            time_lo = edges[:-1]
            time_hi = edges[1:]
            deltat = time_hi - time_lo 

            # Prepare output arrays
            q0_age = np.zeros((Nbins, len(self.Zmet)), dtype='double') 
            if self.nebular:
                lnu_age = np.zeros((Nbins, len(self.Zmet), len(self.logU), len(wave_model)), dtype='double')
                llines_age = np.zeros((Nbins, len(self.Zmet), len(self.logU), len(self.line_names))) if self.line_names is not None else None
            else:
                lnu_age = np.zeros((Nbins, len(self.Zmet), len(wave_model)), dtype='double')
                llines_age = np.zeros((Nbins, len(self.Zmet), len(self.line_names))) if self.line_names is not None else None

            Lbol_age = np.zeros((Nbins, len(self.Zmet)), dtype='double')
            mstar_age = np.zeros((Nbins, len(self.Zmet)), dtype='double')

            # Convert logq0 to linear if present and ensure shape (ntime, nZ)
            if logq0 is not None:
                q0_lin = 10**(logq0.squeeze())
                q0_lin = np.atleast_1d(q0_lin)
                nZ = len(self.Zmet)
                if q0_lin.ndim == 1:
                    # single-column q0 -> replicate across metallicities
                    q0_lin = np.repeat(q0_lin[:, None], nZ, axis=1)
                elif q0_lin.ndim == 2:
                    # if second dimension doesn't match nZ but transpose would, fix it
                    if q0_lin.shape[1] != nZ and q0_lin.shape[0] == nZ and q0_lin.shape[1] == len(time):
                        q0_lin = q0_lin.T
                    elif q0_lin.shape[1] != nZ:
                        raise ValueError('logq0 dataset shape is incompatible with Z grid')
            else:
                q0_lin = None
           

            # Helper to sum and align line arrays into target shape
            def _sum_align_lines(arr, weights, target_shape):
                # arr: ndarray with leading time axis, weights: array of shape (ntime,) or scalar
                weights = np.atleast_1d(weights)
                summed = np.sum(arr * weights.reshape((-1,) + (1,) * (arr.ndim - 1)), axis=0)
                summed = np.squeeze(summed)
                # If shapes match, return
                if summed.shape == target_shape:
                    return summed
                # Try permutations + broadcasting to match target_shape
                from itertools import permutations
                import numpy as _np
                for perm in permutations(range(summed.ndim)):
                    permuted = np.transpose(summed, perm)
                    try:
                        b = _np.broadcast_to(permuted, target_shape)
                        return b
                    except Exception:
                        continue
                # If still mismatched but total size equal, reshape
                if _np.prod(summed.shape) == _np.prod(target_shape):
                    return summed.reshape(target_shape)
                # As a last resort, try broadcasting summed directly (maybe singleton axes are missing)
                try:
                    return _np.broadcast_to(summed, target_shape)
                except Exception:
                    pass
                raise ValueError(f'Could not align summed lines array of shape {summed.shape} to target {target_shape}')

            for i in range(Nbins):
                ti = self.age[i]
                tf = self.age[i + 1]

                fullbins = (time_hi < tf) & (time_lo >= ti)
                partialbinlo = ((time_lo < ti) & (time_hi > ti))
                partialbinhi = ((time_lo < tf) & (time_hi > tf))

                # Sum full bins
                if q0_lin is not None:
                    q0_age[i, :] = np.sum(q0_lin[fullbins, :] * deltat[fullbins, None], axis=0)
                Lbol_age[i, :] = np.sum(np.atleast_2d(Lbol[fullbins]) * deltat[fullbins, None], axis=0)
                mstar_age[i, :] = np.sum(np.atleast_2d(mstar[fullbins]) * deltat[fullbins, None], axis=0)

                if self.nebular:
                    lnu_age[i, ...] = np.sum(lnu_obs[fullbins, ...] * deltat[fullbins, None, None, None], axis=0)
                    if lines_lum is not None:
                        # lines_lum expected (ntime, nZ, nlogU, Nlines)
                        tmp = np.sum(lines_lum[fullbins, ...] * deltat[fullbins, None, None, None], axis=0)
                        # tmp should be (nZ, nlogU, Nlines)
                        # If axis order differs, try to locate nZ, nlogU, Nlines
                        if tmp.shape != (len(self.Zmet), len(self.logU), len(self.line_names)):
                            # try to permute axes to match
                            axes = list(tmp.shape)
                            try:
                                idx_nZ = axes.index(len(self.Zmet))
                                idx_nlogU = axes.index(len(self.logU))
                                idx_Nlines = axes.index(len(self.line_names))
                                tmp = np.transpose(tmp, (idx_nZ, idx_nlogU, idx_Nlines))
                            except ValueError:
                                pass
                        llines_age[i, ...] = tmp
                else:
                    lnu_age[i, ...] = np.sum(lnu_obs[fullbins, ...] * deltat[fullbins, None, None], axis=0)
                    if lines_lum is not None:
                        # lines_lum may have shape (ntime, nZ, nlogU, Nlines) or (ntime, nZ, Nlines)
                        block = lines_lum[fullbins, ...]
                        # Sum over time axis with correct broadcasting
                        weights = deltat[fullbins]
                        tmp = np.sum(block * weights.reshape((-1,) + (1,) * (block.ndim - 1)), axis=0)
                        # tmp may be (nZ, Nlines) or (nZ, nlogU, Nlines) or permuted; try to reduce to (nZ, Nlines)
                        if tmp.ndim == 3:
                            # assume (nZ, nlogU, Nlines) -> take first logU index
                            # but if axes are permuted, try to find Nlines axis
                            axes = list(tmp.shape)
                            if len(self.line_names) in axes:
                                idx_N = axes.index(len(self.line_names))
                                if idx_N != 2:
                                    # move lines axis to last
                                    order = [i for i in range(tmp.ndim) if i != idx_N] + [idx_N]
                                    tmp = np.transpose(tmp, order)
                            # now take first index of middle axis (logU)
                            tmp = tmp[:, 0, :]
                        elif tmp.ndim == 2:
                            # if shape is (Nlines, nZ), transpose
                            if tmp.shape[0] == len(self.line_names) and tmp.shape[1] == len(self.Zmet):
                                tmp = tmp.T
                        # final check
                        if tmp.shape != (len(self.Zmet), len(self.line_names)):
                            # attempt reshape if possible
                            if np.prod(tmp.shape) == len(self.Zmet) * len(self.line_names):
                                tmp = tmp.reshape((len(self.Zmet), len(self.line_names)))
                            else:
                                raise ValueError(f'Cannot reduce lines_lum block shape {tmp.shape} to target {(len(self.Zmet), len(self.line_names))}')
                        llines_age[i, ...] = tmp

                # Partial low-edge bin contribution
                if np.any(partialbinlo):
                    deltat_partial = time_hi[partialbinlo] - ti
                    if q0_lin is not None:
                        q0_age[i, :] += np.squeeze(q0_lin[partialbinlo, :] * deltat_partial)
                    mstar_age[i, :] += np.squeeze(np.atleast_2d(mstar[partialbinlo]) * deltat_partial)
                    Lbol_age[i, :] += np.squeeze(np.atleast_2d(Lbol[partialbinlo]) * deltat_partial)
                    lnu_age[i, ...] += np.squeeze(lnu_obs[partialbinlo, ...] * deltat_partial)
                    if lines_lum is not None:
                        block = lines_lum[partialbinlo, ...]
                        weights = deltat_partial
                        tmp = np.sum(block * weights.reshape((-1,) + (1,) * (block.ndim - 1)), axis=0)
                        # reduce tmp to (nZ, Nlines)
                        if tmp.ndim == 3:
                            axes = list(tmp.shape)
                            if len(self.line_names) in axes:
                                idx_N = axes.index(len(self.line_names))
                                if idx_N != 2:
                                    order = [i for i in range(tmp.ndim) if i != idx_N] + [idx_N]
                                    tmp = np.transpose(tmp, order)
                            tmp = tmp[:, 0, :]
                        elif tmp.ndim == 2:
                            if tmp.shape[0] == len(self.line_names) and tmp.shape[1] == len(self.Zmet):
                                tmp = tmp.T
                        if tmp.shape != (len(self.Zmet), len(self.line_names)):
                            if np.prod(tmp.shape) == len(self.Zmet) * len(self.line_names):
                                tmp = tmp.reshape((len(self.Zmet), len(self.line_names)))
                            else:
                                raise ValueError(f'Cannot reduce lines_lum partial block shape {tmp.shape} to target {(len(self.Zmet), len(self.line_names))}')
                        llines_age[i, ...] += tmp

                # Partial high-edge bin contribution
                if np.any(partialbinhi):
                    deltat_partial = tf - time_lo[partialbinhi]
                    if q0_lin is not None:
                        q0_age[i, :] += np.squeeze(q0_lin[partialbinhi, :] * deltat_partial)
                    mstar_age[i, :] += np.squeeze(np.atleast_2d(mstar[partialbinhi]) * deltat_partial)
                    Lbol_age[i, :] += np.squeeze(np.atleast_2d(Lbol[partialbinhi]) * deltat_partial)
                    lnu_age[i, ...] += np.squeeze(lnu_obs[partialbinhi, ...] * deltat_partial)
                    if lines_lum is not None:
                        block = lines_lum[partialbinhi, ...]
                        weights = deltat_partial
                        tmp = np.sum(block * weights.reshape((-1,) + (1,) * (block.ndim - 1)), axis=0)
                        # reduce tmp to (nZ, Nlines)
                        if tmp.ndim == 3:
                            axes = list(tmp.shape)
                            if len(self.line_names) in axes:
                                idx_N = axes.index(len(self.line_names))
                                if idx_N != 2:
                                    order = [i for i in range(tmp.ndim) if i != idx_N] + [idx_N]
                                    tmp = np.transpose(tmp, order)
                            tmp = tmp[:, 0, :]
                        elif tmp.ndim == 2:
                            if tmp.shape[0] == len(self.line_names) and tmp.shape[1] == len(self.Zmet):
                                tmp = tmp.T
                        if tmp.shape != (len(self.Zmet), len(self.line_names)):
                            if np.prod(tmp.shape) == len(self.Zmet) * len(self.line_names):
                                tmp = tmp.reshape((len(self.Zmet), len(self.line_names)))
                            else:
                                raise ValueError(f'Cannot reduce lines_lum partial block shape {tmp.shape} to target {(len(self.Zmet), len(self.line_names))}')
                        llines_age[i, ...] += tmp


            # After integration over time bins, set q0_age to None if not available
            if q0_lin is None:
                q0_age = None

        else:
            # Continuous age grid: interpolate model to requested ages
            from scipy.interpolate import interp1d
            finterp = interp1d(time, lnu_obs, axis=0, bounds_error=False, fill_value=0.0)
            lnu_age = finterp(self.age)

            # Interpolate scalar quantities
            if logq0 is not None:
                q0_age = np.interp(self.age, time, 10**(logq0.squeeze()))
            else:
                q0_age = None
            # Prepare Lbol and mstar arrays: preserve trailing dims if present
            shape_tail = Lbol.shape[1:] if Lbol.ndim > 1 else ()
            Lbol_age = np.zeros((len(self.age),) + tuple(shape_tail))
            mstar_age = np.zeros_like(Lbol_age)
            if Lbol.ndim == 1:
                Lbol_age[:] = np.interp(self.age, time, Lbol)
                mstar_age[:] = np.interp(self.age, time, mstar)
            else:
                for j in range(Lbol.shape[1]):
                    Lbol_age[:, j] = np.interp(self.age, time, Lbol[:, j])
                    mstar_age[:, j] = np.interp(self.age, time, mstar[:, j])

        # Assign attributes expected by the rest of the codebase
        self.mstar = mstar_age
        self.Lbol = Lbol_age
        self.q0 = q0_age
        # If step integration produced binned line luminosities, use those; otherwise fall back to the file-provided lines_lum.
        if 'llines_age' in locals() and llines_age is not None:
            self.line_lum = llines_age
        else:
            self.line_lum = lines_lum

        # Set wave and nu grids (rest-frame). If caller provided a wave_grid, resample
        # the model spectra to that grid (assumed to be in Angstroms, matching wave_model).
        if wave_grid is not None:
            # Interpolate spectral axis (last axis) to user-provided grid
            from scipy.interpolate import interp1d
            finterp = interp1d(wave_model, lnu_age, axis=-1, bounds_error=False, fill_value=0.0)
            lnu_age_interp = finterp(wave_grid)
            # Ensure non-negative
            lnu_age_interp[lnu_age_interp < 0.0] = 0.0

            # Compute frequency grid from wavelength (Angstroms -> Hz)
            c_AA = const.c.to(u.AA / u.s).value
            nu_grid = c_AA / wave_grid

            self.wave_grid_rest = np.array(wave_grid)
            self.wave_grid_obs = self.wave_grid_rest * (1 + self.redshift)
            self.nu_grid_rest = nu_grid
            self.nu_grid_obs = self.nu_grid_rest * (1 + self.redshift)
            self.Lnu_obs = lnu_age_interp
        else:
            self.wave_grid_rest = wave_model
            self.wave_grid_obs = wave_model_obs
            self.nu_grid_rest = nu_model
            self.nu_grid_obs = nu_model_obs
            # Lnu_obs: shape (Nages, ... , Nwave). Keep consistent naming with BPASS (lnu_obs)
            self.Lnu_obs = lnu_age

    def get_mstar_coeff(self, Z):
        '''Return the Mstar coefficients as a function of age for a given metallicity.

        Parameters
        ----------
        Z : array-like (Nmodels,)
            Stellar metallicity

        Returns
        -------
        Mstar : (Nmodels, Nages)
            Surviving stellar mass as function of age per 1 Msun yr-1 of SFR.
        '''

        Z = np.atleast_1d(Z)
        # Handle single-metallicity grid: return mstar column repeated for each requested Z.
        if np.asarray(self.Zmet).size == 1:
            base = np.take(self.mstar, 0, axis=1)  # shape (Nages,)
            Mstar = np.repeat(base[None, :], Z.size, axis=0)  # (Nmodels, Nages)
            return Mstar
        # Otherwise interpolate in metallicity dimension and return (Nmodels, Nages)
        finterp_mstar = interp1d(self.Zmet, self.mstar, axis=1, bounds_error=False, fill_value='extrapolate')
        return np.swapaxes(finterp_mstar(Z), 0, 1)



    def get_model_lnu_hires(self, sfh, sfh_param, params, exptau=None, exptau_youngest=None, stepwise=False):
        '''Construct the high-res stellar spectrum.

        Given a SFH instance and set of parameters, the corresponding high-resolution spectrum
        is constructed. Optionally, attenuation is applied and the attenuated power is returned.

        Parameters
        ----------
        sfh : instance of lightning.sfh.PiecewiseConstSFH or lightning.sfh.FunctionalSFH
            Star formation history model.
        sfh_params : np.ndarray, (Nmodels, Nparam) or (Nparam,), float32
            Parameters for the star formation history.
        params : np.ndarray, (Nmodels, 1) or (Nmodels,)
            Values for logU, if the model includes a nebular component.
        exptau : np.ndarray, (Nmodels, Nwave) or (Nwave,), float32
            ``exp(-tau)`` as a function of wavelength. If this is 2D, the
            size of the first dimension must match the size of the first
            dimension of ``sfh_params``.
        exptau_youngest : np.ndarray, (Nmodels, Nwave) or (Nwave,), float32
            This doesn't do anything at the moment, until I figure out how
            to flexibly decide which ages to apply the birth cloud attenuation to.
        stepwise : bool
            If true, the spectrum is returned as a function of stellar age.

        Returns
        -------
        lnu_attenuated : np.ndarray, (Nmodels, Nwave), (Nmodels, Nages, Nwave), or (Nwave,), float32
            The stellar spectrum as seen after the application of the ISM dust
            attenuation model.
        lnu_unattenuated : np.ndarray, (Nmodels, Nwave), (Nmodels, Nages, Nwave), or (Nwave,), float32
            The intrinsic stellar spectrum.
        L_TIR : np.ndarray, (Nmodels,) or (Nmodels, Nages)
            The total attenuated power of the stellar population.

        '''

        # sfh_shape = sfh.shape # expecting ndarray(Nmodels, n_steps)
        if (len(sfh_param.shape) == 1):
            sfh_param = sfh_param.reshape(1, -1)

        if (self.step):
            assert (sfh.type == 'piecewise'), 'Binned stellar populations require a piecewise-defined SFH.'

        Nmodels = sfh_param.shape[0]

        if (self.nebular):
            assert (params is not None), 'POSYDON models with the Cloudy component enabled require logU to be specified.'
            assert (params.shape[0] == Nmodels), 'First dimension of logU array must match first dimension of SFH.'
            ob_mask = self._check_bounds(params)
            if np.any(ob_mask):
                raise ValueError('%d logU value(s) are out of bounds [-4,-1]' % (np.count_nonzero(ob_mask)))

        # Explicit dependence of the attenuation on stellar age is not currently fully implemented, but would be easy to
        # do so, if we make our attenuation model functions return arrays shaped like (Nmodels, Nages, N)
        if (exptau is None):
            exptau = np.full((Nmodels,len(self.wave_grid_rest)),1)
        else:
            assert (exptau.shape[0] == Nmodels), 'First dimension of exptau must match first dimension of SFH.'

        if (exptau_youngest is None):
            exptau_youngest = exptau
        else:
            assert (exptau.shape[0] == Nmodels), 'First dimension of exptau_youngest must match first dimension of SFH.'


        if (self.nebular):
            # No boundary handling since we did it above.
            #print('Lnu_obs gridded shape: ', self.Lnu_obs.shape, len(self.Lnu_obs.shape))
            Zmet = params[:,0]
            logU = params[:,1]

            Lnu_transp = np.transpose(self.Lnu_obs, axes=[1,2,0,3])
            # Handle case where metallicity grid has a single point: interpolate only in logU
            if np.asarray(self.Zmet).size == 1:
                # Lnu_transp shape -> (nZ=1, nlogU, Nages, Nwave)
                Lnu_z0 = Lnu_transp[0]
                Lnu_z0_log = np.log10(Lnu_z0,
                                       where=(Lnu_z0 > 0),
                                       out=(np.zeros_like(Lnu_z0) - 1000.0))
                finterp_logU = interp1d(self.logU, Lnu_z0_log, axis=0, bounds_error=False, fill_value=-1000.0)
                # logU values for each model
                interp_vals = finterp_logU(logU)
                # interp_vals shape -> (Nmodels, Nages, Nwave)
                Lnu_obs = 10**interp_vals
            else:
                Lnu_obs = 10**interpn((self.Zmet, self.logU),
                                       np.log10(Lnu_transp,
                                                where=(Lnu_transp > 0),
                                                out=(np.zeros_like(Lnu_transp) - 1000.0)),
                                       params,
                                       method='linear')

        else:
            # We need only interpolate in the metallicity dimension.
            if np.asarray(self.Zmet).size == 1:
                # Single-metallicity grid: take the only metallicity and broadcast to models
                base = np.take(self.Lnu_obs, 0, axis=1)  # shape (Nages, Nwave)
                # Broadcast to (Nages, Nmodels, Nwave)
                Lnu_obs = np.repeat(base[:, None, :], Nmodels, axis=1)
                # Now swap axes to (Nmodels, Nages, Nwave)
                Lnu_obs = np.swapaxes(Lnu_obs, 0, 1)
            else:
                finterp = interp1d(self.Zmet, np.log10(self.Lnu_obs), axis=1)
                Lnu_obs = 10**finterp(params.flatten())
                #print('Shape after interpolation: ', Lnu_obs.shape)
                Lnu_obs = np.swapaxes(Lnu_obs, 0, 1)

        # It is sometimes useful to have the spectra evaluated at each stellar age
        if stepwise:

            ages_lnu_unattenuated = sfh.multiply(sfh_param, Lnu_obs)
            ages_lnu_attenuated = ages_lnu_unattenuated.copy()
            ages_lnu_attenuated = ages_lnu_attenuated * exptau[:,None,:]
            ages_L_TIR = np.abs(trapz(ages_lnu_unattenuated - ages_lnu_attenuated, self.nu_grid_obs, axis=2))

            if (Nmodels == 1):
                ages_lnu_unattenuated = ages_lnu_unattenuated.reshape(self.Nages,-1)
                ages_lnu_attenuated = ages_lnu_attenuated.reshape(self.Nages,-1)
                ages_L_TIR = ages_L_TIR.flatten()

            return ages_lnu_attenuated, ages_lnu_unattenuated, ages_L_TIR

        else:
            # We distinguish between piecewise and continuous SFHs here
            if (self.step):
                lnu_unattenuated = sfh.sum(sfh_param, Lnu_obs)
            else:
                lnu_unattenuated = sfh.integrate(sfh_param, Lnu_obs)

            lnu_attenuated = lnu_unattenuated.copy()
            lnu_attenuated = lnu_unattenuated * exptau
            L_TIR = np.abs(trapz(lnu_unattenuated - lnu_attenuated, self.nu_grid_obs, axis=1))

            if (Nmodels == 1):
                lnu_unattenuated = lnu_unattenuated.flatten()
                lnu_attenuated = lnu_attenuated.flatten()
                L_TIR = L_TIR.flatten()

            return lnu_attenuated, lnu_unattenuated, L_TIR


    def get_model_lnu(self, sfh, sfh_param, params, exptau=None, exptau_youngest=None, stepwise=False):
        '''Construct the stellar SED as observed in the given filters.

        Given a SFH instance and set of parameters, the corresponding high-resolution spectrum
        is constructed and convolved with the filters. Optionally, attenuation is applied and
        the attenuated power is returned.

        Parameters
        ----------
        sfh : instance of lightning.sfh.PiecewiseConstSFH or lightning.sfh.FunctionalSFH
            Star formation history model.
        sfh_params : np.ndarray, (Nmodels, Nparam) or (Nparam,), float32
            Parameters for the star formation history.
        params : np.ndarray, (Nmodels, 1) or (Nmodels,)
            Values for logU, if the model includes a nebular component.
        exptau : np.ndarray, (Nmodels, Nwave) or (Nwave,), float32
            ``exp(-tau)`` as a function of wavelength. If this is 2D, the
            size of the first dimension must match the size of the first
            dimension of ``sfh_params``.
        exptau_youngest : np.ndarray, (Nmodels, Nwave) or (Nwave,), float32
            This doesn't do anything at the moment, until I figure out how
            to flexibly decide which ages to apply the birth cloud attenuation to.
        stepwise : bool
            If true, the spectrum is returned as a function of stellar age.

        Returns
        -------
        lnu_attenuated : np.ndarray, (Nmodels, Nfilters), (Nmodels, Nages, Nfilters), or (Nfilters,), float32
            The stellar spectrum as seen after the application of the ISM dust
            attenuation model.
        lnu_unattenuated : np.ndarray, (Nmodels, Nfilters), (Nmodels, Nages, Nfilters), or (Nfilters,), float32
            The intrinsic stellar spectrum.
        L_TIR : np.ndarray, (Nmodels,) or (Nmodels, Nages)
            The total attenuated power of the stellar population.

        '''

        if (len(sfh_param.shape) == 1):
            sfh_param = sfh_param.reshape(1, -1)


        if (self.step):
            assert (sfh.type == 'piecewise'), 'Binned stellar populations require a piecewise-defined SFH.'

        Nmodels = sfh_param.shape[0]

        if (self.nebular):
            assert (params is not None), 'BPASS models with the Cloudy component enabled require logU to be specified.'
            assert (params.shape[0] == Nmodels), 'First dimension of logU array must match first dimension of SFH.'

        if (exptau is None):
            exptau = np.full((Nmodels,len(self.wave_grid_rest)),1)
        else:
            assert (exptau.shape[0] == Nmodels), 'First dimension of exptau must match first dimension of SFH.'


        if (exptau_youngest is None):
            exptau_youngest = exptau
        else:
            assert (exptau.shape[0] == Nmodels), 'First dimension of exptau_youngest must match first dimension of SFH.'

        # It is sometimes useful to have the spectra evaluated at each stellar age
        if stepwise:

            ages_lnu_attenuated, ages_lnu_unattenuated, ages_L_TIR = self.get_model_lnu_hires(sfh, sfh_param, params=params, exptau=exptau, exptau_youngest=exptau_youngest, stepwise=True)

            if (Nmodels == 1):
                ages_lnu_attenuated = ages_lnu_attenuated.reshape(1, self.Nages, -1)
                ages_lnu_unattenuated = ages_lnu_unattenuated.reshape(1, self.Nages, -1)

            # Integrate steps_lnu_unattenuated - steps_lnu_attenuated to get the dust luminosity per bin
            # Comes out negative since self.nu_grid_obs is monotonically decreasing.

            ages_lmod_attenuated = np.zeros((Nmodels, self.Nages, self.Nfilters))
            ages_lmod_unattenuated = np.zeros((Nmodels, self.Nages, self.Nfilters))

            for i, filter_label in enumerate(self.filters):
                # Recall that the filters are normalized to 1 when integrated against wave_grid_obs
                # so we can integrate with that to get the mean Lnu in each band.
                ages_lmod_attenuated[:,:,i] = trapz(self.filters[filter_label][None,None,:] * ages_lnu_attenuated, self.wave_grid_obs, axis=2)
                ages_lmod_unattenuated[:,:,i] = trapz(self.filters[filter_label][None,None,:] * ages_lnu_unattenuated, self.wave_grid_obs, axis=2)

            if (Nmodels == 1):
                ages_lmod_attenuated = ages_lmod_attenuated.flatten()
                ages_lmod_unattenuated = ages_lmod_unattenuated.flatten()
                ages_L_TIR = ages_L_TIR.flatten()

            return ages_lmod_attenuated, ages_lmod_unattenuated, ages_L_TIR
        else:

            lnu_attenuated, lnu_unattenuated, L_TIR = self.get_model_lnu_hires(sfh, sfh_param, params, exptau=exptau, exptau_youngest=exptau_youngest, stepwise=False)

            if (Nmodels == 1):
                lnu_attenuated = lnu_attenuated.reshape(1,-1)
                lnu_unattenuated = lnu_unattenuated.reshape(1,-1)

            lmod_attenuated = np.zeros((Nmodels, self.Nfilters))
            lmod_unattenuated = np.zeros((Nmodels, self.Nfilters))

            for i, filter_label in enumerate(self.filters):
                # Recall that the filters are normalized to 1 when integrated against wave_grid_obs
                # so we can integrate with that to get the mean Lnu in each band.
                lmod_attenuated[:,i] = trapz(self.filters[filter_label][None,:] * lnu_attenuated, self.wave_grid_obs, axis=1)
                lmod_unattenuated[:,i] = trapz(self.filters[filter_label][None,:] * lnu_unattenuated, self.wave_grid_obs, axis=1)

            if (Nmodels == 1):
                lmod_attenuated = lmod_attenuated.flatten()
                lmod_unattenuated = lmod_unattenuated.flatten()
                L_TIR = L_TIR.flatten()


            return lmod_attenuated, lmod_unattenuated, L_TIR


    def get_model_lines(self, sfh, sfh_param, params, stepwise=False):
        '''Get the integrated luminosity of all the lines available to the nebular model.

        See self.line_names for a full list of lines. In the future we'll need to redden these lines
        to compare them to the observed lines, such that our line attenuation is consistent with
        the attenuation of the stellar population model broadly.

        Parameters
        ----------
        sfh : instance of lightning.sfh.PiecewiseConstSFH or lightning.sfh.FunctionalSFH
            Star formation history model.
        sfh_params : np.ndarray, (Nmodels, Nparam) or (Nparam,), float32
            Parameters for the star formation history.
        params : np.ndarray, (Nmodels, 2)
            Values for Z and logU.
        stepwise : bool
            If true, the lines are returned as a function of stellar age.

        Returns
        -------
        Lmod_lines :  np.ndarray, (Nmodels, Nlines) or (Nmodels, Nages, Nlines)
            Integrated line luminosities, optionally as a function of age.

        '''

        if (len(sfh_param.shape) == 1):
            sfh_param = sfh_param.reshape(1, -1)


        if (self.step):
            assert (sfh.type == 'piecewise'), 'Binned stellar populations require a piecewise-defined SFH.'

        Nmodels = sfh_param.shape[0]

        assert (self.nebular), 'Models were created without nebular emission; there are no lines.'
        assert (params.shape[0] == Nmodels), 'First dimension of logU array must match first dimension of SFH.'

        #print(self.line_lum)
        lines_transp = np.transpose(self.line_lum, axes=[1,2,0,3])
        # Handle case where metallicity grid has a single point: interpolate only in logU
        if np.asarray(self.Zmet).size == 1:
            lines_z0 = lines_transp[0]
            lines_z0_log = np.log10(lines_z0,
                                     where=(lines_z0 > 0),
                                     out=(np.zeros_like(lines_z0) - 1000.0))
            finterp_logU = interp1d(self.logU, lines_z0_log, axis=0, bounds_error=False, fill_value=-1000.0)
            L_lines = 10**finterp_logU(params[:,1])
        else:
            L_lines = 10**interpn((self.Zmet, self.logU),
                                 np.log10(lines_transp,
                                          where=(lines_transp > 0),
                                          out=(np.zeros_like(lines_transp) - 1000.0)),
                                 params,
                                 method='linear')

        L_lines[np.isnan(L_lines)] = 0

        # finterp = interp1d(self.logU, self.line_lum, axis=1)
        # L_lines = finterp(params.flatten()) # (Nages, Nmodels, Nlines)
        #L_lines = np.swapaxes(L_lines, 0, 1) # (Nmodels, Nages, Nlines)

        if stepwise:
            ages_Lmod_lines = sfh.multiply(sfh_param, L_lines)

            return ages_Lmod_lines

        else:
            if (self.step):
                Lmod_lines = sfh.sum(sfh_param, L_lines)
            else:
                Lmod_lines = sfh.integrate(sfh_param, L_lines)

            return Lmod_lines

