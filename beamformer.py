#!/usr/bin/env python3

import math
import cmath
import numpy as np
from numba import jit, cuda, prange
from numba.cuda.cudadrv.error import CudaSupportError
import matplotlib.pyplot as plt
from tqdm import tqdm

from tools import timer


class BeamformerGPU(object):

    def __init__(self, ha0, dec0, dHACOSDEC, dDEC):
        # define constants
        self.ntab = 12
        self.nfreq = 1536
        self.ndish = 8
        self.bcq = 144

        # select a GPU to run on
        self.gpu = cuda.get_current_device()

        # construct frequency axis for Apertif data: 1220 to 1520 MHz
        flo = 1220.  # MHz
        df = 300. / self.nfreq  # MHz
        self.freqs = np.arange(self.nfreq, dtype=np.float32) * df + flo
        # convert to wavelength using wavelength = c / frequency
        self.lamb = 299792456. * 1e-6 / self.freqs
        # transfer to GPU
        self.d_lamb = cuda.to_device(self.lamb)

        # construct array of (equidistant) dish positions
        self.dish_pos = np.arange(self.ndish) * self.bcq
        # transfer to GPU
        self.d_dish_pos = cuda.to_device(self.dish_pos)

        # store phase center in radians
        self.ha0 = ha0 * np.pi / 180.
        self.dec0 = dec0 * np.pi / 180.

        # extract size of coordinate grid
        assert len(dHACOSDEC.shape) == len(dDEC.shape) == 2
        self.ndec, self.nha = dHACOSDEC.shape

        # transfer coordinate grid to GPU (in radians)
        self.d_dHACOSDEC = cuda.to_device(dHACOSDEC * np.pi / 180.)
        self.d_dDEC = cuda.to_device(dDEC * np.pi / 180.)

        # calculate TAB phase offsets and copy to GPU
        self.d_phase_tab = self.get_tab_phases()

        # create device array to hold geometric phase offset
        self.d_phase_geom = cuda.device_array((self.ndish, self.nfreq, self.ndec, self.nha), dtype=np.float32)

        # create device array for output TABs
        self.d_tabs = cuda.device_array((self.ntab, self.nfreq, self.ndec, self.nha), dtype=np.float32)

        # get number of threads and blocks for geomtric phase method and beamforming method
        # dimensions assumed to be integer multiple of warp size
        # nthread = (self.gpu.WARP_SIZE, self.gpu.WARP_SIZE, self.gpu.WARP_SIZE)
        # nblock = (int(self.nfreq / nthread[0]), int(self.ndec / nthread[1]), int(self.nha / nthread[2]))
        nthread = (self.gpu.WARP_SIZE, self.gpu.WARP_SIZE)
        nblock = (int(self.ndec / nthread[0]), int(self.nha / nthread[1]))

        # calculate geometric phases on GPU
        # can only jit static methods, so pass on all arguments here
        self.get_geom_phases[nblock, nthread](self.ha0, self.dec0, self.d_dish_pos, self.d_lamb, self.d_dHACOSDEC, self.d_dDEC, self.d_phase_geom)

        # beamform all TABs
        # for tab in tqdm(range(self.ntab)):
        for tab in range(self.ntab):
            self.beamform[nblock, nthread](self.d_phase_tab, self.d_phase_geom, self.d_tabs, self.ndish, self.nfreq, tab)

        # copy beamformed intensity beams to host
        self.tabs = self.d_tabs.copy_to_host()
        # scale by ndish squared to get global max of one
        self.tabs /= self.ndish**2

    def get_tab_phases(self):
        # calculate the phase offset of each dish, TAB
        phases = 2 * np.pi * np.arange(self.ndish, dtype=np.float32)[..., None] * \
            np.arange(self.ntab, dtype=np.float32) / float(self.ntab)
        # transfer to GPU
        return cuda.to_device(phases)

    @staticmethod
    @cuda.jit()
    def get_geom_phases(ha0, dec0, d_dish_pos, d_lamb, d_dHACOSDEC, d_dDEC, d_phase_geom):
        # wavelength, ha offset, dec offset
        # lamb_ind, dec_ind, ha_ind = cuda.grid(3)
        dec_ind, ha_ind = cuda.grid(2)

        # extract values
        # lamb = d_lamb[lamb_ind]
        # extract coordinates
        dhacosdec = d_dHACOSDEC[dec_ind, ha_ind]
        ddec = d_dDEC[dec_ind, ha_ind]

        # compute the geometrical phase offset and store to output array for each dish
        for dish_ind, dish in enumerate(d_dish_pos):
            for lamb_ind, lamb in enumerate(d_lamb):
                d_phase_geom[dish_ind, lamb_ind, dec_ind, ha_ind] = 2 * np.pi * dish / lamb * \
                    (math.sin(dec0) * math.sin(ha0) * ddec -
                     math.cos(dec0) * math.cos(ha0) * dhacosdec)

    @staticmethod
    @cuda.jit()
    def beamform(d_phase_tab, d_phase_geom, d_tabs, ndish, nfreq, tab):
        # wavelength, ha offset, dec offset
        # lamb_ind, dec_ind, ha_ind = cuda.grid(3)
        dec_ind, ha_ind = cuda.grid(2)

        for lamb_ind in range(nfreq):
            tmp = 0j
            for d in range(ndish):
                # extract tab phase
                phase_tab = d_phase_tab[d, tab]
                # extract geometrical phase
                phase_geom = d_phase_geom[d, lamb_ind, dec_ind, ha_ind]
                # # add to voltage beam (= tmp value)
                tmp += cmath.exp(1j * (phase_tab + phase_geom))
            # store intensity
            d_tabs[tab, lamb_ind, dec_ind, ha_ind] = abs(tmp)**2


if __name__ == '__main__':
    # verify GPU is available
    if not cuda.is_available():
        raise CudaSupportError("No CUDA-compatible device found")

    # initialize coordinate grid (in degrees)
    ha0 = 30.
    dec0 = 45.
    dhacosdec = np.linspace(-.5, .5, 128, dtype=np.float32)
    ddec = np.linspace(-.5, .5, 128, dtype=np.float32)
    dHACOSDEC, dDEC = np.meshgrid(dhacosdec, ddec)

    # run beamformer
    with timer('GPU'):
        bf_gpu = BeamformerGPU(ha0, dec0, dHACOSDEC, dDEC)
    # bf_gpu.tabs shape is (ntab, nfreq, ndec, nha)
    # select one dec
    bf_gpu.tabs = bf_gpu.tabs[:, :, 63]

    # plot a few tabs
    tabs_to_plot = [0, 3, 6, 9]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    X, Y = np.meshgrid(dhacosdec, bf_gpu.freqs)
    for i, tab in enumerate(tabs_to_plot):
        ax = axes[i]
        ax.pcolormesh(X, Y, bf_gpu.tabs[tab])
        ax.set_xlabel('dHA cos(dec) (deg)')
        ax.set_ylabel('Frequency (MHz)')
        ax.set_title('TAB{:02d}'.format(tab))
        ax.label_outer()

    plt.show()
