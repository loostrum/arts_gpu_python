#!/usr/bin/env python3

import math

import numpy as np
from numba import jit, cuda, prange
from numba.cuda.cudadrv.error import CudaSupportError
import matplotlib.pyplot as plt

from tools import timer


# Dispersion constant, assuming freq in MHz, time in sec, DM in pc cm^-3
K_DM = 4.15E3
# Sampling time (s)
TSAMP = 81.92E-6


@jit(nopython=True)
def get_shift(dm, f, f_ref):
    """
    Calculate integer sample shift for given DM
    :param dm: dispersion measure (pc / cc)
    :param f: frequency (MHz)
    :param f_ref: reference frequency (MHz)
    :param tsamp: sampling time (s)
    :return:
    """
    # Dispersion constant, assuming freq in MHz, time in sec, DM in pc cm^-3
    # K_DM = 4.15E3
    # Sampling time (s)
    # TSAMP = 81.92E-6

    # note: numba does not support np.round
    shift = K_DM * dm * (f ** -2 - f_ref ** -2) / TSAMP
    return shift


@cuda.jit('int32(float32, float32, float32)',
          device=True)
def get_shift_gpu(dm, f, f_ref):
    """
    Calculate integer sample shift for given DM
    :param dm: dispersion measure (pc / cc)
    :param f: frequency (MHz)
    :param f_ref: reference frequency (MHz)
    :param tsamp: sampling time (s)
    :return:
    """
    #: Dispersion constant, assuming freq in MHz, time in sec, DM in pc cm^-3
    # K_DM = 4.15E3
    #: Sampling time (s)
    # TSAMP = 81.92E-6

    # note: numba does not support np.round
    shift = K_DM * dm * (f ** -2 - f_ref ** -2) / TSAMP
    return int(shift)


def dedisp_cpu(data, dms, freqs):
    """
    Dedisperse to a range of DMs using CPU
    :param data: (nfreq, ntime) data
    :param dms: array of DMs to dedisperse to
    :param freqs: Frequency of each channel
    :return: DM-time data
    """
    # data.shape = nfreq, ntime
    # output shape = ndm, ntime
    nfreq, ntime = data.shape
    ndm = len(dms)
    ref_freq = freqs.max()

    # init output
    output = np.zeros((ndm, ntime))
    for i in prange(len(dms)):
        dm = dms[i]
        shifts = get_shift(dm, freqs, ref_freq)
        for n in range(nfreq):
            output[i] += np.roll(data[n], -int(shifts[n]))
    return output


dedisp_cpu_par = jit(dedisp_cpu, parallel=True)


@cuda.jit()
def dedisp_gpu(data, dms, freqs, ref_freq, output):
    # data = freq, time
    # output = dm, time
    # threadblocks = dm, time

    dm_ind, time_ind = cuda.grid(2)
    ndm, ntime = output.shape

    if dm_ind >= ndm or time_ind >= ntime:
        return

    dm = dms[dm_ind]

    tmp = 0
    for freq_ind, freq in enumerate(freqs):
        # get DM delay of this freq and DM
        shift = get_shift_gpu(dm, freq, ref_freq)
        # time_ind is output index, add shift to get input index
        # achieve wrap-around through modulo
        input_ind = (time_ind + shift) % ntime
        tmp += data[freq_ind, input_ind]

    output[dm_ind, time_ind] = tmp


if __name__ == '__main__':
    # verify GPU is available
    if not cuda.is_available():
        raise CudaSupportError("No CUDA-compatible device found")

    # Load FRB 191108 data, rounded to nearest int as used for real raw data
    data = np.round(np.load('FRB191108_DM588.13.npy')).astype(np.int32)
    # data are already dedispersed to value in filename, which is also the optimal value
    dm = 588.13
    # data shape is frequency, time
    nfreq, ntime = data.shape

    # construct frequency axis for Apertif data: 1220 to 1520 MHz
    flo = 1220.  # MHz
    df = 300. / nfreq  # MHz
    freqs = np.arange(nfreq, dtype=np.float32) * df + flo

    # reference frequency for dedispersion, use highest frequency
    ref_freq = freqs.max()

    # construct dm range to search around known value
    dm_offsets = np.arange(-16, 16, .1, dtype=np.float32)
    ndm = len(dm_offsets)

    print("CUDA info:")
    cuda.detect()
    print("DMs: {}".format(ndm))
    print("Freqs: {}".format(nfreq))
    print("Times: {}".format(ntime))

    # serial CPU. CPU is assumed to be right
    with timer('CPU'):
        truth = dedisp_cpu(data, dm_offsets, freqs)

    # Parallelized CPU.
    with timer('CPU parallel'):
        cpu_par = dedisp_cpu_par(data, dm_offsets, freqs)
    assert np.allclose(cpu_par, truth, atol=1E-6)

    # GPU
    gpu = cuda.get_current_device()
    nthread = (gpu.WARP_SIZE, gpu.WARP_SIZE)
    nx = math.ceil(ndm / nthread[0])
    ny = math.ceil(ntime / nthread[1])
    nblock = (nx, ny)

    output_gpu = np.empty((ndm, ntime), dtype=np.float32)

    with timer('GPU'):
        dedisp_gpu[nblock, nthread](data, dm_offsets, freqs, ref_freq, output_gpu)
    assert np.allclose(output_gpu, truth, atol=1E-6)

    # GPU without copies of data arrays
    ddata = cuda.to_device(data)
    doutput = cuda.device_array(shape=(ndm, ntime), dtype=np.float32)

    with timer('GPU without host <-> device copies'):
        dedisp_gpu[nblock, nthread](ddata, dm_offsets, freqs, ref_freq, doutput)
    output_gpu_nocopy = doutput.copy_to_host()
    assert np.allclose(output_gpu_nocopy, truth)

    # plot output DM-time arrays
    times = np.arange(ntime) * TSAMP
    X, Y = np.meshgrid(times, dm + dm_offsets)
    fig, axes = plt.subplots(ncols=4, sharex=True, sharey=True)
    axes = axes.flatten()

    kwargs = {'vmin': truth.min(), 'vmax': truth.max()}

    ax = axes[0]
    ax.pcolormesh(X, Y, truth, **kwargs)
    ax.set_title('CPU')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'DM (pc $cm^{-3}$)')

    ax = axes[1]
    ax.pcolormesh(X, Y, cpu_par, **kwargs)
    ax.set_title('CPU parallel')
    ax.set_xlabel('Time (s)')

    ax = axes[2]
    ax.pcolormesh(X, Y, output_gpu, **kwargs)
    ax.set_title('GPU')
    ax.set_xlabel('Time (s)')

    ax = axes[3]
    ax.pcolormesh(X, Y, output_gpu_nocopy, **kwargs)
    ax.set_title('GPU no copy')
    ax.set_xlabel('Time (s)')

    plt.show()
