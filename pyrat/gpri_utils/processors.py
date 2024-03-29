from ..fileutils import gpri_files as _gpf
import numpy as np
import pyfftw.interfaces.numpy_fft as _fftp
from . import calibration as _cal



def process_undecimated_slc(slc, squint_rate, phase_center_shift, ws=0.6, decimation_factor=5, correct_azimuth=True, interp=_gpf.interpolation_1D):
    """
    This function processes an undecimated SLC to which no squint correction was applies. It is a combination of squint
    correction, azimut correction and decimation in a single step.
    Parameters
    ----------
    slc
    squint_rate
    phase_center_shift
    integration_length
    decimation_factor
    correct_azimuth

    Returns
    -------

    """
    slc_desq = _gpf.correct_squint_in_SLC(slc, squint_rate=squint_rate, interp=interp).astype(slc.dtype)
    #Prepare filter
    ws_samp = ws // slc.GPRI_az_angle_step
    theta = np.arange(-ws_samp // 2, ws_samp // 2) * np.deg2rad(slc.GPRI_az_angle_step)
    #Convert into grid
    r = slc.r_vec
    rr, tt = np.meshgrid(r, theta, indexing='ij')
    #Compute antenna phase center lever arm
    r_ant = np.linalg.norm(slc.phase_center[0:2])
    lam = _gpf.lam( slc.radar_frequency)
    filt2d, dist2d = _cal.distance_from_phase_center(r_ant, phase_center_shift, rr, tt, lam, wrap=False)
    matched_filter2d = np.exp(-1j*filt2d)
    filt_hat = matched_filter2d
    #Decimate and filter
    arr_dec = np.zeros((slc.shape[0], int(slc_desq.shape[1]//decimation_factor)), dtype=np.complex64)
    #Pad to filter length
    for idx_az in range(arr_dec.shape[1]):
        #start of convolution window
        start_idx = int(idx_az * decimation_factor - np.floor(ws_samp/2))
        stop_idx = int(idx_az * decimation_factor + np.ceil(ws_samp/2))
        #Padding left and right
        pad_left = 0 if start_idx > 0 else -start_idx
        pad_right = 0 if stop_idx < slc_desq.shape[1] else stop_idx - slc_desq.shape[1]
        #Indices to extract
        current_idx = slice(np.clip(start_idx,0,slc_desq.shape[1]) , np.clip(stop_idx,0,slc_desq.shape[1]))
        slc_pad = np.pad(slc_desq[:, current_idx],((0,0) ,(int(pad_left), int(pad_right))), mode='constant')
        if correct_azimuth:
            pass
        else:
            filt_hat = np.ones(slc_pad.shape[1])
        arr_dec[:, idx_az] = np.mean(slc_pad[:,::-1] * filt_hat[::], axis=1)
    # arr_dec = fftp.fft(arr,axis=0)[0:slc.shape[0]] * shift[:,None]
    # slc_corr = fftp.rfft(arr_dec, axis=0) *
    # rgb, *rest = vf.dismph(slc_corr, k=0.5, sf=0.2)
    # plt.imshow(rgb)
    # plt.show()
    slc_corr = slc.__array_wrap__(arr_dec)
    slc_corr.GPRI_az_angle_step = decimation_factor * slc_corr.GPRI_az_angle_step
    slc_corr.azimuth_line_time = decimation_factor * slc_corr.azimuth_line_time
    slc_corr.prf = slc_corr.prf / decimation_factor
    slc_corr._params.add_parameter('GPRI_decimation_factor', decimation_factor)
    # else:
    #     arr_dec.GPRI_decimation_factor = dec
    slc_corr.azimuth_lines = arr_dec.shape[1]

    return slc_corr