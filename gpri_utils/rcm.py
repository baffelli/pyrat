import pyrat.fileutils.gpri_files as gpf
import numpy as np
import matplotlib.pyplot as plt
import pyrat.visualization.visfun as vf
import scipy.ndimage as nd
import pyrat.gpri_utils.calibration as cal


slc_VV = '/data/HIL/20160224/rvp.slc'
slc_VV_par = '/data/HIL/20160224/rvp.slc.par'
# slc_HH = '/data/HIL/20160224/slc_desq/20160224_140626_AAAl.slc'


data_VV, par = gpf.load_dataset(slc_VV_par, slc_VV)

def azspec(image):
    image_hat = np.fft.fftshift(np.fft.fft(image,axis=1),axes=(1))
    return image_hat

VV_hat = azspec(data_VV)

#
# RGB, pal, map = vf.dismph(VV_hat, k=0.2)

#
# #Position of reflector of interest
#
#
# def filtered_spectrum(data, ref_pos,r_win, az_win):
#     r_filt = np.hamming(r_win)
#     az_filt = np.hamming(az_win)
#     filt_2d = np.sqrt(np.outer(r_win, az_win))
#     data_filt = data[(ref_pos[0] - r_win/2):(ref_pos[0] + r_win/2),
#                 (ref_pos[1] - az_win/2):(ref_pos[1] + az_win/2)] * filt_2d
#     data_filt_hat = np.fft.fftshift(np.fft.fft2(data_filt,s=(6250,4096),),(0,1))
#     return  data_filt_hat
#
#
# ref_pos = (521, 4437)
# r_win = 1000
# az_win = 90
# HH_azspec = filtered_spectrum(data_HH, ref_pos, r_win, az_win)
# VV_azspec = filtered_spectrum(data_VV, ref_pos, r_win, az_win)
#
# k =  0.4
# R = vf.scale_array(np.abs(VV_azspec)**k)
# G = vf.scale_array(np.abs(HH_azspec)**k) * 0
# B = vf.scale_array(np.abs(HH_azspec)**k)
# plt.imshow(np.dstack((R,G,B)))
#
# fig, ax = plt.subplots()
# plt.subplot(2,1,1)
# plt.imshow(np.abs(VV_azspec))
# ax = plt.gca()
# plt.subplot(2,1,2, sharex=ax, sharey=ax)
# plt.imshow(np.abs(HH_azspec))
