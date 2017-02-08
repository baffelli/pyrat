import pyrat.fileutils.parameters as par
import pyrat.core.matrices as mat
import pyrat.gpri_utils.calibration as cal
import pyrat.core.corefun as cf
import pyrat.fileutils.gpri_files as gpf
import numpy as np
import matplotlib.pyplot as plt
import pyrat.visualization.visfun as vf

def polcal_par(input, output, threads, config, params, wildcards):
    C_matrix_flat = mat.coherencyMatrix(params['C_root'], params['C_par'], basis='lexicographic', gamma=True,
                                        bistatic=True)
    #        #The indices are in the raw coordinates, needs to decimate
    pt = []
    cal_ref = config['list_of_reflectors'][config['calibration']['reflector_index']]
    # for cal_ref in config['list_of_reflectors']:
    ridx = cal_ref['ridx']
    azidx = cal_ref['azidx'] // C_matrix_flat.GPRI_decimation_factor
    RCS = cal.cr_rcs(cal_ref['side'], C_matrix_flat.radar_frequency, type=cal_ref['type'])
    sw = tuple(params.sw)
    av_win = [5, 10]  # averaging window
    C_matrix_flat_av = C_matrix_flat.boxcar_filter(av_win)
    C_ptarg, rplot, azplot, mx_pos, ptarg_info, r_vec, az_vec = cf.ptarg(C_matrix_flat_av, ridx, azidx, sw=sw, azwin=50,
                                                                         rwin=20, )
    rgb, pal, rest = vf.dismph(C_ptarg[:,:,0,3])
    plt.imshow(rgb)
    plt.plot(mx_pos[1], mx_pos[0], 'bo')
    plt.show()
    C_cal = C_matrix_flat_av[mx_pos]
    # determine imbalance
    phi_t, phi_r, f, g = cal.measure_imbalance(C_ptarg[mx_pos[0], mx_pos[1]], C_matrix_flat_av)
    K = cal.gpri_radcal(C_matrix_flat[:, :, 0, 0], [ridx, azidx], RCS)
    cal_dict = {}
    cal_dict['phi_t'] = {'value': phi_t}
    cal_dict['phi_r'] = {'value': phi_r}
    cal_dict['f'] = {'value': f.real}
    cal_dict['g'] = {'value': g.real}
    cal_dict['K'] = {'value': K.real}
    cal_dict = par.ParameterFile(cal_dict)
    gpf.dict_to_par(cal_dict, output.cal_par)

polcal_par(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
    snakemake.wildcards)