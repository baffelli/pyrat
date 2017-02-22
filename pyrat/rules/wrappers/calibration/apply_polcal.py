import pyrat.fileutils.parameters as par
import pyrat.core.matrices as mat
import pyrat.gpri_utils.calibration as cal
import pyrat.core.corefun as cf
import pyrat.fileutils.gpri_files as gpf
import numpy as np
import matplotlib.pyplot as plt
import pyrat.visualization.visfun as vf

def polcal(input, output, threads, config, params, wildcards):
    C_matrix_flat = mat.coherencyMatrix(params['C_input_root'], params['C_input_root'] + '.par', basis='lexicographic',
                                        gamma=True, bistatic=True)
    cal_dict = gpf.par_to_dict(input.cal_par)
    phi_t = cal_dict['phi_t']
    phi_r = cal_dict['phi_r']
    f = cal_dict['f']
    g = cal_dict['g']
    K = cal_dict['K']
    D = cal.distortion_matrix(phi_t, phi_r, f, g)
    D_inv = np.diag(1 / np.diag(D))
    C_cal = np.float(K) * C_matrix_flat.transform(D_inv, D_inv.T.conj())
    C_cal.tofile(params['C_output_root'], bistatic=True)

polcal(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
    snakemake.wildcards)