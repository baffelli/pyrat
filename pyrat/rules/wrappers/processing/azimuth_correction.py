import pyrat.fileutils.gpri_files as gpf
import pyrat.gpri_utils.calibration as cal

def azimuth_correction(input, output, threads, config, params, wildcards):
    slc_input = gpf.gammaDataset(input.slc_par, input.slc)
    slc_corr = cal.azimuth_correction(slc_input, params.r_ph, ws=params.integration_length)
    slc_corr.tofile(output.slc_par, output.slc)

    

azimuth_correction(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
    snakemake.wildcards)