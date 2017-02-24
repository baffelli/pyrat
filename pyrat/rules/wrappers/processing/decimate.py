import pyrat.fileutils.gpri_files as gpf
import pyrat.gpri_utils.calibration as cal
from snakemake import shell

def decimate(input, output, threads, config, params, wildcards):
    slc_input = gpf.gammaDataset(input.slc_par, input.slc)
    slc_dec = slc_input.decimate(params.dec, mode=params.dec_mode)
    slc_dec.tofile(output.slc_par, output.slc)


decimate(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
         snakemake.wildcards)