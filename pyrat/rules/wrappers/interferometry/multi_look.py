import pyrat.fileutils.gpri_files as gpf
import pyrat.gpri_utils.calibration as cal
from snakemake import shell

def multi_look(input, output, threads, config, params, wildcards):
    shell("multi_look {input.slc} {input.slc_par} {output.mli} {output.mli_par} {params.rlks} {params.azlks}")


multi_look(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
    snakemake.wildcards)