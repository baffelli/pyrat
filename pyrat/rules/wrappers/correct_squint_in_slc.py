import pyrat.fileutils.gpri_files as gpf


def correct_squint_in_slc(input, output, threads, config, params, wildcards):
    slc = gpf.gammaDataset(input.slc_par, input.slc)
    raw_desq = gpf.correct_squint_in_SLC(slc, squint_rate=params.squint_rate)
    raw_desq.tofile(output.corr_par, output.corr)

    

correct_squint_in_slc(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
    snakemake.wildcards)