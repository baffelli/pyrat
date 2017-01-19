import pyrat.fileutils.gpri_files as gpf


def correct_squint(input, output, threads, config, params, wildcards):
    raw = gpf.rawData(input.raw_par, input.raw)
    raw_desq = gpf.correct_squint(raw, squint_rate=params.squint_rate)
    raw_desq.tofile(output.corr_par, output.corr)

    

correct_squint(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
    snakemake.wildcards)