import pyrat.fileutils.gpri_files as gpf


def range_compression(input, output, threads, config, params, wildcards):
    raw = gpf.rawData(input.raw_par, input.raw)
    slc = gpf.range_compression(raw, rmin=params.rmin, rmax=params.rmax,
                                kbeta=params.k, dec=1, zero=params.z,
                                rvp_corr=False, scale=True)
    slc.tofile(output.slc_par, output.slc)

    

range_compression(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
    snakemake.wildcards)