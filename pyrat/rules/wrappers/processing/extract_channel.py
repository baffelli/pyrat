import pyrat.fileutils.gpri_files as gpf


def extract_channel(input, output, threads, config, params, wildcards):
    raw = gpf.rawData(input.raw_par, input.raw, channel_mapping=config['channel_mapping'])
    print(wildcards.chan)
    raw_chan = raw.extract_channel(wildcards.chan[0:3], wildcards.chan[-1])
    raw_chan.tofile(output.chan_par, output.chan)



extract_channel(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
    snakemake.wildcards)