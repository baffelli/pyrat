from snakemake import shell

import pyrat.fileutils.gpri_files as gpf

import pyrat.geo.geofun as geo

def shadow_map(input, output, threads, config, params, wildcards):
    u, inc_par = gpf.load_dataset(input.reference_mli_par, input.u)
    inc, inc_par = gpf.load_dataset(input.reference_mli_par, input.inc)
    sh_map = geo.shadow_map(u, inc)
    sh_map.T.astype(gpf.type_mapping['UCHAR']).tofile(output.sh_map)


shadow_map(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
                 snakemake.wildcards)
