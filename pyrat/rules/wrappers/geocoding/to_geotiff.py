from snakemake import shell

import pyrat.fileutils.gpri_files as gpf

import pyrat.geo.geofun as geo

def to_geotiff(input, output, threads, config, params, wildcards):
    dt_code = gpf.gt_mapping_from_extension(input.file)
    gt_cmd = "data2geotiff {{input.dem_seg_par}} {{input.file}} {dt_code}  {{output.gt}}".format(dt_code=dt_code)
    shell(gt_cmd)

to_geotiff(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
                 snakemake.wildcards)
