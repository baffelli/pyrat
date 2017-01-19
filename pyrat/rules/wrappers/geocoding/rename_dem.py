import pyrat.fileutils.gpri_files as gpf
import pyrat.gpri_utils.calibration as cal
from snakemake import shell
import gdal
import pyrat.geo.geofun as geo

def rename_dem(input, output, threads, config, params, wildcards):
    shell("cp {input.dem} {output.dem}")
    shell("cp {input.dem_par} {output.dem_par}")

rename_dem(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
    snakemake.wildcards)