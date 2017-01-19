import pyrat.fileutils.gpri_files as gpf
import pyrat.gpri_utils.calibration as cal
from snakemake import shell
import gdal
import pyrat.geo.geofun as geo

def convert_dem(input, output, threads, config, params, wildcards):
    # Open the data set
    DS = gdal.Open(input.dem)
    # Convert
    geo.geotif_to_dem(input.dem, output.gamma_dem_par, output.gamma_dem)

convert_dem(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
    snakemake.wildcards)