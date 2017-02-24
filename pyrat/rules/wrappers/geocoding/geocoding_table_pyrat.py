from snakemake import shell

import pyrat.fileutils.gpri_files as gpf
from osgeo import gdal
import pyrat.geo.geofun as gf



def geocoding_table(input, output, threads, config, params, wildcards):
    #Load dem
    dem = gdal.Open(input.dem)
    mli = gpf.gammaDataset(input.reference_mli_par, input.reference_mli)
    gc_map = gf.gc_map(dem, mli, heading=params.scan_heading)


geocoding_table(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
                snakemake.wildcards)
