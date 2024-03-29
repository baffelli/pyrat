from osgeo import ogr
from osgeo import gdal
from pyrat.geo import geofun
from osgeo import gdalnumeric

def segment_mask(inputs, outputs, threads, config, params, wildcards):
    ds = gdal.Open(inputs.mask)
    mask_seg = geofun.clip_dataset(ds, inputs.dem_seg_par)
    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.CreateCopy(outputs.mask, mask_seg, 1)
    gdalnumeric.SaveArray(dst_ds.ReadAsArray(), outputs.mask_bmp, format='BMP')


segment_mask(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
              snakemake.wildcards)