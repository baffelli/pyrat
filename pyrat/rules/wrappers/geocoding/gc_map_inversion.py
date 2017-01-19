from snakemake import shell

import pyrat.fileutils.gpri_files as gpf


def gc_map_inversion(input, output, threads, config, params, wildcards):
    dem_width = gpf.get_width(input.dem_seg_par)
    data_width = gpf.get_width(input.reference_mli_par)
    par_dict = gpf.par_to_dict(input.reference_mli_par)
    nlines = par_dict['azimuth_lines']
    cmd = "gc_map_inversion  {input.lut} {dem_width} {output.inverse_lut} {data_width} {nlines} 0".format(
        data_width=data_width,
        dem_width=dem_width, nlines=nlines, input=input, output=output)
    shell(cmd)


gc_map_inversion(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
                 snakemake.wildcards)
