from snakemake import shell

import pyrat.fileutils.gpri_files as gpf


def geocode_forward(input, output, threads, config, params, wildcards):
    dem_width = gpf.get_width(input.dem_seg_par)
    data_width = gpf.get_width(input.reference_mli_par)
    dem_par_dict = gpf.par_to_dict(input.dem_seg_par)
    par_dict = gpf.par_to_dict(input.reference_mli_par)
    nlines = par_dict.azimuth_lines
    filetype = gpf.gamma_datatype_code_from_extension(input.data)
    cmd = "geocode {input.lut} {input.data} {out_width} {output.geocoded_file} {data_width} {nlines} 0 {dtype}".format(
        data_width=data_width,
        out_width=dem_width, nlines=nlines, dtype=filetype, input=input, output=output)
    shell(cmd)


geocode_forward(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
                snakemake.wildcards)
