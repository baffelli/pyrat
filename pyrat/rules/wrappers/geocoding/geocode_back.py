from snakemake import shell

import pyrat.fileutils.gpri_files as gpf


def geocode_back(input, output, threads, config, params, wildcards):
    dem_width = gpf.get_width(input.dem_seg_par)
    data_width = gpf.get_width(input.reference_mli_par)
    dem_par_dict = gpf.par_to_dict(input.dem_seg_par)
    par_dict = gpf.par_to_dict(input.reference_mli_par)
    nlines = dem_par_dict['nlines']
    filetype = gpf.gamma_datatype_code_from_extension(input.data)
    cmd = "geocode_back  {input.data} {data_width} {input.lut} " \
          "{output.geocoded_file} {out_width} 0 1 {dtype} - - - 10".format(data_width=data_width,
    out_width=dem_width, nlines=nlines, dtype=filetype, output=output, input=input)
    shell(cmd)


geocode_back(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
                snakemake.wildcards)
