from snakemake import shell

import pyrat.fileutils.gpri_files as gpf


def geocoding_table(input, output, threads, config, params, wildcards):
    # Set scan heading
    ref_mli_par = gpf.par_to_dict(input.reference_mli_par)
    ref_mli_par.GPRI_scan_heading = params.scan_heading
    gpf.dict_to_par(ref_mli_par, input.reference_mli_par)
    gc_cmd = "gc_GPRI_map {ref_mli} {input.dem_par} {input.dem} " \
             "{output.dem_seg_par} {output.dem_seg} {output.lut} " \
             "{params.lat_ovr} {params.lon_ovr} {output.sim_sar} " \
             "{output.lv_theta} {output.lv_phi} {output.u} {output.v} " \
             "{output.inc} {output.psi} {output.pix} {output.ls_map} " \
             "{params.frame}"
    shell(gc_cmd.format(ref_mli=input.reference_mli_par,params=params, output=output, input=input))
    # Geocode forwards
    dem_width = gpf.par_to_dict(output.dem_seg_par).width
    data_width = ref_mli_par['range_samples']
    data_nlines = ref_mli_par['azimuth_lines']
    fwd_cmd = "geocode {output.lut} {output.sim_sar} {width_in} {output.sim_sar_azd} " \
              "{data_width} {data_nlines} 0 0".format(input=input, output=output,
                                                      width_in=dem_width,
                                                      data_width=data_width, data_nlines=data_nlines)
    shell(fwd_cmd)
    #Table refinement starts here
    #Offset parameters
    diff_par_cmd = "create_diff_par {input.reference_mli_par} - " \
                   "{params.diff_par} 1 0".format(input=input, params=params)
    shell(diff_par_cmd)
    #Compute offset
    off_cmd = "init_offsetm {input.reference_mli} {output.sim_sar_azd} " \
              "{params.diff_par} 1 1 - - - - - 256 1".format(
        output=output, params=params, input=input)
    shell(off_cmd)
    off_cmd = "offset_pwrm {input.reference_mli} {output.sim_sar_azd} " \
              "{params.diff_par} {params.offs} " \
              "{params.ccp} 64 64 - - 5 5 - - 0 0 - ".format(
        output=output, params=params, input=input)
    shell(off_cmd)
    #Improve estimation with LS fit
    fit_cmd = "offset_fitm {params.offs} {params.ccp} {params.diff_par} - - - 1 0".format(input=input,
                                                                                          params=params)
    shell(fit_cmd)
    #Load offset paramters
    off_par = gpf.par_to_dict(params.diff_par)
    #Shift in azimuth by the amount specified by the offset
    ref_mli_par.GPRI_scan_heading += off_par.azimuth_offset_polynomial[0] * ref_mli_par.GPRI_az_angle_step
    #Range shift
    r_shift = off_par.range_offset_polynomial[0] * ref_mli_par.range_pixel_spacing
    #Shift in range
    ref_mli_par.near_range_slc += r_shift
    ref_mli_par.center_range_slc += r_shift
    ref_mli_par.far_range_slc += r_shift
    gpf.dict_to_par(ref_mli_par, params.temp_mli_par)
    #Rerun geocoding
    shell(gc_cmd.format(ref_mli=params.temp_mli_par,params=params, output=output, input=input))



geocoding_table(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
                snakemake.wildcards)
