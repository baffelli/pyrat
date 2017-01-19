from snakemake import shell

import pyrat.fileutils.gpri_files as gpf


def geocoding_table(input, output, threads, config, params, wildcards):
    # Set scan heading
    ref_mli_par = gpf.par_to_dict(input.reference_mli_par)
    ref_mli_par.GPRI_scan_heading = params.scan_heading
    gpf.dict_to_par(ref_mli_par, input.reference_mli_par)
    gc_cmd = "gc_GPRI_map {input.reference_mli_par} {input.dem_par} {input.dem} " \
             "{output.dem_seg_par} {output.dem_seg} {output.lut} " \
             "{params.lat_ovr} {params.lon_ovr} {output.sim_sar} " \
             "{output.lv_theta} {output.lv_phi} {output.u} {output.v} " \
             "{output.inc} {output.psi} {output.pix} {output.ls_map} " \
             "{params.frame}".format(params=params, output=output, input=input)
    shell(gc_cmd)
    # Geocode forwards
    dem_width = gpf.par_to_dict(output.dem_seg_par).width
    data_width = ref_mli_par['range_samples']
    data_nlines = ref_mli_par['azimuth_lines']
    fwd_cmd = "geocode {output.lut} {output.sim_sar} {width_in} {output.sim_sar_azd} " \
              "{data_width} {data_nlines} 0 0".format(input=input, output=output,
                                                      width_in=dem_width,
                                                      data_width=data_width, data_nlines=data_nlines)
    shell(fwd_cmd)
    if config['geocoding'].get('refine_table'):
        diff_par_cmd = "create_diff_par {input.reference_mli_par} - " \
                       "{params.diff_par} 1 0".format(input=input, params=params)
        shell(diff_par_cmd)
        off_cmd = "init_offsetm {input.reference_mli} {output.sim_sar_azd} " \
                  "{params.diff_par} 1 1 - - - - - 256 1".format(
            output=output, params=params, input=input)
        shell(off_cmd)
        off_cmd = "offset_pwrm {input.reference_mli} {output.sim_sar_azd} " \
                  "{params.diff_par} {params.offs} " \
                  "{params.ccp} 64 64 - - 5 5 - - 0 0 - ".format(
            output=output, params=params, input=input)
        shell(off_cmd)
        fit_cmd = "offset_fitm {params.offs} {params.ccp} {params.diff_par} - - - 1 0".format(input=input,
                                                                                              params=params)
        shell(fit_cmd)
        fine_cmd = "gc_map_fine {output.lut} {wd} " \
                   "{params.diff_par} {params.lut_fine} 0".format(wd=dem_width,
                                                                  params=params, output=output)
        shell(fine_cmd)
        # Finally, we copy the improved LUT
        shell("cp {params.lut_fine} {output.lut}".format(output=output, params=params))
        # And now we transform the remaining parameters
        for dt in [output.lv_theta, output.lv_phi, output.u, output.v, output.psi]:
            cmd_interp = "interp_real {dt} {params.diff_par} temp - - 1".format(dt=dt, params=params)
            shell(cmd_interp)
            shell("cp temp {dt}".format(dt=dt))


geocoding_table(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
                snakemake.wildcards)
