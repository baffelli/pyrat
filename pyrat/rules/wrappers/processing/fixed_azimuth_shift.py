import pyrat.fileutils.gpri_files as gpf

from snakemake import shell


def fixed_shift(input, output, threads, config, params, wildcards):
    #Create par
    diff_cmd = "create_offset {input.slc_par} {input.slc_par} temp_shift 1 1 1 0".format(
        input=input, params=params, output=output)
    shell(diff_cmd)
    #Load slc par
    slc_par = gpf.par_to_dict(input.slc_par)
    # Compute the amount of shift (0 for AAA, the full for BBB and 1/2 for the remaining)
    off_multiplier = {'AAA': 0.0, 'BBB': 1.0, 'BAA': 0.5, 'ABB': 0.5}
    az_mutliplier = off_multiplier[wildcards.chan]
    coreg_off_par = gpf.par_to_dict('temp_shift')
    # Modify the azimuth shift using the multplier
    coreg_off_par.azimuth_offset_polynomial[0] = params.azimuth_shift/ slc_par.GPRI_az_angle_step
    coreg_off_par.azimuth_offset_polynomial = [coeff * az_mutliplier for coeff in coreg_off_par.azimuth_offset_polynomial]
    coreg_off_par.range_offset_polynomial = [0 for el in
                                             coreg_off_par.range_offset_polynomial]  # We do not need to correct range shift so we set it to zero
    gpf.dict_to_par(coreg_off_par, "temp_shift")
    interp_cmd = "SLC_interp {input.slc} {input.slc_par} {input.slc_par} temp_shift {output.coreg_slc} -"
    cp_cmd = "cp {input.slc_par} {output.coreg_slc_par}"
    shell(interp_cmd)
    shell(cp_cmd)

fixed_shift(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
    snakemake.wildcards)