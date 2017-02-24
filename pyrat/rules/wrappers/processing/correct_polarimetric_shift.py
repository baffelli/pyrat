import pyrat.fileutils.gpri_files as gpf
from snakemake import shell


def correct_shift(input, output, threads, config, params, wildcards):
    # Compute the amount of shift (0 for AAA, the full for BBB and 1/2 for the remaining)
    off_multiplier = {'AAA': 0.0, 'BBB': 1.0, 'BAA': 0.5, 'ABB': 0.5}
    az_mutliplier = off_multiplier[wildcards.chan]
    coreg_off_par = gpf.par_to_dict(input.off_par)
    # Modify the azimuth shift using the multplier
    coreg_off_par.azimuth_offset_polynomial =[coeff * az_mutliplier for coeff in coreg_off_par.azimuth_offset_polynomial]
    coreg_off_par.range_offset_polynomial = [0 for el in
                                             coreg_off_par.range_offset_polynomial]  # We do not need to correct range shift so we set it to zero
    gpf.dict_to_par(coreg_off_par, params.coreg_off_par)
    interp_cmd = "SLC_interp {input.slc} {input.slc_par} {input.slc_par} {params.coreg_off_par} {output.coreg_slc} -"
    cp_cmd = "cp {input.slc_par} {output.coreg_slc_par}"
    shell(interp_cmd)
    shell(cp_cmd)

    

correct_shift(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
    snakemake.wildcards)