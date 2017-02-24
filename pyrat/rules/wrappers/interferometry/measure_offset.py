from snakemake import shell

import pyrat.fileutils.gpri_files as gpf


def measure_offset(input, output, threads, config, params, wildcards):
    # Create offset parameters
    diff_cmd = "create_offset {input.master_slc_par} {input.slave_slc_par} {output.shift_par} 1 {params.rlks} {params.azlks} 0".format(
        input=input, params=params, output=output)
    shell(diff_cmd)
    # Initial offset estimate
    for rlks, azlks in zip(range(5, 1, -1), range(10, 2, -1)):
        # Read initial estimate
        off_par = gpf.par_to_dict(output.shift_par)
        roff = off_par['range_offset_polynomial'][0]
        azoff = off_par['azimuth_offset_polynomial'][0]
        init_offset_cmd = "init_offset {input.master_slc}  {input.slave_slc} {input.master_slc_par} {input.slave_slc_par} " \
                          "{output.shift_par} {rlks} {azlks} {params.ridx} {params.azidx} " \
                          "{roff} {azoff} 0.1 {params.rwin} {params.azwin} 1".format(input=input, output=output,
                                                                                     params=params, rlks=rlks,
                                                                                     azlks=azlks, roff=roff, azoff=azoff)

        shell(init_offset_cmd)
    # Compute offset field
    offset_cmd = "offset_pwr {input.master_slc} {input.slave_slc} {input.master_slc_par} {input.slave_slc_par} " \
                 "{output.shift_par} {output.offs} {output.snr} {params.rwin} " \
                 "{params.azwin} - - {params.noff_r} {params.noff_az} 0.3 ".format(input=input, output=output, params=params)
    #Call shell command
    shell(offset_cmd)
    #Fit offset polynomial
    fit_cmd = "offset_fit {output.offs} {output.snr} {output.shift_par} - - 0.3 1 0".format(output=output)
    shell(fit_cmd)


measure_offset(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
               snakemake.wildcards)
