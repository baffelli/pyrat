import pyrat.fileutils.gpri_files as gpf
import pyrat.gpri_utils.processors as proc


def run(input, output, threads, config, params, wildcards):
    slc = gpf.gammaDataset(input.slc_par, input.slc)
    slc_corr_phase = proc.process_undecimated_slc(slc, params.squint_rate, params.phase_center_shift,
                                                  params.integration_length, decimation_factor=params.dec,
                                                  correct_azimuth=True)

    slc_corr_phase.tofile(output.slc_par, output.slc)


run(snakemake.input, snakemake.output, snakemake.threads, snakemake.config, snakemake.params,
    snakemake.wildcards)
