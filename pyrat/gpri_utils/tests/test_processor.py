from unittest import TestCase
import pyrat as pt
import matplotlib.pyplot as plt
import pyrat.gpri_utils.simulator as sim
import pyrat.fileutils.gpri_files as gpf
import numpy as np
import pyrat.visualization.visfun as vf
import pyrat.gpri_utils.processors as proc
import pyrat.gpri_utils.calibration as cal
import pyrat.core.polfun as pf


class TestProcessor(TestCase):

    def setUp(self):
        pass

    def TestCompareProcessors(self):
        #Import prf file
        prf = gpf.default_prf()
        targets = [[600,0,0.01]]
        r_ph = -0.11
        squint_rate = 4.2e-9
        prf.STP_antenna_start = -2
        prf.STP_antenna_end = 2
        prf.STP_rotation_speed = 5
        ws = 0.8
        df = 10
        z=500
        rmax=1000
        rmin =500
        kbeta=1
        #Simulate with squint
        ras = sim.RadarSimulator(targets, prf , r_ph, squint=True, antenna_bw=0.5, chan='BBBl')
        raw = ras.simulate()
        #Correct squint
        raw_desq = gpf.correct_squint(raw, squint_rate=squint_rate)
        #Range process
        slc = gpf.range_compression(raw, rmin=rmin,rmax=rmax, zero=z, kbeta=kbeta)
        slc_desq = gpf.range_compression(raw_desq, rmin=rmin, rmax=rmax, zero=z, kbeta=kbeta)
        slc_desq_1 = gpf.correct_squint_in_SLC(slc, squint_rate=squint_rate)
        #Azimuth correction
        slc_corr = cal.azimuth_correction(slc_desq, r_ph, ws=ws, filter_fun=cal.filter1d)
        slc_corr_dec = slc_corr.decimate(df)
        slc_corr_1 = proc.process_undecimated_slc(slc, squint_rate, r_ph, ws=ws, decimation_factor=df, correct_azimuth=True)
        #Check processor consistency
        f, (a1,a2) = plt.subplots(2,1, sharex=True, sharey=True)
        rgb1, *rest = vf.dismph(slc_corr_dec)
        rgb2, *rest = vf.dismph(slc_corr_1)
        a1.imshow(rgb1)
        a2.imshow(rgb2)
        plt.show()
        self.assertTrue(np.allclose(np.abs(slc_corr_dec).astype(np.float64),np.abs(slc_corr_1).astype(np.float64)))




