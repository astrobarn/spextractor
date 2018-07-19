# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:13:56 2015
Automated pEW and velocity extractor example.
@author: Seméli Papadogiannakis
"""
from __future__ import division

import matplotlib.pyplot as plt
import spextractor
plt.ioff()

# Just a test case with type Ia SN sn2006mo from the CfA sample.
filename_Ia = 'sn2006mo/sn2006mo-20061113.21-fast.flm'
z = 0.0459
pew, pew_err, vel, vel_err, gp_model = spextractor.process_spectra(filename_Ia, z, downsampling=3, plot=True, type='Ia',sigma_outliers=3)

# velocities are given in 10**3 km/s and pEW in Å
print(pew, pew_err, vel, vel_err)
print(gp_model.kern)

plt.show()
plt.savefig('Ia_example.png')


# Test case with a type Ib
filename_Ib = 'tns_2018_aqf/tns_2018aqf_2018-04-10.0_P200_DBSP_ZTF.txt'
z = 0.033
pew, pew_err, vel, vel_err = spextractor.process_spectra(filename_Ib, z, downsampling=3, plot=True, type='Ib',sigma_outliers=3)
# velocities are given in 10**3 km/s and pEW in Å
print(pew, pew_err, vel, vel_err)
plt.savefig('Ib_example.png')


filename_Ic = 'sn2017ixh/2017ixh.txt'
z = 0.011
pew, pew_err, vel, vel_err = spextractor.process_spectra(filename_Ic, z, plot=True, type='Ic',sigma_outliers=3)
# velocities are given in 10**3 km/s and pEW in Å
print(pew, pew_err, vel, vel_err)
plt.savefig('Ic_example.png')
plt.show()

