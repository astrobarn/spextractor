# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:13:56 2015
Automated pEW and velocity extractor example.
@author: Seméli Papadogiannakis
"""
from __future__ import division, print_function

import matplotlib.pyplot as plt
import spextractor

plt.ioff()


def report(magnitude, error):
    for k in sorted(magnitude):
        print(k, magnitude[k], '+-', error[k])


# Just a test case with type Ia SN sn2006mo from the CfA sample.
filename_Ia = 'sn2006mo/sn2006mo-20061113.21-fast.flm'
z = 0.0459
pew, pew_err, vel, vel_err, gp_model = spextractor.process_spectra(filename_Ia, z, downsampling=3, plot=True, type='Ia',
                                                                   sigma_outliers=3)

# velocities are given in 10**3 km/s and pEW in Å
print('pew (Å)')
report(pew, pew_err)
print('velocities (10**3 km/s)')
report(vel, vel_err)
print()

plt.title('Ia example')
plt.tight_layout()
plt.savefig('Ia_example.png', dpi=300)

out = spextractor.process_spectra(filename_Ia, z, downsampling=3, plot=True, type='Ia', sigma_outliers=3,
                                  high_velocity=True)
pew, pew_err, vel, vel_err, lambda_hv_results, lambda_hv_err_results, vel_hv, vel_hv_err, gp_model = out

for line in ('Ca II H&K', 'Si II 6150A'):
    print(line)
    for l, l_err in zip(lambda_hv_results[line], lambda_hv_err_results[line]):
        print('         :', l, '+-', l_err)

plt.title('Ia example, with high velocity features')
plt.tight_layout()
plt.savefig('Ia_hv_example.png', dpi=300)

# Test case with a type Ib
filename_Ib = 'tns_2018_aqf/tns_2018aqf_2018-04-10.0_P200_DBSP_ZTF.txt'
z = 0.033
# In this case, we set auto_prune to False to analyse the whole spectrum.
pew, pew_err, vel, vel_err, gp_model = spextractor.process_spectra(filename_Ib, z, downsampling=3, plot=True, type='Ib',
                                                                   sigma_outliers=3, auto_prune=False)
# velocities are given in 10**3 km/s and pEW in Å
print('pew (Å)')
report(pew, pew_err)
print('velocities (10**3 km/s)')
report(vel, vel_err)
print()
plt.title('Ib example')
plt.tight_layout()
plt.savefig('Ib_example.png', dpi=300)

filename_Ic = 'sn2017ixh/2017ixh.txt'
z = 0.011
pew, pew_err, vel, vel_err, gp_model = spextractor.process_spectra(filename_Ic, z, plot=True, type='Ic',
                                                                   sigma_outliers=3)
# velocities are given in 10**3 km/s and pEW in Å
print('pew (Å)')
report(pew, pew_err)
print('velocities (10**3 km/s)')
report(vel, vel_err)
print()

plt.title('Ic example')
plt.tight_layout()
plt.savefig('Ic_example.png', dpi=300)
plt.show()
