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
filename = 'sn2006mo/sn2006mo-20061113.21-fast.flm'
z = 0.0459
pew, pew_err, vel = spextractor.process_spectra(filename, z, downsampling=3, plot=True)

# velocities are given in 10**3 km/s and pEW in Å
print(pew, pew_err, vel)

plt.show()

