import numpy as np
from math import sqrt, atan2, pi, floor,exp


def rayleigh_test(phases):
""" Test if the phase distribution is uniform

Parameters
----------
phases: 

Returns
----------
pval: p-value of the Rayleigh test 

"""
    ph = phases.reshape(-1)
    n = len(ph)
    w = np.ones(n)
    r = np.sum(w*np.exp(1j*ph))
    r = abs(r)/np.sum(w)
    R = n*r
    z = R**2 / n
    pval = exp(sqrt(1+4*n+4*(n**2-R**2))-(1+2*n))
    return pval

def circular_mean(phases):
""" Compute the circular mean

Parameters
----------
phases: 

Returns
----------
circular mean

"""
    ang = np.mean(np.exp(1j*phases))
    return atan2(ang.imag, ang.real)


def InstPhase(dataD, times, Fs, frequencies, n_cycles):
""" Compute the instantaneous phase for each trial at onset_time

Parameters
----------
dataD:
time:
Fs:
frequencies:
n_cycles: 

Returns
----------
phase_onset: 

"""

     t = np.linspace(times[0],times[1],(times[1]-times[0])*1000+1)
     Ws =morlet(Fs,frequencies, n_cycles,zero_mean=True)[0]

     n_frequencies = len(frequencies)
     n_epochsD, n_times = dataD[:, :].shape
 
     pad = (Ws.size- len(t)) / 2
     Ws = Ws[pad:-pad]
     padstart = int(n_times-((times[1]-times[0])*1000+1))

     phase_onset =np.empty((n_epochsD))

     for epind in range(n_epochsD):   
            #phase_onset[epind]= np.angle(np.convolve(dataD[epind], Ws, 'same'))[np.argmin(np.abs(t - 0.))]
            phase_onset[epind]= np.angle(np.sum(dataD[epind,padstart:] * Ws[::-1])) 

     return phase_onset

