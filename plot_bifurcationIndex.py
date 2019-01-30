"""
Bifurcation index
-----------------

This example compute and plot the bifurcation index from the sample MNE-python dataset.

"""

import numpy as np
import mne
import matplotlib.pyplot as plt
from copy import deepcopy
from mnephase import bifurcation_index

### Create epochs ###
#####################

# Load the sample dataset from MNE python
data_path = mne.datasets.sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + ('/MEG/sample/sample_audvis_filt-0-40_raw-'
                           'eve.fif')
raw = mne.io.read_raw_fif(raw_fname, preload=True)
events = mne.read_events(event_fname)

# We will compare two conditions: 
# left-ear auditory stimulus (LA: event_id =1) vs. right-ear auditory stimulus (RA: event_id =2)

# Construct Epochs
include = []
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,
                       stim=False, include=include, exclude='bads')
event_id = dict(LA=1, RA=2)
tmin, tmax = -1., 1.
baseline = (None, 0)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=baseline, reject=dict(grad=4000e-13, mag=4e-12),
                    preload=True)
epochs.resample(150., npad='auto')  # resample to reduce computation time

### Compute Bifurcation index ###
#################################

# Compute phase-locking value (also called inter-trial coherence) for each condition
freqs = np.logspace(*np.log10([6, 35]), num=8) # define frequencies of interest (log-spaced)
n_cycles = freqs / 2.  # different number of cycle per frequency
PLV = np.zeros((2,np.shape(epochs.get_data())[1],len(freqs), np.shape(epochs.get_data())[2]))
for c,condition in enumerate(['LA', 'RA']):
    power, itc = mne.time_frequency.tfr_morlet(epochs[condition], freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
    PLV[c] = itc.data
                        
# Compute PLV for concatenated trials
power, itc = mne.time_frequency.tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, n_jobs=1)
PLV_AB = itc.data

# Compute BI
BI = bifurcation_index(PLV[0], PLV[1], PLV_AB)
BI_topo = deepcopy(power)
BI_topo.data = BI

### Plot Bifurcation index ###
##############################

# Plot BI averaged across sensors
plt.figure()
plt.imshow(np.mean(BI,0), extent=[epochs.times[0], epochs.times[-1],freqs[0], freqs[-1]],aspect='auto', 
            origin='lower')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.yscale('log')
plt.title('Bifurcation index between LA and RA')
cbar=plt.colorbar()

# Add a topoplot of the magnetometers for a specific time and frequency points
BI_topo.plot_topomap(ch_type = 'mag', tmin = 0, tmax = 0.1, fmin = 7, fmax = 17, baseline = None, 
                    title = 'BI 7-17Hz', show=False)
plt.show()



