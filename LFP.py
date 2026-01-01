import numpy as np
from scipy.signal import hilbert, sosfiltfilt, butter, iirnotch, tf2sos, windows, convolve, find_peaks


# define hyper parameter for reusability
# gaussian kernel for smoothing ripple band power
gauss_kernel = windows.gaussian(101,12.5) # gaussian smoothing s.d. 12.5ms
gauss_kernel /= gauss_kernel.sum()  # normalize

fs = 1000


def preprocess_LFP(lfp):
    
    # notch filter, remove 60Hz and harmonics, iirnotch is better than butterworth for this purpose
    
    preprocessed = lfp.copy()
    
    for f0 in [60, 120, 180, 240]:

        b, a = iirnotch(f0,f0,fs)
        sos = tf2sos(b,a)
        preprocessed = sosfiltfilt(sos, preprocessed)
    
    return preprocessed



def freq_band(preprocessed, band):

    # 1. bandpass for the given band frequency
    sos = butter(5, band, btype='bandpass', output='sos', fs=fs)
    band_sig = sosfiltfilt(sos, preprocessed,axis=0)

    # 2. calculate the power and phase of this frequency band

    hilb_sig = hilbert(band_sig, axis=0)
    power = np.abs(hilb_sig)**2
    phase = np.angle(hilb_sig)


    return band_sig, power, phase



def ripple_detection(preprocessed, band):

    band_sig, power, _ = freq_band(preprocessed, band)
    smoothed = convolve(power.ravel(),gauss_kernel,mode='same')
    zscored = (smoothed-smoothed.mean(axis=0,keepdims=True))/smoothed.std(axis=0,keepdims=True)
    
    candidate_peaks,_ = find_peaks(zscored,height=3,distance=50)

    true_peaks = []

    for peak_ in candidate_peaks:

        on_ = peak_ - np.argmax(zscored[:peak_][::-1]<3)
        off_ = peak_ + np.argmax(zscored[peak_:]<3)
        
        if off_-on_>50:
            
            true_peaks.append(np.array([peak_, on_, off_]))
            
    if len(true_peaks)>0:
        true_peaks = np.stack(true_peaks)

        # need to modify this, it happens when two peaks are >50ms away, 
        # but detecting the same ripple since its duration is quite long 

    return band_sig, power, zscored, true_peaks


def compute_phase_alignment(phase):

    phase_alignment = np.zeros(phase.shape[1])
    
    for tt in range(phase.shape[1]):
        phase_alignment[tt] = np.abs(np.sum(np.exp(1j*phase[:,tt])))/phase.shape[0]

    return phase_alignment