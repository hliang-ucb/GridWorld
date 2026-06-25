import numpy as np
from scipy.signal import hilbert, sosfiltfilt, butter, iirnotch, tf2sos, windows, convolve, find_peaks, coherence
import pywt 
from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic


# define hyper parameter for reusability
# gaussian kernel for smoothing ripple band power
gauss_kernel = windows.gaussian(101,12.5) # gaussian smoothing s.d. 12.5ms
gauss_kernel /= gauss_kernel.sum()  # normalize

fs = 1000


def notchfilter(lfp):
    
    # notch filter, remove 60Hz and harmonics, iirnotch is better than butterworth for this purpose
    
    notchfiltered = lfp.copy()
    
    for f0 in [60, 120, 180, 240]:

        b, a = iirnotch(f0,f0/1.2,fs)
        sos = tf2sos(b,a)
        notchfiltered = sosfiltfilt(sos, notchfiltered)
    
    return notchfiltered



def freq_band(notchfiltered, band_freq):

    # 1. bandpass for the given band frequency
    sos = butter(5, band_freq, btype='bandpass', output='sos', fs=fs)
    band_sig = sosfiltfilt(sos, notchfiltered,axis=0)

    # 2. calculate the power and phase of this frequency band

    hilb_sig = hilbert(band_sig, axis=0)
    power = np.abs(hilb_sig)**2
    phase = np.angle(hilb_sig)


    return band_sig, power, phase



def ripple_detection(notchfiltered, band):

    band_sig, power, _ = freq_band(notchfiltered, band)
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


def wavelet(notchfiltered, freq_range, wavelet = 'cmor1.5-1.0', fs=1000):

    # using complex Morlet with bandwidth=1.5, center freq=1.0
    
    # Convert frequencies to scales for Morlet wavelet in pywt
    center_freq = pywt.central_frequency(wavelet)
    scales = center_freq * fs / freq_range
    
    # Compute the Continuous Wavelet Transform (CWT)
    coeffs, freqs = pywt.cwt(notchfiltered, scales, wavelet, sampling_period=1/fs)

    return freqs, coeffs


def cross_spectrum(xf,yf):

    SXX = np.real(xf*np.conj(xf))
    SYY = np.real(yf*np.conj(yf))
    SXY = xf*np.conj(yf)  
    # cross_power = np.abs(SXY)**2
    # phase = np.angle(SXY)
    
    return SXX, SYY, SXY


def flatten_spec(freqs,power,max_n_peaks=6):

    fm = FOOOF(max_n_peaks=max_n_peaks)
    fm.fit(freqs, power.mean(axis=1), [freqs[0], freqs[-1]])
    init_ap_fit = gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
    flat_spec = fm.power_spectrum-init_ap_fit

    return init_ap_fit, flat_spec


def coherogram_fft(sig1, sig2, window = 1000, slide=25, fs=1000):
    
    timestamps = np.arange(window, sig1.shape[1]-window+slide, slide)
    
    Cxy = np.zeros((sig1.shape[0],len(timestamps),251))

    for trial in tqdm(range(sig1.shape[0])):

        for ii, tt in enumerate(timestamps):
        
            f_coh, Cxy[trial,ii,:] = coherence(sig1[trial,tt-window:tt+window],sig2[trial,tt-window:tt+window],
                                               fs=fs,nperseg=fs/2,noverlap=fs/2*0.75)

    return f_coh, Cxy


# def find_high_power_channel():