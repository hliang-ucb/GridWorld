# a more traditional way to calcualte spike-LFP coherence
# multitaper LFP and then FFT, do FFT on spikes, then clculate conjugate to get power
# this method is subject to changes in firing rates and therefore cannot compare two neurons with different firing rates

yf = rfft((y[k,:]-mean(y[k,:])) *hanning(N))    # Hanning taper the field,
nf = rfft((n[k,:]-mean(n[k,:])))                # ... but do not taper the spikes.
SYY = SYY + ( real( yf*conj(yf) ) )/K           # Field spectrum
SNN = SNN + ( real( nf*conj(nf) ) )/K           # Spike spectrum
SYN = SYN + (          yf*conj(nf)   )/K        # Cross spectrum