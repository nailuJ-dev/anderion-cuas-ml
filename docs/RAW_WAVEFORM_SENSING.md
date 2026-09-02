# Generic raw-waveform channel estimation and range-Doppler pipeline

## Raw I/Q contract

`RawIqCapture` is the common complex-sample container. It carries a bounded sample vector plus
sample rate, center frequency, capture id, and timestamp. All values are validated before DSP.

`RawSensingFrame` binds a raw capture to the existing ISAC authorization/session model without
changing the existing feature-based `SensingFrame` API.

## Known-reference channel estimation

`KnownReferenceChannelEstimator` implements a deterministic regularized least-squares estimate:

`H[k] = Y[k] * conj(X[k]) / (|X[k]|^2 + lambda)`

The module also provides:

- delay-insensitive known-reference CFO estimation using differential complex correlation;
- CFO compensation before the channel FFT;
- frequency response;
- inverse-FFT complex impulse response;
- local-maximum multipath tap extraction;
- delay in samples and seconds;
- relative tap power;
- normalized frequency-domain reconstruction error.

## Range-Doppler processing

`RangeDopplerProcessor` processes a coherent pulse/CPI sequence using:

1. optional Hann taper on the matched-filter reference waveform only (never on the received range
   record, avoiding range-dependent receive attenuation);
2. FFT matched filtering against the known reference waveform;
3. causal positive-delay range bins;
4. optional slow-time Hann window;
5. slow-time FFT for Doppler;
6. FFT shift to signed Doppler bins;
7. physical range conversion;
8. optional radial-velocity conversion from carrier frequency.

The geometry can be `OneWay` or `MonostaticRoundTrip`, so the factor-of-two conversion is explicit
rather than silently assumed.

## CA-CFAR

`ca_cfar_2d` provides a two-dimensional cell-averaging CFAR over the resulting map. Guard windows,
training windows, and false-alarm probability are explicit configuration. Doppler training wraps
cyclically at the FFT edges while range training remains causal and non-wrapping. Detections include range,
Doppler, optional velocity, CUT power, threshold, and signal-to-threshold ratio.

## 5G / ISAC integration

The DSP engine is waveform-generic. 5G/ISAC-specific reference-signal extraction can be implemented
above it by constructing the known reference waveform and authorized `RawSensingFrame` captures.
The core does not assume a specific OFDM numerology, PRS/SRS layout, or RAN vendor. Reference and
received captures must declare coherent sample-rate and RF-center-frequency metadata. Raw sensing
frame timestamps must exactly match the timestamp carried by their underlying I/Q capture.
