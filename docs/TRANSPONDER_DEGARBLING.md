# Mode S / ATCRBS raw-waveform degarbling

## Scope

`TransponderDegarbler` operates on complex raw I/Q, not on already-extracted feature vectors.
It is intended for passive 1090 MHz receive chains and performs deterministic waveform-level
separation before the rest of the C-UAS perception stack.

The older `DegarblingModel`, `IdentityDegarbler`, and `PrototypeMaskDegarbler` APIs are retained
for compatibility. They are feature-space source-separation utilities and are not the professional
Mode S / ATCRBS degarbling path.

## Mode S processing

The implementation uses the Mode S downlink signal structure:

- four-pulse 8 microsecond preamble;
- preamble pulse leading edges at 0, 1.0, 3.5, and 4.5 microseconds;
- 0.5 microsecond pulse width;
- 1 Mbit/s PPM data;
- 56- or 112-bit replies;
- CRC-24 polynomial `0xFFF409`;
- DF extraction from the first five data bits.

For DF17/DF18/DF19 extended-squitter formats, a non-zero direct CRC syndrome rejects the
candidate. DF0/4/5/16 and the DF24 Comm-D family use address-parity semantics. DF11 exposes the
parity/interrogator-identifier syndrome. DF20/DF21 are deliberately classified as
`AddressOrDataParity`: without the corresponding interrogation context the receiver cannot safely
claim that the overlay syndrome is an ICAO address because Comm-B may use Data Parity. The SDK
therefore never exposes a DF20/DF21 overlay as a recovered address by assumption.

## ATCRBS processing

The implementation detects the F1/F2 framing pair and samples the standard pulse grid:

- F1 at 0 microseconds;
- F2 at 20.3 microseconds;
- information-pulse spacing of 1.45 microseconds;
- nominal pulse width of 0.45 microseconds;
- optional SPI at 24.65 microseconds from F1.

The decoded object exposes:

- all 13 information positions between F1/F2;
- Mode A digits `[A, B, C, D]`;
- the 11 Mode C Gillham information bits in order
  `C1,A1,C2,A2,C4,A4,B1,B2,D2,B4,D4`;
- SPI presence;
- confidence and estimated carrier-frequency offset.

Mode C semantic altitude conversion is deliberately not inferred from the waveform alone because
the reply does not carry the interrogation context. The raw Gillham bits are preserved for a caller
that has Mode C interrogation context.

## Successive interference cancellation

For each iteration the degarbler:

1. computes a power-envelope prefix integral;
2. finds the highest selection-metric valid Mode S or ATCRBS candidate, combining decode confidence
   with measured framing/preamble power so SIC preferentially removes strong reliable replies;
3. reconstructs its protocol pulse mask;
4. estimates residual carrier-frequency offset from gated complex samples;
5. estimates the complex reply gain;
6. subtracts a bounded fraction of the reconstructed reply from the residual;
7. re-runs detection on the residual.

The result includes all recovered replies and the residual I/Q capture so downstream algorithms can
inspect cancellation quality or run a different separator.

## Sampling guidance

The API accepts sample rates from 2 MS/s upward. For actual overlapping Mode S/ATCRBS separation,
8 MS/s or above is strongly recommended because 0.45/0.5 microsecond pulse timing and carrier-offset
estimation become materially better resolved.

## Public safety boundary

This module is receive-side sensing and decoding only. It contains no interrogation transmitter,
jamming, spoofing, takeover, effector, targeting, or neutralization control.
