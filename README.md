# anderion-cuas-ml v0.7.0 direct files

This package contains only the files needed to add the approved professional raw-waveform path to the existing `release/vnext` repository.

Use `MANUAL_EDITS.md` as the copy list.

Core additions:

- real Mode S / ATCRBS raw-I/Q detection and degarbling with successive interference cancellation;
- Mode S 56/112-bit PPM decoding with DF19/DF24 handling and conservative AP/DP parity semantics;
- ATCRBS F1/F2 pulse-grid decoding, Mode A digits, Mode C Gillham-bit extraction, SPI;
- robust gated transponder CFO/complex-gain estimation plus delay-insensitive known-reference CFO estimation;
- regularized known-reference channel estimation, CIR and multipath taps;
- generic matched-filter + slow-time FFT range-Doppler map;
- 2D CA-CFAR;
- raw sensing frames bound to the existing authorized ISAC session contract.

# anderion-cuas-ml

**Open-source multi-sensor perception for Counter-UAS research and airspace sensing.**

`anderion-cuas-ml` provides reusable Rust components for detecting, classifying, localizing and tracking aerial objects using multiple sensing modalities.

The project focuses exclusively on the **perception side** of Counter-UAS systems.

It does not provide jamming, takeover, neutralization, weapon control or autonomous engagement functionality.

## What it does

The SDK combines information from sources such as:

* radar
* RF observations
* Remote ID
* ADS-B
* AIS where relevant
* recorded or authorized 5G / ISAC sensing
* trajectory information
* multimodal observations

to build more reliable local perception than any individual sensor can provide alone.

## Core capabilities

* aerial-object detection
* drone classification
* localization
* multi-object tracking
* sensor fusion
* uncertainty estimation
* trajectory analysis
* multimodal alignment
* cooperative identification correlation
* robustness evaluation
* deterministic verification

## Core additions:

* real Mode S / ATCRBS raw-I/Q detection and degarbling with successive interference cancellation;
* Mode S 56/112-bit PPM decoding with DF19/DF24 handling and conservative AP/DP parity semantics;
* ATCRBS F1/F2 pulse-grid decoding, Mode A digits, Mode C Gillham-bit extraction, SPI;
* robust gated transponder CFO/complex-gain estimation plus delay-insensitive known-reference CFO estimation;
* regularized known-reference channel estimation, CIR and multipath taps;
* generic matched-filter + slow-time FFT range-Doppler map;
* 2D CA-CFAR;
* raw sensing frames bound to the existing authorized ISAC session contract.

## Cooperative identity trust

Remote ID and other cooperative broadcasts are useful evidence, but they should not automatically be treated as physical truth.

The SDK can compare cooperative declarations with sensor observations:

```text
Remote ID
position / velocity / identity
             ↓
         consistency
             ↑
radar / RF / trajectory / ISAC
```

The result can distinguish:

```text
CONSISTENT
WEAKLY_CONSISTENT
CONFLICT
INSUFFICIENT_EVIDENCE
```

without automatically interpreting an inconsistency as malicious activity.

This provides a practical foundation for detecting corrupted, incorrect or inconsistent cooperative information.

## Physics-guided micro-Doppler

Rotating blades, wings and other moving structures create characteristic Doppler modulations.

The SDK exposes interpretable micro-Doppler features including:

* Doppler centroid
* Doppler bandwidth
* periodicity
* harmonic structure
* spectral entropy
* sideband symmetry
* feature confidence

These descriptors can help distinguish drones from birds and other aerial objects, including under degraded SNR conditions.

## 5G / ISAC dual-view sensing

For recorded or authorized ISAC observations, the SDK can combine complementary representations such as:

```text
micro-Doppler
      +
HRRP / range structure
      +
kinematics
      ↓
fusion
```

This is useful when one sensing representation alone is ambiguous.

The library remains a sensing consumer and does not control cellular infrastructure.

## Occlusion-aware tracking

Real targets disappear temporarily behind buildings, vegetation, terrain or sensor coverage gaps.

The tracking layer can maintain an uncertainty envelope during missed detections and evaluate later observations for reacquisition.

The reacquisition score can combine:

* predicted kinematics
* elapsed time
* embedding similarity
* class consistency
* cooperative evidence

Example:

```text
track 42
visible
   ↓
occluded
   ↓
uncertainty grows
   ↓
candidate reappears
   ↓
reacquisition confidence
```

## Local group perception

The SDK can analyze multiple simultaneous tracks and identify local motion structure such as:

* group formation
* spatial coherence
* velocity coherence
* convergence
* divergence
* fragmentation
* synchronization

This is perception, not intent attribution.

The library describes observable motion relationships rather than claiming to infer operational intent.

## Raw-waveform transponder degarbling

The professional degarbling path operates directly on complex I/Q and supports both Mode S and
ATCRBS replies. It detects protocol timing, decodes candidate replies, validates Mode S CRC semantics
where applicable, estimates residual carrier offset, reconstructs replies, and performs successive
interference cancellation before re-detecting weaker overlapping replies.

```rust
use anderion_cuas_ml::{RawIqCapture, TransponderDegarbler};

let result = TransponderDegarbler::default().separate(&capture)?;
for reply in result.replies() {
    println!("{:?} confidence={:.3}", reply.protocol(), reply.confidence());
}
```

The legacy `PrototypeMaskDegarbler` remains available for feature-space experiments but is not a
Mode S / ATCRBS waveform degarbler.

See `docs/TRANSPONDER_DEGARBLING.md`.

## Generic raw-waveform sensing

The SDK now includes raw-I/Q channel estimation and range-Doppler processing:

* delay-insensitive known-reference CFO estimation and compensation;
* regularized LS frequency-response estimation;
* complex CIR and multipath taps;
* delay-invariant FFT matched filtering with optional reference-waveform Hann taper;
* coherent slow-time Doppler FFT;
* one-way or monostatic physical range/velocity axes;
* 2D CA-CFAR detections with cyclic Doppler-edge training;
* `RawSensingFrame` validation against authorized ISAC sessions.

See `docs/RAW_WAVEFORM_SENSING.md`.

## Sensor contribution ledger

A fused decision can expose which observations contributed to it.

Example:

```text
DRONE: 0.94

micro-Doppler      +0.30
RF observation     +0.23
trajectory         +0.17
Remote ID conflict +0.11
ISAC range profile +0.09
uncertainty        -0.04
```

These values describe documented fusion contributions.

They should not be interpreted as universal causal explanations unless the underlying model explicitly supports that interpretation.

## Current problems addressed

The project is intended to help research and engineering teams investigate practical sensing problems such as:

* drone vs bird discrimination
* weak targets in clutter
* inconsistent Remote ID
* multiple simultaneous drones
* temporary sensor dropout
* track loss during occlusion
* radar/RF disagreement
* sensor uncertainty
* multi-sensor association
* dense urban airspace

## Example architecture

```text
Radar ───────┐
RF ──────────┤
Remote ID ───┤
ADS-B ───────┤
ISAC ────────┤
             ↓
      multi-sensor fusion
             ↓
     classification
     localization
     tracking
     trust analysis
     group perception
             ↓
       evidence output
```

## Quick start

```bash
git clone https://github.com/nailuJ-dev/anderion-cuas-ml.git
cd anderion-cuas-ml

cargo build --release
cargo test --all-targets --all-features
```

Run the reference demonstration where available:

```bash
cargo run --release --locked --bin cuas-golden-demo
```

## Integration with `spectra-sim`

Synthetic radar, RF and ISAC scenarios can be generated externally and passed to this SDK.

```text
spectra-sim
    ↓
physical sensing scenario
    ↓
anderion-cuas-ml
    ↓
perception + fusion + verification
```

The projects remain independently usable.

## Safety boundary

This repository is deliberately limited to sensing and perception.

It does **not** provide:

* RF jamming
* communications takeover
* protocol exploitation
* effector control
* weapon guidance
* target engagement
* autonomous interception
* neutralization logic

The objective is better situational understanding.

## Design principles

* multi-sensor evidence over single-source certainty
* uncertainty-aware tracking
* physics-informed features
* explainable fusion
* deterministic verification where possible
* perception-only architecture
* reproducible evaluation

## Contributing

Contributions are particularly useful in:

* drone/bird datasets
* radar and micro-Doppler processing
* multi-target tracking
* sensor fusion
* Remote ID consistency
* ISAC sensing
* robustness benchmarks
* uncertainty calibration
* real-world validation

Please include reproducible tests and technical references where applicable.

## Support the project

If you find the project useful:

* star the repository
* test it against public datasets
* report difficult scenarios
* contribute sensing algorithms
* contribute reproducible benchmarks
* share the project with radar, RF, robotics and airspace-safety teams

The most useful support is independent testing against real sensing problems.

## About

`anderion-cuas-ml` is part of the open-source RF and sensing initiative developed by **Anderion Systems**.

The aim is to make advanced electromagnetic and multi-sensor perception tooling more accessible to engineers, researchers and organizations building safer and more reliable sensing systems.

**Anderion Systems** — https://anderion-systems.com

## License

See the repository `LICENSE` file.
