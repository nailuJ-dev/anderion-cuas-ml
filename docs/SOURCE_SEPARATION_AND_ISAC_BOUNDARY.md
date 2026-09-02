# Source-separation and ISAC capability boundary

## Generic source separation

The public API names `DegarblingModel`, `IdentityDegarbler` and `PrototypeMaskDegarbler` are retained for semantic-versioning compatibility. Their current reference semantics are **generic feature-space source separation**, not protocol-specific Mode S / ATCRBS reply degarbling.

`IdentityDegarbler` is a no-op. `PrototypeMaskDegarbler` softly assigns abstract non-negative feature components to prototypes and reconstructs component observations. It does not parse Mode S preambles, Manchester symbols, squitters, parity/CRC or overlapping transponder replies.

New documentation should therefore use “source separation” as the capability name and mention the legacy API identifier only where needed.

## 5G / ISAC integration

The SDK is a sensing consumer and integration layer. It can validate operator-authorized sensing sessions and consume caller/provider-supplied sensing frames or derived representations. It does not implement a cellular RAN controller, waveform scheduler, raw 5G channel estimator or a generic raw-I/Q-to-range-Doppler processor.

Any external adapter that produces HRRP, range, Doppler or other ISAC-derived features is responsible for its own physical validation and authorization. The SDK validates bounded contracts and combines evidence; it does not claim equivalence to the upstream sensing stack.
