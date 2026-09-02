# Cooperative correlation, optional source separation, and 5G-MIMO/ISAC integration

This extension is designed to keep the C-UAS SDK self-contained and perception-only while allowing applications to enrich ML results with cooperative identity evidence and operator-authorized infrastructure sensing.

## Cooperative correlation

The SDK does not connect to AIS, ADS-B or Remote ID networks. The caller supplies already acquired and parsed `CooperativeTrack` records. Supported identity kinds are `Ais`, `Adsb`, `RemoteId`, and `Custom`.

`CooperativeCorrelator` applies a deterministic gate and score over:

- horizontal geodesic distance;
- timestamp delta;
- optional NED velocity delta;
- source confidence supplied by the caller.

Results are sorted deterministically by score, identity kind, identity, distance and timestamp delta. Raw ML class scores remain unchanged. `CooperativeDisposition::MatchedCooperative` is additional evidence, not a replacement classifier.

Use AIS primarily for maritime/coastal context, ADS-B for cooperative aircraft, and Remote ID for cooperative UAS. The generic contract allows applications to enable only the sources relevant to their environment.

## Optional source separation (compatibility API: degarbling)

`DegarblingModel` is a replaceable pre-inference interface. `IdentityDegarbler` is the no-op mode. `PrototypeMaskDegarbler` is the public deterministic reference implementation: each feature is softly assigned to non-negative component prototypes and component observations reconstruct the original mixed observation.

The reference algorithm is intentionally generic. Production users can train or implement another separator behind the same public trait without changing `EnhancedPerceptionPipeline`.

## Operator-authorized 5G-MIMO/ISAC

`ActiveSensingProvider` is an integration port, not a base-station controller.

A sensing session is accepted only when:

1. `OperatorAuthorization` is valid for the current time;
2. authorization and request target the same infrastructure;
3. the requested sensing mode is explicitly allowed;
4. the provider advertises that mode;
5. requested feature dimensions stay inside configured bounds.

`SensingMode::OperatorManagedActive` represents active sensing explicitly controlled by the infrastructure operator. The SDK contains no waveform generator, scheduler manipulation, RF transmission command, private RAN API, or vendor-specific implementation.

The included `RecordedSensingProvider` enables deterministic development, tests and replay without network infrastructure. A real operator/vendor adapter implements `ActiveSensingProvider` outside this crate and performs its own authentication, legal authorization and equipment-specific control.

## Deterministic verification binding

A `SensingFrame` can derive a `VerificationContext` whose configuration digest includes:

- the operator authorization artifact digest;
- the sensing configuration digest;
- the infrastructure identity;
- the sensing mode;
- the feature dimension.

The ordinary deterministic verifier then binds this context to the model, observation and result. A replay under a different operator authorization or sensing configuration therefore cannot be considered context-identical.

## Trust boundary

The following are untrusted inputs and must be revalidated: cooperative tracks, authorization objects, sensing requests, sensing capabilities, sensing frames and serialized degarbling models.

The SDK does not claim that a cooperative match proves identity, that an authorization digest proves legal authority, or that ML degarbling recovers physical sources perfectly. Those outputs remain evidence with explicit confidence and validation boundaries.
