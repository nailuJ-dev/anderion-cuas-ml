# Golden Path: recorded sensing to verified C-UAS perception

The Golden Path is a self-contained reference integration for detection, classification, localization, cooperative-track comparison and deterministic verification. It remains perception-only and contains no neutralization, jamming, takeover or effector-control logic.

## Run it

```bash
./scripts/run_golden_path.sh
```

Or, with Docker and no local Rust toolchain:

```bash
./scripts/run_golden_path_docker.sh
```

The default scenario is a drone-like recorded observation with a matching Remote ID track. The command:

1. loads the bundled recorded training fixtures;
2. converts physical reference measurements to a deterministic 13-feature observation;
3. fits the public encoder, detector, classifier and localizer;
4. runs optional degarbling infrastructure (identity by default);
5. compares the candidate with AIS, ADS-B and Remote ID tracks;
6. enriches the public ontology;
7. issues deterministic result certificates;
8. performs an exact replay check;
9. evaluates a held-out fixture set;
10. writes `artifacts/golden-path/result.json`.

Try the other bundled scenarios:

```bash
./scripts/run_golden_path.sh --scenario demo-data/golden-path/scenario_unmatched_drone.json
./scripts/run_golden_path.sh --scenario demo-data/golden-path/scenario_adsb_aircraft.json
./scripts/run_golden_path.sh --scenario demo-data/golden-path/scenario_bird.json
./scripts/run_golden_path.sh --scenario demo-data/golden-path/scenario_isac_drone.json
```

The ISAC scenario uses `RecordedSensingProvider` and an explicit operator authorization/configuration token to exercise the public 5G-MIMO/ISAC contract without controlling a live RAN.

## Reference measurement contract

The recorded adapter accepts bounded, finite measurements such as range, radial velocity, azimuth/elevation, SNR, Doppler spread, micro-Doppler index, RCS proxy, RF-energy proxy, burstiness and angular rate. These are normalized into a fixed 13-dimensional observation.

## What this proves

It proves the integration path and deterministic reuse of results. The fixtures are synthetic/recorded reference data and **must not** be presented as field performance. A real deployment still requires sensor-specific ingestion, representative training data and field validation.
