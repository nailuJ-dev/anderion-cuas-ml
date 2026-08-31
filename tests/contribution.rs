use anderion_cuas_ml::{ContributionLedger, SensorContribution};

#[test]
fn contribution_ledger_is_sorted_and_sums_terms() {
    let ledger = ContributionLedger::new(vec![
        SensorContribution::new("trajectory", 0.2, 0.9, "stable motion").unwrap(),
        SensorContribution::new("radar", 0.4, 0.95, "micro-Doppler").unwrap(),
        SensorContribution::new("remote_id", -0.1, 0.8, "identity conflict").unwrap(),
    ])
    .unwrap();
    assert_eq!(ledger.entries()[0].source, "radar");
    assert!((ledger.total_contribution() - 0.5).abs() < 1e-6);
}
