#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::env;
use std::fs;
use std::path::PathBuf;

use serde::Serialize;

use anderion_cuas_ml::{
    GoldenCuasEvaluation, GoldenCuasModel, GoldenCuasReport, ReplayStatus,
    load_golden_cuas_scenario, load_golden_cuas_scenarios, load_golden_cuas_training,
};

#[derive(Debug, Serialize)]
struct GoldenDemoOutput {
    report: GoldenCuasReport,
    replay: ReplayStatus,
    evaluation: GoldenCuasEvaluation,
    note: &'static str,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut training = PathBuf::from("demo-data/golden-path/training.json");
    let mut scenario = PathBuf::from("demo-data/golden-path/scenario_remote_id_drone.json");
    let mut evaluation = PathBuf::from("demo-data/golden-path/evaluation.json");
    let mut output = PathBuf::from("artifacts/golden-path/result.json");
    let mut seed = 42_u64;

    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--training" => training = PathBuf::from(next_arg(&mut args, "--training")?),
            "--scenario" => scenario = PathBuf::from(next_arg(&mut args, "--scenario")?),
            "--evaluation" => evaluation = PathBuf::from(next_arg(&mut args, "--evaluation")?),
            "--output" => output = PathBuf::from(next_arg(&mut args, "--output")?),
            "--seed" => seed = next_arg(&mut args, "--seed")?.parse()?,
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    let training_file = load_golden_cuas_training(&training)?;
    let scenario_file = load_golden_cuas_scenario(&scenario)?;
    let evaluation_file = load_golden_cuas_scenarios(&evaluation)?;
    let model = GoldenCuasModel::fit(&training_file.samples, seed)?;
    let report = model.infer(&scenario_file)?;
    let (_, replay) = model.replay(&scenario_file, &report)?;
    let evaluation_report = model.evaluate(&evaluation_file.scenarios)?;

    if let Some(parent) = output
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let document = GoldenDemoOutput {
        report,
        replay,
        evaluation: evaluation_report,
        note: "Fixture/recorded reference metrics only; do not interpret as field performance.",
    };
    fs::write(&output, serde_json::to_vec_pretty(&document)?)?;

    let component = document.report.components().first();
    println!("C-UAS Golden Path");
    println!("  scenario: {}", document.report.scenario_name());
    if let Some(value) = component {
        println!("  detected: {}", value.detected());
        println!(
            "  classification: {} ({:.1}%)",
            value.predicted_label(),
            value.probability() * 100.0
        );
        println!(
            "  cooperative matches: {}",
            value.cooperative_correlations().len()
        );
        println!("  verification: {:?}", value.verification_decision());
    }
    println!("  replay: {:?}", document.replay);
    println!(
        "  fixture check: {}/{} labeled fixtures correct",
        document.evaluation.correct, document.evaluation.scenarios
    );
    println!("  result: {}", output.display());
    println!("  NOTE: fixture/recorded metrics are pipeline validation, not field performance.");
    Ok(())
}

fn next_arg(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next()
        .ok_or_else(|| format!("missing value for {flag}").into())
}

fn print_help() {
    println!(
        "cuas-golden-demo [--training PATH] [--scenario PATH] [--evaluation PATH] [--output PATH] [--seed N]"
    );
}
