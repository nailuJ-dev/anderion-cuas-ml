use anderion_cuas_ml::{
    AutoregressiveTrajectoryPredictor, ClassScore, ComputeBackend, CpuBackend,
    DistilledPrototypeClassifier, Embedding, FederatedAverager, FederatedDelta, GraphEdge,
    GraphMessagePasser, Localization, MultimodalSelfAttention, NeuralLocalizer, PairedModalAligner,
    Position3, SoftLabelSample, SymmetricQuantizer, TrajectorySample, UncertaintyWeightedFusion,
    WeightedPerceptionEnsemble,
};
use anderion_cuas_ml::{Classifier, Localizer};

fn embedding(values: &[f32]) -> Embedding {
    Embedding::new(values.to_vec()).expect("test embedding is valid")
}

#[test]
fn graph_message_passing_preserves_node_count_and_dimension() {
    let graph = GraphMessagePasser::new(0.6, 0.4).expect("graph model");
    let nodes = vec![embedding(&[1.0, 0.0]), embedding(&[0.0, 1.0])];
    let edges = vec![
        GraphEdge::new(0, 1, 1.0).expect("edge"),
        GraphEdge::new(1, 0, 1.0).expect("edge"),
    ];
    let output = graph.propagate(&nodes, &edges).expect("propagation");
    assert_eq!(output.len(), 2);
    assert!(output.iter().all(|item| item.dim() == 2));
}

#[test]
fn neural_localizer_trains_and_returns_finite_uncertainty() {
    let samples = vec![
        (
            embedding(&[0.0, 0.0]),
            Position3::new(0.0, 0.0, 0.0).expect("position"),
        ),
        (
            embedding(&[1.0, 0.0]),
            Position3::new(1.0, 0.0, 0.0).expect("position"),
        ),
        (
            embedding(&[0.0, 1.0]),
            Position3::new(0.0, 1.0, 0.0).expect("position"),
        ),
    ];
    let model = NeuralLocalizer::fit(&samples, 4, 20, 0.05, 0.0, 17).expect("fit localizer");
    let localized = model.localize(&embedding(&[0.5, 0.5])).expect("localize");
    assert!(localized.sigma_m.is_finite());
    assert!((0.0..=1.0).contains(&localized.confidence));
}

#[test]
fn trajectory_predictor_learns_generic_motion_mapping() {
    let samples = vec![
        TrajectorySample::new(
            Position3::new(0.0, 0.0, 0.0).expect("position"),
            [1.0, 0.0, 0.0],
            1.0,
            Position3::new(1.0, 0.0, 0.0).expect("position"),
        )
        .expect("sample"),
        TrajectorySample::new(
            Position3::new(0.0, 0.0, 0.0).expect("position"),
            [0.0, 1.0, 0.0],
            1.0,
            Position3::new(0.0, 1.0, 0.0).expect("position"),
        )
        .expect("sample"),
    ];
    let predictor =
        AutoregressiveTrajectoryPredictor::fit(&samples, 40, 0.05, 0.0).expect("fit predictor");
    let predicted = predictor
        .predict(
            Position3::new(0.0, 0.0, 0.0).expect("position"),
            [1.0, 0.0, 0.0],
            1.0,
        )
        .expect("predict");
    assert!(predicted.x.is_finite() && predicted.y.is_finite() && predicted.z.is_finite());
}

#[test]
fn multimodal_attention_and_self_supervised_alignment_are_generic() {
    let attention = MultimodalSelfAttention::new(0.8).expect("attention");
    let pooled = attention
        .aggregate(&[
            ("sensor-a".to_string(), embedding(&[1.0, 0.0])),
            ("sensor-b".to_string(), embedding(&[0.8, 0.2])),
        ])
        .expect("aggregate");
    assert_eq!(pooled.dim(), 2);

    let aligner = PairedModalAligner::fit(&[
        ("sensor-a".to_string(), embedding(&[1.0, 0.0])),
        ("sensor-b".to_string(), embedding(&[0.0, 1.0])),
        ("sensor-a".to_string(), embedding(&[0.8, 0.2])),
        ("sensor-b".to_string(), embedding(&[0.2, 0.8])),
    ])
    .expect("aligner");
    let aligned = aligner
        .align("sensor-a", &embedding(&[1.0, 0.0]))
        .expect("align");
    assert_eq!(aligned.dim(), 2);
}

#[test]
fn federated_averager_clips_and_aggregates_caller_supplied_deltas() {
    let averager = FederatedAverager::new(8, 16, 1.0).expect("averager");
    let deltas = vec![
        FederatedDelta::new("client-a", vec![1.0, 0.0], 10).expect("delta"),
        FederatedDelta::new("client-b", vec![0.0, 1.0], 10).expect("delta"),
    ];
    let aggregate = averager.aggregate(&deltas).expect("aggregate");
    assert_eq!(aggregate.len(), 2);
    assert!(aggregate.iter().all(|value| value.is_finite()));
}

#[test]
fn distillation_and_quantization_are_standalone() {
    let samples = vec![
        SoftLabelSample::new(
            embedding(&[1.0, 0.0]),
            vec![
                ClassScore::new("drone", 0.9).expect("score"),
                ClassScore::new("other", 0.1).expect("score"),
            ],
        )
        .expect("soft sample"),
        SoftLabelSample::new(
            embedding(&[0.0, 1.0]),
            vec![
                ClassScore::new("drone", 0.1).expect("score"),
                ClassScore::new("other", 0.9).expect("score"),
            ],
        )
        .expect("soft sample"),
    ];
    let student = DistilledPrototypeClassifier::fit(&samples, 1.0).expect("student");
    let scores = student.classify(&embedding(&[0.9, 0.1])).expect("classify");
    assert_eq!(
        scores.first().map(|score| score.label.as_str()),
        Some("drone")
    );

    let quantizer = SymmetricQuantizer::new(8).expect("quantizer");
    let restored = quantizer
        .dequantize(
            &quantizer
                .quantize(&embedding(&[-1.0, 0.5, 1.0]))
                .expect("quantize"),
        )
        .expect("dequantize");
    assert_eq!(restored.dim(), 3);
}

#[test]
fn heterogeneous_backend_and_ensemble_have_open_contracts() {
    let backend = CpuBackend;
    let out = backend
        .matvec(&[vec![1.0, 2.0], vec![0.5, -1.0]], &[2.0, 1.0])
        .expect("matvec");
    assert_eq!(out.len(), 2);

    let ensemble = WeightedPerceptionEnsemble::new(vec![0.6, 0.4]).expect("ensemble");
    let fused = ensemble
        .combine_classifications(&[
            vec![
                ClassScore::new("drone", 0.8).expect("score"),
                ClassScore::new("other", 0.2).expect("score"),
            ],
            vec![
                ClassScore::new("drone", 0.6).expect("score"),
                ClassScore::new("other", 0.4).expect("score"),
            ],
        ])
        .expect("combine");
    assert_eq!(
        fused.first().map(|score| score.label.as_str()),
        Some("drone")
    );
}

#[test]
fn uncertainty_weighted_fusion_returns_a_valid_localization() {
    let fusion = UncertaintyWeightedFusion;
    let localizations = vec![
        Localization::new(Position3::new(0.0, 0.0, 0.0).expect("position"), 2.0, 0.8)
            .expect("localization"),
        Localization::new(Position3::new(2.0, 0.0, 0.0).expect("position"), 1.0, 0.9)
            .expect("localization"),
    ];
    let output = fusion.fuse(&localizations).expect("fusion");
    assert!(output.position.x > 0.0 && output.position.x < 2.0);
    assert!((0.0..=1.0).contains(&output.confidence));
}

#[test]
fn multimodal_transformer_encoder_returns_finite_embedding() {
    let model =
        anderion_cuas_ml::MultimodalTransformerEncoder::new(1.0, 0.25).expect("transformer");
    let output = model
        .aggregate(&[
            ("a".to_string(), embedding(&[1.0, 0.0])),
            ("b".to_string(), embedding(&[0.5, 0.5])),
        ])
        .expect("aggregate");
    assert_eq!(output.dim(), 2);
    assert!(output.values().iter().all(|value| value.is_finite()));
}
