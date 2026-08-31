use anderion_cuas_ml::{fuse_isac_dual_view, MicroDopplerFeatures};

#[test]
fn concentrated_hrrp_strengthens_confident_micro_doppler_view() {
    let micro=MicroDopplerFeatures{centroid_hz:0.0,bandwidth_hz:180.0,spectral_entropy:0.25,harmonicity:0.9,sideband_symmetry:0.9,confidence:0.85};
    let result=fuse_isac_dual_view(&micro,&[0.01,0.02,1.0,0.03,0.01]).unwrap();
    assert!(result.hrrp_concentration>0.8);
    assert!(result.fused_confidence>0.8);
}
