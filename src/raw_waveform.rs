use crate::Result;
use crate::channel_estimation::{
    ChannelEstimate, ChannelEstimatorConfig, KnownReferenceChannelEstimator,
};
use crate::range_doppler::{
    CaCfar2DConfig, RangeDopplerConfig, RangeDopplerDetection, RangeDopplerMap,
    RangeDopplerProcessor, ca_cfar_2d,
};
use crate::raw_iq::RawIqCapture;

#[derive(Debug, Clone, Default)]
pub struct RawWaveformPipelineConfig {
    channel: ChannelEstimatorConfig,
    range_doppler: RangeDopplerConfig,
    cfar: CaCfar2DConfig,
}

impl RawWaveformPipelineConfig {
    pub fn new(
        channel: ChannelEstimatorConfig,
        range_doppler: RangeDopplerConfig,
        cfar: CaCfar2DConfig,
    ) -> Self {
        Self {
            channel,
            range_doppler,
            cfar,
        }
    }

    pub fn channel(self) -> ChannelEstimatorConfig {
        self.channel
    }

    pub fn range_doppler(self) -> RangeDopplerConfig {
        self.range_doppler
    }

    pub fn cfar(self) -> CaCfar2DConfig {
        self.cfar
    }
}

#[derive(Debug, Clone, Default)]
pub struct RawWaveformPipeline {
    config: RawWaveformPipelineConfig,
}

impl RawWaveformPipeline {
    pub fn new(config: RawWaveformPipelineConfig) -> Self {
        Self { config }
    }

    pub fn estimate_channel(
        &self,
        reference: &RawIqCapture,
        received: &RawIqCapture,
    ) -> Result<ChannelEstimate> {
        KnownReferenceChannelEstimator::new(self.config.channel)?.estimate(reference, received)
    }

    pub fn range_doppler(
        &self,
        reference: &RawIqCapture,
        received_pulses: &[RawIqCapture],
        pulse_repetition_interval_s: f64,
    ) -> Result<RangeDopplerMap> {
        RangeDopplerProcessor::new(self.config.range_doppler)?.process(
            reference,
            received_pulses,
            pulse_repetition_interval_s,
        )
    }

    pub fn detect_range_doppler(
        &self,
        map: &RangeDopplerMap,
    ) -> Result<Vec<RangeDopplerDetection>> {
        ca_cfar_2d(map, self.config.cfar)
    }
}
