#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::env;

use anderion_cuas_ml::service::{ServiceState, build_router, load_perception_bundle};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = env::var("CUAS_MODEL_MANIFEST")?;
    let payload = env::var("CUAS_MODEL_PAYLOAD")?;
    let bind_addr = env::var("BIND_ADDR").unwrap_or_else(|_| "127.0.0.1:8080".to_string());
    let pipeline = load_perception_bundle(manifest, payload)?.into_pipeline()?;
    let app = build_router(ServiceState::new(pipeline));
    let listener = tokio::net::TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}
