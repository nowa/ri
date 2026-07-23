//! Refresh the embedded model catalog (`src/models_generated/*.json`) from
//! models.dev, under the conservative policy documented in
//! `ri_llm_provider::models_generator`.
//!
//! Usage:
//!   generate-models              # fetch + report (no writes)
//!   generate-models --write      # fetch + report + update changed files
//!   generate-models --source F   # use a local models.dev api.json snapshot

use ri_llm_provider::{
    fetch_models_dev_catalog, generated_catalog, parse_models_dev_catalog, plan_catalog_refresh,
    render_generated_provider_json,
};
use std::path::PathBuf;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let write = args.iter().any(|arg| arg == "--write");
    let source = args
        .iter()
        .position(|arg| arg == "--source")
        .and_then(|index| args.get(index + 1))
        .cloned();

    let models_dev = match source {
        Some(path) => {
            let body = std::fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("failed to read {path}: {error}"));
            let json = serde_json::from_str(&body)
                .unwrap_or_else(|error| panic!("{path} is not valid JSON: {error}"));
            parse_models_dev_catalog(&json)
        }
        None => match fetch_models_dev_catalog().await {
            Ok(catalog) => catalog,
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        },
    };

    let plan = plan_catalog_refresh(&models_dev, generated_catalog());
    println!("{}", plan.summary());
    if !write {
        if plan.has_changes() {
            println!("\nrun with --write to update src/models_generated/");
        }
        return;
    }

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/models_generated");
    let mut written = 0usize;
    for (provider_id, refresh) in &plan.providers {
        if refresh.updated.is_empty() && refresh.added.is_empty() {
            continue;
        }
        let path = dir.join(format!("{provider_id}.json"));
        std::fs::write(&path, render_generated_provider_json(&refresh.models))
            .unwrap_or_else(|error| panic!("failed to write {}: {error}", path.display()));
        written += 1;
    }
    println!("\nwrote {written} provider file(s); re-run the catalog tests before committing");
}
