#[tokio::main]
async fn main() {
    if let Err(error) = ri_llm_provider::run_cli_with_env_args().await {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
