use clap::{Parser, Subcommand};
use cli::*;

#[derive(Parser)]
#[command(name = "recall")]
#[command(version = "1.0.0")]
#[command(about = "Local memory layer for LLM applications")]
#[command(long_about = "Recall is a local-first memory engine for LLMs. \
It provides persistent semantic memory, context optimization, and retrieval for AI applications.")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a Recall project
    Init,

    /// Start Recall backend and Qdrant
    Serve,

    /// Show memory statistics
    Stats,

    /// Reset all stored memory
    Reset,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Init) => init(),
        Some(Commands::Serve) => serve(),
        Some(Commands::Stats) => stats(),
        Some(Commands::Reset) => reset(),
        None => {
            println!("Recall CLI\nRun `recall --help` for usage.");
        }
    }
}