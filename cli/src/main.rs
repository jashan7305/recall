use clap::{Parser, Subcommand, command};

use cli::*;

#[derive(Parser)]
#[command(name = "recall")]
#[command(version = "0.1")]
#[command(about = "Local memory layer for LLM applications")]

struct  Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Init,
    Serve,
    Stats,
    Reset,
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Init => init(),
        Commands::Serve => serve(),
        Commands::Stats => stats(),
        Commands::Reset => reset(),
    }
}