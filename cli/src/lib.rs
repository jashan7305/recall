use serde::{Deserialize, Serialize};
use dirs;
use ctrlc;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::process::Command;
use std::{fs, u32};
use std::path::Path;

#[derive(Serialize, Deserialize)]
pub  struct Config {
    pub project_name: String,
    pub embedding_model_key: String,
    pub max_memories: u32,
    pub default_max_tokens: u32
}

impl Default for Config {
    fn default() -> Self {
        Self {
            project_name: "my_project".to_string(),
            embedding_model_key: "bge-small".to_string(),
            max_memories: 5000,
            default_max_tokens: 800
        }
    }
}

fn load_config() -> Config {
    let contents = fs::read_to_string("recall.toml")
    .expect("Could not find recall.toml");

    toml::from_str(&contents).expect("Invalid recall.toml")
}

fn ensure_qdrant_running(storage_path: &str) {
    println!("Checking Qdrant container...");

    let output = Command::new("docker")
        .args(["ps", "-a", "--filter", "name=recall-qdrant", "--format", "{{.Names}}"])
        .output()
        .expect("Failed to check Docker containers");

    let container_exists = String::from_utf8_lossy(&output.stdout)
        .lines()
        .any(|line| line == "recall-qdrant");

    if container_exists {
        println!("Qdrant container exists. Starting it...");

        Command::new("docker")
            .args(["start", "recall-qdrant"])
            .status()
            .expect("Failed to start Qdrant container");
    } else {
        println!("Creating Qdrant container...");

        Command::new("docker")
            .args([
                "run",
                "-d",
                "--name",
                "recall-qdrant",
                "-p",
                "6333:6333",
                "-v",
                &format!("{}:/qdrant/storage", storage_path),
                "qdrant/qdrant",
            ])
            .status()
            .expect("Failed to create Qdrant container");
    }

    println!("Qdrant ready");
}

pub fn init() {
    let path = Path::new("recall.toml");

    if path.exists() {
        println!("recall.toml already exists");
        return;
    }

    let project_name = std::env::current_dir()
        .expect("Could not determine current directory")
        .file_name()
        .expect("Could not determine project directory name")
        .to_string_lossy()
        .to_string();

    let config = Config {
        project_name,
        ..Config::default()
    };

    let toml = format!(
r#"# name of this project.
# used as the Qdrant collection name.
project_name = "{}"

# embedding model used for all stored vectors.
# changing this after storing data requires resetting the project.
# one of "bge-small", "bge-base", "minilm"
embedding_model_key = "{}"

# maximum number of stored memories and document chunks.
# least valuable memories are automatically pruned when this limit is reached.
max_memories = {}

# maximum number of tokens returned by a query.
# helps fit retrieved context into an LLM prompt.
default_max_tokens = {}
"#,
        config.project_name,
        config.embedding_model_key,
        config.max_memories,
        config.default_max_tokens
    );

    fs::write(path, toml).expect("Failed to write recall.toml");

    println!(
        "Initialized Recall project '{}'",
        config.project_name
    );
}

pub fn serve() {
    println!("Starting recall server...");

    let config = load_config();

    let home = dirs::home_dir().expect("Could not find home directory");
    let recall_home = home.join(".recall");
    let qdrant_storage = recall_home.join("qdrant_data");

    fs::create_dir_all(&qdrant_storage).unwrap();

    ensure_qdrant_running(qdrant_storage.to_str().unwrap());

    let mut command = Command::new("uv");

    command
        .args(["run", "python", "-m", "backend.src.recall_backend.main"])
        .env("PROJECT_NAME", &config.project_name)
        .env("EMBEDDING_MODEL_KEY", &config.embedding_model_key)
        .env("MAX_MEMORIES", config.max_memories.to_string())
        .env("DEFAULT_MAX_TOKENS", config.default_max_tokens.to_string());

    let mut child = command
        .spawn()
        .expect("Failed to start recall-backend");

    println!("Recall backend running");

    let child_pid = child.id();
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = Arc::clone(&shutdown);

    ctrlc::set_handler(move || {
        println!("\nShutting down Recall...");
        shutdown_clone.store(true, Ordering::SeqCst);

        // kill by PID directly
        #[cfg(unix)]
        unsafe {
            libc::kill(child_pid as i32, libc::SIGTERM);
        }
        #[cfg(windows)]
        {
            let _ = Command::new("taskkill")
                .args(["/PID", &child_pid.to_string(), "/F"])
                .status();
        }

        let _ = Command::new("docker")
            .args(["stop", "recall-qdrant"])
            .status();

        println!("Recall stopped");
    })
    .expect("Error setting Ctrl-C handler");

    // Main thread blocks here — no mutex held, handler can run freely
    let _ = child.wait();

    if shutdown.load(Ordering::SeqCst) {
        println!("Shutdown complete.");
    }
}

pub fn stats() {
    println!("Fetching Recall stats...");

    let output = Command::new("curl")
        .args(["-s", "http://localhost:8732/stats"])
        .output();

    match output {
        Ok(out) => {
            let body = String::from_utf8_lossy(&out.stdout);

            let parsed: serde_json::Value = match serde_json::from_str(&body) {
                Ok(v) => v,
                Err(_) => {
                    println!("Failed to parse stats response.");
                    return;
                }
            };

            println!("\nRecall Stats");
            println!("------------");

            println!(
                "Project: {}",
                parsed["project"].as_str().unwrap_or("unknown")
            );

            println!(
                "Embedding Model: {}",
                parsed["embedding_model"].as_str().unwrap_or("unknown")
            );

            println!(
                "Embedding Dimension: {}",
                parsed["embedding_dim"].as_i64().unwrap_or(0)
            );

            println!(
                "Memory Count: {}",
                parsed["memory_count"].as_i64().unwrap_or(0)
            );
        }
        Err(_) => {
            println!("Recall server is not running.");
        }
    }
}

pub fn reset() {
    println!("Resetting Recall memory...");

    let config = load_config();

    let url = format!(
        "http://localhost:6333/collections/{}",
        config.project_name
    );

    let status = Command::new("curl")
        .args(["-X", "DELETE", &url])
        .status();

    match status {
        Ok(_) => println!("Memory reset."),
        Err(_) => println!("Failed to reset memory."),
    }
}