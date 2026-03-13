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

    let config = Config::default();
    let toml_str = toml::to_string_pretty(&config).unwrap();

    fs::write(path, toml_str).unwrap();

    println!("Created recall.toml");
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
        .args(["run", "recall-backend"])
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

        // Kill by PID directly — no lock contention
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
    println!("Stats...");
}

pub fn reset() {
    println!("Resetting...");
}