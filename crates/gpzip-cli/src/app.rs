use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand, ValueEnum};
use gpzip_codec_cpu::CpuBackend;
use gpzip_codec_gpu::GpuBackend;
use gpzip_core::archive;
use gpzip_core::BackendRegistry;

#[derive(Parser, Debug)]
#[command(name = "gpzip", version, about = "GPU-accelerated archiver")]
struct Cli {
    /// Backend selection. `auto` prefers GPU when available.
    #[arg(long, value_enum, default_value_t = BackendChoice::Auto, global = true)]
    backend: BackendChoice,

    #[command(subcommand)]
    command: Command,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendChoice {
    Auto,
    Cpu,
    Gpu,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Add files/directories to an archive (a = add, like 7z).
    A {
        archive: PathBuf,
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// Compression level 0..=9 (default 5).
        #[arg(short = 'l', long, default_value_t = 5)]
        level: u8,
    },
    /// Extract an archive into a directory (x = extract).
    X {
        archive: PathBuf,
        /// Output directory. Defaults to current directory.
        #[arg(short = 'o', long, default_value = ".")]
        output: PathBuf,
    },
    /// List archive contents (l = list).
    L { archive: PathBuf },
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();
    let registry = build_registry(cli.backend);

    match cli.command {
        Command::A {
            archive: out,
            inputs,
            level,
        } => {
            archive::pack(
                &out,
                &inputs,
                gpzip_core::Level(level),
                &registry,
                gpzip_core::ProgressSink::noop(),
            )
            .map_err(|e| anyhow!(e))?;
        }
        Command::X {
            archive: ar,
            output,
        } => {
            archive::unpack(&ar, &output, &registry, gpzip_core::ProgressSink::noop())
                .map_err(|e| anyhow!(e))?;
        }
        Command::L { archive: ar } => {
            let entries = archive::list_archive(&ar, &registry).map_err(|e| anyhow!(e))?;
            for e in entries {
                let kind = if e.is_dir { "d" } else { "-" };
                println!("{kind} {:>12} {}", e.size, e.path.display());
            }
        }
    }
    Ok(())
}

fn build_registry(choice: BackendChoice) -> BackendRegistry {
    let mut r = BackendRegistry::new();
    let cpu: Arc<dyn gpzip_core::CodecBackend> = Arc::new(CpuBackend::new());

    match choice {
        BackendChoice::Cpu => {
            r.push(cpu);
        }
        BackendChoice::Gpu => {
            if let Ok(gpu) = GpuBackend::try_init() {
                r.push(Arc::new(gpu));
            }
            r.push(cpu);
        }
        BackendChoice::Auto => {
            // Prefer GPU when it initializes; fall through to CPU otherwise.
            if let Ok(gpu) = GpuBackend::try_init() {
                r.push(Arc::new(gpu));
            }
            r.push(cpu);
        }
    }
    r
}
