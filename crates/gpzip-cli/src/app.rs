use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand, ValueEnum};
use gpzip_codec_cpu::CpuBackend;
use gpzip_codec_gpu::GpuBackend;
use gpzip_core::archive;
use gpzip_core::BackendRegistry;

use crate::progress::Progress;

const DEFAULT_CHUNK_BYTES: usize = 2 * 1024 * 1024;

#[derive(Parser, Debug)]
#[command(name = "gpzip", version, about = "GPU-accelerated archiver")]
struct Cli {
    /// Backend selection. `auto` prefers GPU when available.
    #[arg(long, value_enum, default_value_t = BackendChoice::Auto, global = true)]
    backend: BackendChoice,

    /// Worker threads for parallel chunk compression. 0 = number of CPU cores.
    /// `1` forces serial compression (one chunk in flight).
    #[arg(long, default_value_t = 0, global = true)]
    threads: usize,

    /// Per-chunk size in bytes for parallel compression. Each chunk is an
    /// independently-decodable gzip member or zstd frame, so output stays
    /// compatible with standard tools. Larger = better compression ratio,
    /// less parallelism. Default 2 MiB.
    #[arg(long, default_value_t = DEFAULT_CHUNK_BYTES, global = true)]
    chunk_size: usize,

    /// Suppress the progress bar.
    #[arg(short = 'q', long, global = true)]
    quiet: bool,

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
    let threads = if cli.threads == 0 {
        num_cpus::get().max(1)
    } else {
        cli.threads
    };
    if cli.chunk_size == 0 {
        return Err(anyhow!("--chunk-size must be > 0"));
    }
    let registry = build_registry(cli.backend, cli.chunk_size, threads);

    match cli.command {
        Command::A {
            archive: out,
            inputs,
            level,
        } => {
            let (sink, prog) = sink_for(cli.quiet, "packing");
            let res = archive::pack(&out, &inputs, gpzip_core::Level(level), &registry, sink);
            if let Some(p) = prog {
                p.finish();
            }
            res.map_err(|e| anyhow!(e))?;
        }
        Command::X {
            archive: ar,
            output,
        } => {
            let (sink, prog) = sink_for(cli.quiet, "extracting");
            let res = archive::unpack(&ar, &output, &registry, sink);
            if let Some(p) = prog {
                p.finish();
            }
            res.map_err(|e| anyhow!(e))?;
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

fn sink_for(quiet: bool, label: &str) -> (gpzip_core::ProgressSink, Option<Progress>) {
    if quiet {
        (gpzip_core::ProgressSink::noop(), None)
    } else {
        let p = Progress::new(label);
        (p.sink(), Some(p))
    }
}

fn build_registry(choice: BackendChoice, chunk_size: usize, threads: usize) -> BackendRegistry {
    let mut r = BackendRegistry::new();
    let cpu: Arc<dyn gpzip_core::CodecBackend> =
        Arc::new(CpuBackend::with_config(chunk_size, threads));

    match choice {
        BackendChoice::Cpu => {
            r.push(cpu);
        }
        BackendChoice::Gpu | BackendChoice::Auto => {
            if let Ok(gpu) = GpuBackend::try_init() {
                r.push(Arc::new(gpu));
            }
            r.push(cpu);
        }
    }
    r
}
