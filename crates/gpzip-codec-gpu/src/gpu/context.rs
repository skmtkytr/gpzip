//! wgpu device/queue initialization.
//!
//! Adapter selection honors the `GPZIP_GPU` environment variable:
//!   - `high` (default) — request a `HighPerformance` adapter (typically dGPU)
//!   - `low`            — request a `LowPower` adapter (typically iGPU)
//!   - integer index    — pick the Nth adapter from `enumerate_adapters`
//!     (after backend filters), useful when both `high` and `low` resolve
//!     to the same device or when a multi-GPU host has more than one dGPU
//!
//! `GPZIP_GPU_BACKEND=vulkan|metal|dx12|gl|browser_webgpu` can additionally
//! restrict which backend wgpu enumerates. Defaults to all.

use std::fmt;

pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

#[derive(Debug)]
pub enum InitError {
    NoAdapter,
    DeviceRequest(wgpu::RequestDeviceError),
    BadIndex { requested: usize, available: usize },
}

impl fmt::Display for InitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no compatible GPU adapter found"),
            Self::DeviceRequest(e) => write!(f, "device request failed: {e}"),
            Self::BadIndex {
                requested,
                available,
            } => write!(
                f,
                "GPZIP_GPU={requested} but only {available} adapter(s) available"
            ),
        }
    }
}

impl std::error::Error for InitError {}

fn backends_from_env() -> wgpu::Backends {
    match std::env::var("GPZIP_GPU_BACKEND").as_deref() {
        Ok("vulkan") => wgpu::Backends::VULKAN,
        Ok("metal") => wgpu::Backends::METAL,
        Ok("dx12") => wgpu::Backends::DX12,
        Ok("gl") => wgpu::Backends::GL,
        Ok("browser_webgpu") => wgpu::Backends::BROWSER_WEBGPU,
        Ok("all") => wgpu::Backends::all(),
        // Default = PRIMARY (Vulkan / Metal / DX12). Excludes the GLES
        // backend, which on Wayland boxes intermittently fails to
        // initialize via EGL when no surface is attached and trips
        // wgpu's gles::egl into a panic. Tests on those hosts hit this.
        // Set GPZIP_GPU_BACKEND=all to opt back in.
        _ => wgpu::Backends::PRIMARY,
    }
}

impl GpuContext {
    pub fn try_init() -> Result<Self, InitError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: backends_from_env(),
            ..Default::default()
        });

        let adapter = pick_adapter(&instance)?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("gpzip-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))
        .map_err(InitError::DeviceRequest)?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }

    /// Human-readable name + type of the selected adapter. Useful for
    /// logging which device the bench actually ran on.
    pub fn adapter_label(&self) -> String {
        let info = self.adapter.get_info();
        format!(
            "{} ({:?}, backend={:?})",
            info.name, info.device_type, info.backend
        )
    }
}

fn pick_adapter(instance: &wgpu::Instance) -> Result<wgpu::Adapter, InitError> {
    match std::env::var("GPZIP_GPU").as_deref() {
        Ok("low") => request_with_pref(instance, wgpu::PowerPreference::LowPower),
        Ok("high") | Err(_) => request_with_pref(instance, wgpu::PowerPreference::HighPerformance),
        Ok(s) => match s.parse::<usize>() {
            Ok(idx) => pick_by_index(instance, idx),
            Err(_) => {
                tracing::warn!(
                    target: "gpzip-codec-gpu::context",
                    "GPZIP_GPU={s:?} not understood; using HighPerformance"
                );
                request_with_pref(instance, wgpu::PowerPreference::HighPerformance)
            }
        },
    }
}

fn request_with_pref(
    instance: &wgpu::Instance,
    pref: wgpu::PowerPreference,
) -> Result<wgpu::Adapter, InitError> {
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: pref,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or(InitError::NoAdapter)
}

fn pick_by_index(instance: &wgpu::Instance, idx: usize) -> Result<wgpu::Adapter, InitError> {
    let adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(backends_from_env());
    let n = adapters.len();
    adapters.into_iter().nth(idx).ok_or(InitError::BadIndex {
        requested: idx,
        available: n,
    })
}

/// List every adapter wgpu can see, with index, type, name, and backend.
/// Used by the CLI's `--list-gpus` debugging helper.
#[allow(dead_code)]
pub fn list_adapters() -> Vec<(usize, String)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: backends_from_env(),
        ..Default::default()
    });
    instance
        .enumerate_adapters(backends_from_env())
        .into_iter()
        .enumerate()
        .map(|(i, a)| {
            let info = a.get_info();
            (
                i,
                format!(
                    "{} ({:?}, backend={:?})",
                    info.name, info.device_type, info.backend
                ),
            )
        })
        .collect()
}
