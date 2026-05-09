//! wgpu device/queue initialization.

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
}

impl fmt::Display for InitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no compatible GPU adapter found"),
            Self::DeviceRequest(e) => write!(f, "device request failed: {e}"),
        }
    }
}

impl std::error::Error for InitError {}

impl GpuContext {
    pub fn try_init() -> Result<Self, InitError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or(InitError::NoAdapter)?;

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
}
