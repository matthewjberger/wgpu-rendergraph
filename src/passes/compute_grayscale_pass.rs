use std::sync::Arc;
use wgpu::{BindGroup, BindGroupLayout, ComputePipeline};
use wgpu_render_graph::{PassExecutionContext, PassNode};

const COMPUTE_SHADER: &str = "
@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var output_texture: texture_storage_2d<rgba8unorm, write>;

struct Uniforms {
    enabled: u32,
    padding1: u32,
    padding2: u32,
    padding3: u32,
};

@group(0) @binding(2)
var<uniform> uniforms: Uniforms;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let texture_size = textureDimensions(input_texture);

    if (global_id.x >= texture_size.x || global_id.y >= texture_size.y) {
        return;
    }

    let coords = vec2<i32>(i32(global_id.x), i32(global_id.y));
    let color = textureLoad(input_texture, coords, 0);

    var output_color = color;

    if (uniforms.enabled != 0u) {
        let luminance = dot(color.rgb, vec3<f32>(0.299, 0.587, 0.114));
        output_color = vec4<f32>(luminance, luminance, luminance, color.a);
    }

    textureStore(output_texture, vec2<i32>(i32(global_id.x), i32(global_id.y)), output_color);
}
";

pub struct ComputeGrayscalePassData {
    pub pipeline: Arc<ComputePipeline>,
    pub bind_group_layout: Arc<BindGroupLayout>,
}

pub struct ComputeGrayscalePass {
    pub data: ComputeGrayscalePassData,
    cached_bind_group: Option<BindGroup>,
    uniform_buffer: Arc<wgpu::Buffer>,
}

impl ComputeGrayscalePass {
    pub fn new(data: ComputeGrayscalePassData, uniform_buffer: Arc<wgpu::Buffer>) -> Self {
        Self {
            data,
            cached_bind_group: None,
            uniform_buffer,
        }
    }

    pub fn create_pipeline(device: &wgpu::Device) -> (Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Grayscale Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(COMPUTE_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Grayscale Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Example Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Example Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        (Arc::new(pipeline), Arc::new(bind_group_layout))
    }
}

impl PassNode<crate::pass_configs::PassConfigs> for ComputeGrayscalePass {
    fn name(&self) -> &str {
        "compute_grayscale_pass"
    }

    fn reads(&self) -> Vec<&str> {
        vec!["input"]
    }

    fn writes(&self) -> Vec<&str> {
        vec!["output"]
    }

    fn prepare(
        &mut self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        configs: &crate::pass_configs::PassConfigs,
    ) {
        let enabled = if configs.compute_grayscale.enabled {
            1u32
        } else {
            0u32
        };
        let uniform_data = [enabled, 0u32, 0u32, 0u32];
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&uniform_data));
    }

    fn invalidate_bind_groups(&mut self) {
        self.cached_bind_group = None;
    }

    fn execute<'r, 'e>(
        &mut self,
        context: PassExecutionContext<'r, 'e, crate::pass_configs::PassConfigs>,
    ) -> wgpu_render_graph::Result<Vec<wgpu_render_graph::SubGraphRunCommand<'r>>> {
        if self.cached_bind_group.is_none() {
            let input_view = context.get_texture_view("input")?;
            let output_view = context.get_texture_view("output")?;

            self.cached_bind_group = Some(context.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("Compute Grayscale Bind Group"),
                    layout: &self.data.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(input_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(output_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.uniform_buffer.as_entire_binding(),
                        },
                    ],
                },
            ));
        }

        let texture_size = context.get_texture_size("input")?;

        let mut compute_pass = context
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Grayscale Pass"),
                timestamp_writes: None,
            });

        compute_pass.set_pipeline(&self.data.pipeline);
        compute_pass.set_bind_group(0, self.cached_bind_group.as_ref().unwrap(), &[]);

        let workgroup_count_x = texture_size.0.div_ceil(8);
        let workgroup_count_y = texture_size.1.div_ceil(8);
        compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);

        drop(compute_pass);

        Ok(context.into_sub_graph_commands())
    }
}
