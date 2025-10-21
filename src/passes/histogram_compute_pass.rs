use crate::render_graph::{PassExecutionContext, PassNode, ResourceId};
use std::sync::Arc;
use wgpu::{BindGroup, BindGroupLayout, ComputePipeline};

const HISTOGRAM_COMPUTE_SHADER: &str = "
@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var<storage, read_write> histogram: array<atomic<u32>, 256>;

@compute @workgroup_size(16, 16)
fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let texture_size = textureDimensions(input_texture);

    if global_id.x >= texture_size.x || global_id.y >= texture_size.y {
        return;
    }

    let color = textureLoad(input_texture, vec2<i32>(i32(global_id.x), i32(global_id.y)), 0).rgb;

    let luminance = dot(color, vec3<f32>(0.299, 0.587, 0.114));
    let bin = u32(clamp(luminance * 255.0, 0.0, 255.0));

    atomicAdd(&histogram[bin], 1u);
}
";

pub struct HistogramComputePassData {
    pub pipeline: Arc<ComputePipeline>,
    pub bind_group_layout: Arc<BindGroupLayout>,
}

pub struct HistogramComputePass {
    pub data: HistogramComputePassData,
    input: ResourceId,
    cached_bind_group: Option<BindGroup>,
    histogram_buffer: Arc<wgpu::Buffer>,
    histogram_readback_buffer: Arc<wgpu::Buffer>,
    enabled: bool,
    width: u32,
    height: u32,
}

impl HistogramComputePass {
    pub fn new(
        data: HistogramComputePassData,
        input: ResourceId,
        histogram_buffer: Arc<wgpu::Buffer>,
        histogram_readback_buffer: Arc<wgpu::Buffer>,
        width: u32,
        height: u32,
    ) -> Self {
        Self {
            data,
            input,
            cached_bind_group: None,
            histogram_buffer,
            histogram_readback_buffer,
            enabled: false,
            width,
            height,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn invalidate_bind_group(&mut self) {
        self.cached_bind_group = None;
    }

    pub fn update_dimensions(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    pub fn create_pipeline(device: &wgpu::Device) -> (Arc<ComputePipeline>, Arc<BindGroupLayout>) {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Histogram Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(HISTOGRAM_COMPUTE_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Histogram Compute Bind Group Layout"),
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Histogram Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Histogram Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("compute_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        (Arc::new(pipeline), Arc::new(bind_group_layout))
    }

    pub fn clear_histogram(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.histogram_buffer, 0, None);
    }
}

impl PassNode for HistogramComputePass {
    fn name(&self) -> &str {
        "histogram_compute_pass"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![self.input]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![]
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn execute(&mut self, context: PassExecutionContext) {
        if !self.enabled {
            return;
        }

        if self.cached_bind_group.is_none() {
            let input_view = context
                .resources
                .get_texture_view(self.input)
                .expect("Input texture not allocated");

            self.cached_bind_group = Some(context.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("Histogram Compute Bind Group"),
                    layout: &self.data.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(input_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.histogram_buffer.as_entire_binding(),
                        },
                    ],
                },
            ));
        }

        self.clear_histogram(context.encoder);

        let mut compute_pass = context
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Histogram Compute Pass"),
                timestamp_writes: None,
            });

        compute_pass.set_pipeline(&self.data.pipeline);
        compute_pass.set_bind_group(0, self.cached_bind_group.as_ref().unwrap(), &[]);

        let workgroups_x = self.width.div_ceil(16);
        let workgroups_y = self.height.div_ceil(16);

        compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);

        drop(compute_pass);

        context.encoder.copy_buffer_to_buffer(
            &self.histogram_buffer,
            0,
            &self.histogram_readback_buffer,
            0,
            (256 * std::mem::size_of::<u32>()) as u64,
        );
    }
}
