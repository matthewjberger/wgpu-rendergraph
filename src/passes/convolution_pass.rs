use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupLayout, Operations, RenderPassColorAttachment, RenderPipeline, Sampler,
};
use wgpu_render_graph::{PassExecutionContext, PassNode};

const CONVOLUTION_SHADER: &str = "
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index & 1u) << 1u);
    let y = f32((vertex_index & 2u));
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    return out;
}

@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var input_sampler: sampler;

struct ConvolutionKernel {
    row0: vec4<f32>,
    row1: vec4<f32>,
    row2: vec4<f32>,
};

@group(0) @binding(2)
var<uniform> kernel: ConvolutionKernel;

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texture_size = textureDimensions(input_texture);
    let texel_size = 1.0 / vec2<f32>(f32(texture_size.x), f32(texture_size.y));

    var result = vec3<f32>(0.0);

    for (var y: i32 = -1; y <= 1; y++) {
        for (var x: i32 = -1; x <= 1; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            let sample_uv = in.uv + offset;
            let sample_color = textureSample(input_texture, input_sampler, sample_uv).rgb;

            var kernel_value = 0.0;
            if y == -1 {
                kernel_value = kernel.row0[x + 1];
            } else if y == 0 {
                kernel_value = kernel.row1[x + 1];
            } else {
                kernel_value = kernel.row2[x + 1];
            }

            result += sample_color * kernel_value;
        }
    }

    return vec4<f32>(result, 1.0);
}
";

pub struct ConvolutionPassData {
    pub pipeline: Arc<RenderPipeline>,
    pub blit_pipeline: Arc<RenderPipeline>,
    pub bind_group_layout: Arc<BindGroupLayout>,
    pub sampler: Arc<Sampler>,
}

pub struct ConvolutionPass {
    pub data: ConvolutionPassData,
    cached_bind_group_with_convolution: Option<BindGroup>,
    cached_bind_group_without_convolution: Option<BindGroup>,
    kernel_buffer: Arc<wgpu::Buffer>,
}

impl ConvolutionPass {
    pub fn new(data: ConvolutionPassData, kernel_buffer: Arc<wgpu::Buffer>) -> Self {
        Self {
            data,
            cached_bind_group_with_convolution: None,
            cached_bind_group_without_convolution: None,
            kernel_buffer,
        }
    }

    pub fn create_pipeline(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
    ) -> (Arc<RenderPipeline>, Arc<BindGroupLayout>) {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Convolution Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(CONVOLUTION_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Convolution Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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
            label: Some("Convolution Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Convolution Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vertex_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fragment_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        (Arc::new(pipeline), Arc::new(bind_group_layout))
    }
}

impl PassNode<crate::pass_configs::PassConfigs> for ConvolutionPass {
    fn name(&self) -> &str {
        "convolution_pass"
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
        let config = &configs.convolution;
        let kernel_data = [
            [config.kernel[0], config.kernel[1], config.kernel[2], 0.0],
            [config.kernel[3], config.kernel[4], config.kernel[5], 0.0],
            [config.kernel[6], config.kernel[7], config.kernel[8], 0.0],
        ];
        queue.write_buffer(&self.kernel_buffer, 0, bytemuck::cast_slice(&kernel_data));
    }

    fn invalidate_bind_groups(&mut self) {
        self.cached_bind_group_with_convolution = None;
        self.cached_bind_group_without_convolution = None;
    }

    fn execute(&mut self, context: PassExecutionContext<crate::pass_configs::PassConfigs>) {
        if self.cached_bind_group_with_convolution.is_none() {
            let input_view = context.get_texture_view("input");

            self.cached_bind_group_with_convolution = Some(context.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("Convolution Bind Group"),
                    layout: &self.data.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(input_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.data.sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.kernel_buffer.as_entire_binding(),
                        },
                    ],
                },
            ));
        }

        if self.cached_bind_group_without_convolution.is_none() {
            let input_view = context.get_texture_view("input");

            self.cached_bind_group_without_convolution = Some(context.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("Blit Bind Group (for disabled convolution)"),
                    layout: &self.data.blit_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(input_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.data.sampler),
                        },
                    ],
                },
            ));
        }

        let config = &context.configs.convolution;
        let (color_view, color_load_op, color_store_op) = context.get_color_attachment("output");

        let mut render_pass = context
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: if config.enabled {
                    Some("Convolution Render Pass")
                } else {
                    Some("Passthrough Blit Render Pass")
                },
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: Operations {
                        load: color_load_op,
                        store: color_store_op,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

        let (pipeline, bind_group) = if config.enabled {
            (
                &self.data.pipeline,
                self.cached_bind_group_with_convolution.as_ref().unwrap(),
            )
        } else {
            (
                &self.data.blit_pipeline,
                self.cached_bind_group_without_convolution.as_ref().unwrap(),
            )
        };

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}
