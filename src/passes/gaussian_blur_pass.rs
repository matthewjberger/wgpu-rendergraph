use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupLayout, Operations, RenderPassColorAttachment, RenderPipeline, Sampler,
};
use wgpu_render_graph::{PassExecutionContext, PassNode};

const BLUR_SHADER: &str = "
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

struct Uniforms {
    direction: vec2<f32>,
};

@group(0) @binding(2)
var<uniform> uniforms: Uniforms;

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texture_size = textureDimensions(input_texture);
    let texel_size = 1.0 / vec2<f32>(f32(texture_size.x), f32(texture_size.y));

    let weights = array<f32, 5>(0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

    var result = textureSample(input_texture, input_sampler, in.uv).rgb * weights[0];

    for (var i: i32 = 1; i < 5; i++) {
        let offset = uniforms.direction * texel_size * f32(i);
        result += textureSample(input_texture, input_sampler, in.uv + offset).rgb * weights[i];
        result += textureSample(input_texture, input_sampler, in.uv - offset).rgb * weights[i];
    }

    return vec4<f32>(result, 1.0);
}
";

pub struct GaussianBlurPassData {
    pub pipeline: Arc<RenderPipeline>,
    pub blit_pipeline: Arc<RenderPipeline>,
    pub bind_group_layout: Arc<BindGroupLayout>,
    pub sampler: Arc<Sampler>,
}

pub struct GaussianBlurHorizontalPass {
    pub data: GaussianBlurPassData,
    cached_bind_group_with_blur: Option<BindGroup>,
    cached_bind_group_without_blur: Option<BindGroup>,
    uniform_buffer: Arc<wgpu::Buffer>,
}

impl GaussianBlurHorizontalPass {
    pub fn new(data: GaussianBlurPassData, uniform_buffer: Arc<wgpu::Buffer>) -> Self {
        Self {
            data,
            cached_bind_group_with_blur: None,
            cached_bind_group_without_blur: None,
            uniform_buffer,
        }
    }

    pub fn create_pipeline(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
    ) -> (Arc<RenderPipeline>, Arc<BindGroupLayout>) {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Gaussian Blur Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(BLUR_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Gaussian Blur Bind Group Layout"),
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
            label: Some("Gaussian Blur Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Gaussian Blur Pipeline"),
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

impl PassNode<crate::pass_configs::PassConfigs> for GaussianBlurHorizontalPass {
    fn name(&self) -> &str {
        "gaussian_blur_horizontal_pass"
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
        _configs: &crate::pass_configs::PassConfigs,
    ) {
        let direction = [1.0f32, 0.0f32];
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&direction));
    }

    fn invalidate_bind_groups(&mut self) {
        self.cached_bind_group_with_blur = None;
        self.cached_bind_group_without_blur = None;
    }

    fn execute<'r, 'e>(
        &mut self,
        context: PassExecutionContext<'r, 'e, crate::pass_configs::PassConfigs>,
    ) -> wgpu_render_graph::Result<Vec<wgpu_render_graph::SubGraphRunCommand<'r>>> {
        if self.cached_bind_group_with_blur.is_none() {
            let input_view = context.get_texture_view("input")?;

            self.cached_bind_group_with_blur = Some(context.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("Gaussian Blur Horizontal Bind Group"),
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
                            resource: self.uniform_buffer.as_entire_binding(),
                        },
                    ],
                },
            ));
        }

        if self.cached_bind_group_without_blur.is_none() {
            let input_view = context.get_texture_view("input")?;

            self.cached_bind_group_without_blur = Some(context.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("Blit Bind Group (for disabled blur)"),
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

        let config = &context.configs.gaussian_blur;
        let (color_view, color_load_op, color_store_op) = context.get_color_attachment("output")?;

        let mut render_pass = context
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: if config.enabled {
                    Some("Gaussian Blur Horizontal Render Pass")
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
                self.cached_bind_group_with_blur.as_ref().unwrap(),
            )
        } else {
            (
                &self.data.blit_pipeline,
                self.cached_bind_group_without_blur.as_ref().unwrap(),
            )
        };

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);

        drop(render_pass);

        Ok(context.into_sub_graph_commands())
    }
}

pub struct GaussianBlurVerticalPass {
    pub data: GaussianBlurPassData,
    cached_bind_group_with_blur: Option<BindGroup>,
    cached_bind_group_without_blur: Option<BindGroup>,
    uniform_buffer: Arc<wgpu::Buffer>,
}

impl GaussianBlurVerticalPass {
    pub fn new(data: GaussianBlurPassData, uniform_buffer: Arc<wgpu::Buffer>) -> Self {
        Self {
            data,
            cached_bind_group_with_blur: None,
            cached_bind_group_without_blur: None,
            uniform_buffer,
        }
    }
}

impl PassNode<crate::pass_configs::PassConfigs> for GaussianBlurVerticalPass {
    fn name(&self) -> &str {
        "gaussian_blur_vertical_pass"
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
        _configs: &crate::pass_configs::PassConfigs,
    ) {
        let direction = [0.0f32, 1.0f32];
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&direction));
    }

    fn invalidate_bind_groups(&mut self) {
        self.cached_bind_group_with_blur = None;
        self.cached_bind_group_without_blur = None;
    }

    fn execute<'r, 'e>(
        &mut self,
        context: PassExecutionContext<'r, 'e, crate::pass_configs::PassConfigs>,
    ) -> wgpu_render_graph::Result<Vec<wgpu_render_graph::SubGraphRunCommand<'r>>> {
        if self.cached_bind_group_with_blur.is_none() {
            let input_view = context.get_texture_view("input")?;

            self.cached_bind_group_with_blur = Some(context.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("Gaussian Blur Vertical Bind Group"),
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
                            resource: self.uniform_buffer.as_entire_binding(),
                        },
                    ],
                },
            ));
        }

        if self.cached_bind_group_without_blur.is_none() {
            let input_view = context.get_texture_view("input")?;

            self.cached_bind_group_without_blur = Some(context.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("Blit Bind Group (for disabled blur)"),
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

        let config = &context.configs.gaussian_blur;
        let (color_view, color_load_op, color_store_op) = context.get_color_attachment("output")?;

        let mut render_pass = context
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: if config.enabled {
                    Some("Gaussian Blur Vertical Render Pass")
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
                self.cached_bind_group_with_blur.as_ref().unwrap(),
            )
        } else {
            (
                &self.data.blit_pipeline,
                self.cached_bind_group_without_blur.as_ref().unwrap(),
            )
        };

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);
        drop(render_pass);

        Ok(context.into_sub_graph_commands())
    }
}
