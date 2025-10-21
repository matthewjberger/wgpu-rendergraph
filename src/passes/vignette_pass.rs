use super::shader_common::FULLSCREEN_VERTEX_SHADER;
use crate::render_graph::{PassExecutionContext, PassNode, ResourceId};
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupLayout, Operations, RenderPassColorAttachment, RenderPipeline, Sampler,
};

const VIGNETTE_FRAGMENT_SHADER: &str = "
@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var input_sampler: sampler;

struct VignetteUniforms {
    strength: f32,
    radius: f32,
    color_tint: vec3<f32>,
    padding: f32,
};

@group(0) @binding(2)
var<uniform> uniforms: VignetteUniforms;

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(input_texture, input_sampler, in.uv).rgb;

    let center = vec2<f32>(0.5, 0.5);
    let distance = length(in.uv - center);

    let vignette = 1.0 - smoothstep(uniforms.radius, 1.0, distance * uniforms.strength);

    let tinted_color = mix(uniforms.color_tint, color, vignette);

    return vec4<f32>(tinted_color, 1.0);
}
";

pub struct VignettePassData {
    pub pipeline: Arc<RenderPipeline>,
    pub blit_pipeline: Arc<RenderPipeline>,
    pub bind_group_layout: Arc<BindGroupLayout>,
    pub sampler: Arc<Sampler>,
}

pub struct VignettePass {
    pub data: VignettePassData,
    input: ResourceId,
    output: ResourceId,
    cached_bind_group_with_vignette: Option<BindGroup>,
    cached_bind_group_without_vignette: Option<BindGroup>,
    uniform_buffer: Arc<wgpu::Buffer>,
    pub strength: f32,
    pub radius: f32,
    pub color_tint: [f32; 3],
    enabled: bool,
}

impl VignettePass {
    pub fn new(
        data: VignettePassData,
        input: ResourceId,
        output: ResourceId,
        uniform_buffer: Arc<wgpu::Buffer>,
    ) -> Self {
        Self {
            data,
            input,
            output,
            cached_bind_group_with_vignette: None,
            cached_bind_group_without_vignette: None,
            uniform_buffer,
            strength: 1.5,
            radius: 0.3,
            color_tint: [0.0, 0.0, 0.0],
            enabled: false,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn update_uniforms(&self, queue: &wgpu::Queue) {
        let uniform_data = [
            self.strength,
            self.radius,
            0.0,
            0.0,
            self.color_tint[0],
            self.color_tint[1],
            self.color_tint[2],
            0.0,
        ];
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&uniform_data));
    }

    pub fn invalidate_bind_groups(&mut self) {
        self.cached_bind_group_with_vignette = None;
        self.cached_bind_group_without_vignette = None;
    }

    pub fn create_pipeline(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
    ) -> (Arc<RenderPipeline>, Arc<BindGroupLayout>) {
        let shader_source = format!("{}\n{}", FULLSCREEN_VERTEX_SHADER, VIGNETTE_FRAGMENT_SHADER);

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vignette Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(shader_source)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Vignette Bind Group Layout"),
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
            label: Some("Vignette Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Vignette Pipeline"),
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

impl PassNode for VignettePass {
    fn name(&self) -> &str {
        "vignette_pass"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![self.input]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.output]
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn execute(&mut self, context: PassExecutionContext) {
        if self.cached_bind_group_with_vignette.is_none() {
            let input_view = context
                .resources
                .get_texture_view(self.input)
                .expect("Input texture not allocated");

            self.cached_bind_group_with_vignette = Some(context.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("Vignette Bind Group"),
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

        if self.cached_bind_group_without_vignette.is_none() {
            let input_view = context
                .resources
                .get_texture_view(self.input)
                .expect("Input texture not allocated");

            self.cached_bind_group_without_vignette = Some(context.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("Blit Bind Group (for disabled vignette)"),
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

        let (color_view, color_load_op, color_store_op) =
            context.resources.get_color_attachment(self.output);

        let mut render_pass = context
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: if self.enabled {
                    Some("Vignette Render Pass")
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

        let (pipeline, bind_group) = if self.enabled {
            (
                &self.data.pipeline,
                self.cached_bind_group_with_vignette.as_ref().unwrap(),
            )
        } else {
            (
                &self.data.blit_pipeline,
                self.cached_bind_group_without_vignette.as_ref().unwrap(),
            )
        };

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}
