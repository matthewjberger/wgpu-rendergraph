use crate::render_graph::{PassExecutionContext, PassNode, ResourceId};
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupLayout, Operations, RenderPassColorAttachment, RenderPipeline, Sampler,
};

const POST_PROCESS_SHADER: &str = "
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
var hdr_texture: texture_2d<f32>;

@group(0) @binding(1)
var hdr_sampler: sampler;

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let hdr_color = textureSample(hdr_texture, hdr_sampler, in.uv).rgb;
    let tone_mapped = aces_tonemap(hdr_color);
    return vec4<f32>(tone_mapped, 1.0);
}
";

pub struct PostProcessPassData {
    pub pipeline: Arc<RenderPipeline>,
    pub bind_group_layout: Arc<BindGroupLayout>,
    pub sampler: Arc<Sampler>,
}

pub struct PostProcessPass {
    pub data: PostProcessPassData,
    hdr_input: ResourceId,
    color_output: ResourceId,
    cached_bind_group: Option<BindGroup>,
}

impl PostProcessPass {
    pub fn new(data: PostProcessPassData, hdr_input: ResourceId, color_output: ResourceId) -> Self {
        Self {
            data,
            hdr_input,
            color_output,
            cached_bind_group: None,
        }
    }

    pub fn invalidate_bind_group(&mut self) {
        self.cached_bind_group = None;
    }

    pub fn create_pipeline(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
    ) -> (Arc<RenderPipeline>, Arc<BindGroupLayout>) {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Post Process Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(POST_PROCESS_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Post Process Bind Group Layout"),
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
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Post Process Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Post Process Pipeline"),
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

impl PassNode for PostProcessPass {
    fn name(&self) -> &str {
        "post_process_pass"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![self.hdr_input]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.color_output]
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn execute(&mut self, context: PassExecutionContext) {
        if self.cached_bind_group.is_none() {
            let hdr_texture_view = context
                .resources
                .get_texture_view(self.hdr_input)
                .expect("HDR texture not allocated");

            self.cached_bind_group = Some(context.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("Post Process Bind Group"),
                    layout: &self.data.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(hdr_texture_view),
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
            context.resources.get_color_attachment(self.color_output);

        let mut render_pass = context
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Post Process Render Pass"),
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

        render_pass.set_pipeline(&self.data.pipeline);
        render_pass.set_bind_group(0, self.cached_bind_group.as_ref().unwrap(), &[]);
        render_pass.draw(0..3, 0..1);
    }
}
