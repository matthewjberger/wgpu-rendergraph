use crate::render_graph::{PassExecutionContext, PassNode, ResourceId};
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupLayout, Operations, RenderPassColorAttachment, RenderPipeline, Sampler,
};

const EDGE_DETECTION_SHADER: &str = "
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

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let texture_size = textureDimensions(input_texture);
    let texel_size = vec2<f32>(1.0 / f32(texture_size.x), 1.0 / f32(texture_size.y));

    let tl = luminance(textureSample(input_texture, input_sampler, in.uv + vec2<f32>(-texel_size.x, -texel_size.y)).rgb);
    let tm = luminance(textureSample(input_texture, input_sampler, in.uv + vec2<f32>(0.0, -texel_size.y)).rgb);
    let tr = luminance(textureSample(input_texture, input_sampler, in.uv + vec2<f32>(texel_size.x, -texel_size.y)).rgb);

    let ml = luminance(textureSample(input_texture, input_sampler, in.uv + vec2<f32>(-texel_size.x, 0.0)).rgb);
    let mr = luminance(textureSample(input_texture, input_sampler, in.uv + vec2<f32>(texel_size.x, 0.0)).rgb);

    let bl = luminance(textureSample(input_texture, input_sampler, in.uv + vec2<f32>(-texel_size.x, texel_size.y)).rgb);
    let bm = luminance(textureSample(input_texture, input_sampler, in.uv + vec2<f32>(0.0, texel_size.y)).rgb);
    let br = luminance(textureSample(input_texture, input_sampler, in.uv + vec2<f32>(texel_size.x, texel_size.y)).rgb);

    let gx = -tl - 2.0 * ml - bl + tr + 2.0 * mr + br;
    let gy = -tl - 2.0 * tm - tr + bl + 2.0 * bm + br;

    let edge_strength = sqrt(gx * gx + gy * gy);

    let original = textureSample(input_texture, input_sampler, in.uv).rgb;
    let edge_color = vec3<f32>(edge_strength);

    let result = mix(original, edge_color, 0.7);

    return vec4<f32>(result, 1.0);
}
";

pub struct EdgeDetectionPassData {
    pub pipeline: Arc<RenderPipeline>,
    pub blit_pipeline: Arc<RenderPipeline>,
    pub bind_group_layout: Arc<BindGroupLayout>,
    pub sampler: Arc<Sampler>,
}

pub struct EdgeDetectionPass {
    pub data: EdgeDetectionPassData,
    input: ResourceId,
    output: ResourceId,
    cached_bind_group: Option<BindGroup>,
    edge_detection_enabled: bool,
}

impl EdgeDetectionPass {
    pub fn new(data: EdgeDetectionPassData, input: ResourceId, output: ResourceId) -> Self {
        Self {
            data,
            input,
            output,
            cached_bind_group: None,
            edge_detection_enabled: false,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.edge_detection_enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.edge_detection_enabled
    }

    pub fn invalidate_bind_group(&mut self) {
        self.cached_bind_group = None;
    }

    pub fn create_pipeline(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
    ) -> (Arc<RenderPipeline>, Arc<BindGroupLayout>) {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Edge Detection Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(EDGE_DETECTION_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Edge Detection Bind Group Layout"),
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
            label: Some("Edge Detection Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Edge Detection Pipeline"),
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

impl PassNode for EdgeDetectionPass {
    fn name(&self) -> &str {
        "edge_detection_pass"
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
        if self.cached_bind_group.is_none() {
            let input_view = context
                .resources
                .get_texture_view(self.input)
                .expect("Input texture not allocated");

            self.cached_bind_group = Some(context.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("Edge Detection Bind Group"),
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
                    ],
                },
            ));
        }

        let (color_view, color_load_op, color_store_op) =
            context.resources.get_color_attachment(self.output);

        let mut render_pass = context
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: if self.edge_detection_enabled {
                    Some("Edge Detection Render Pass")
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

        let pipeline = if self.edge_detection_enabled {
            &self.data.pipeline
        } else {
            &self.data.blit_pipeline
        };

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, self.cached_bind_group.as_ref().unwrap(), &[]);
        render_pass.draw(0..3, 0..1);
    }
}
