use super::shader_common::FULLSCREEN_VERTEX_SHADER;
use std::sync::Arc;
use wgpu::{
    BindGroup, BindGroupLayout, Operations, RenderPassColorAttachment, RenderPipeline, Sampler,
};
use wgpu_render_graph::{PassExecutionContext, PassNode};

const GRAYSCALE_FRAGMENT_SHADER: &str = "
@group(0) @binding(0)
var input_texture: texture_2d<f32>;

@group(0) @binding(1)
var input_sampler: sampler;

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(input_texture, input_sampler, in.uv).rgb;
    let luminance = dot(color, vec3<f32>(0.299, 0.587, 0.114));
    return vec4<f32>(vec3<f32>(luminance), 1.0);
}
";

pub struct GrayscalePassData {
    pub pipeline: Arc<RenderPipeline>,
    pub blit_pipeline: Arc<RenderPipeline>,
    pub bind_group_layout: Arc<BindGroupLayout>,
    pub sampler: Arc<Sampler>,
}

pub struct GrayscalePass {
    pub data: GrayscalePassData,
    cached_bind_group_with_grayscale: Option<BindGroup>,
    cached_bind_group_without_grayscale: Option<BindGroup>,
}

impl GrayscalePass {
    pub fn new(data: GrayscalePassData) -> Self {
        Self {
            data,
            cached_bind_group_with_grayscale: None,
            cached_bind_group_without_grayscale: None,
        }
    }

    pub fn create_pipeline(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
    ) -> (Arc<RenderPipeline>, Arc<BindGroupLayout>) {
        let shader_source = format!(
            "{}\n{}",
            FULLSCREEN_VERTEX_SHADER, GRAYSCALE_FRAGMENT_SHADER
        );

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Grayscale Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(shader_source)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Grayscale Bind Group Layout"),
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
            label: Some("Grayscale Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Grayscale Pipeline"),
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

impl PassNode<crate::pass_configs::PassConfigs> for GrayscalePass {
    fn name(&self) -> &str {
        "grayscale_pass"
    }

    fn reads(&self) -> Vec<&str> {
        vec!["input"]
    }

    fn writes(&self) -> Vec<&str> {
        vec!["output"]
    }

    fn invalidate_bind_groups(&mut self) {
        self.cached_bind_group_with_grayscale = None;
        self.cached_bind_group_without_grayscale = None;
    }

    fn execute<'r, 'e>(
        &mut self,
        context: PassExecutionContext<'r, 'e, crate::pass_configs::PassConfigs>,
    ) -> wgpu_render_graph::Result<Vec<wgpu_render_graph::SubGraphRunCommand<'r>>> {
        if self.cached_bind_group_with_grayscale.is_none() {
            let input_view = context.get_texture_view("input")?;

            self.cached_bind_group_with_grayscale = Some(context.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("Grayscale Bind Group"),
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

        if self.cached_bind_group_without_grayscale.is_none() {
            let input_view = context.get_texture_view("input")?;

            self.cached_bind_group_without_grayscale = Some(context.device.create_bind_group(
                &wgpu::BindGroupDescriptor {
                    label: Some("Blit Bind Group (for disabled grayscale)"),
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

        let config = &context.configs.grayscale;
        let (color_view, color_load_op, color_store_op) = context.get_color_attachment("output")?;

        let mut render_pass = context
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: if config.enabled {
                    Some("Grayscale Render Pass")
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
                self.cached_bind_group_with_grayscale.as_ref().unwrap(),
            )
        } else {
            (
                &self.data.blit_pipeline,
                self.cached_bind_group_without_grayscale.as_ref().unwrap(),
            )
        };

        render_pass.set_pipeline(pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);
        drop(render_pass);

        Ok(context.into_sub_graph_commands())
    }
}
