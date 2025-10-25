use wgpu::{Operations, RenderPassColorAttachment};
use wgpu_render_graph::{PassExecutionContext, PassNode};

pub struct EguiPass;

impl EguiPass {
    pub fn new() -> Self {
        Self
    }
}

impl PassNode<crate::pass_configs::PassConfigs> for EguiPass {
    fn name(&self) -> &str {
        "egui_pass"
    }

    fn reads(&self) -> Vec<&str> {
        Vec::new()
    }

    fn writes(&self) -> Vec<&str> {
        vec!["color_target"]
    }

    fn reads_writes(&self) -> Vec<&str> {
        Vec::new()
    }

    fn execute<'r, 'e>(
        &mut self,
        context: PassExecutionContext<'r, 'e, crate::pass_configs::PassConfigs>,
    ) -> Vec<wgpu_render_graph::SubGraphRunCommand<'r>> {
        let config = &context.configs.egui;
        let (color_view, color_load_op, color_store_op) =
            context.get_color_attachment("color_target");

        let render_pass = context
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Egui Render Pass"),
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

        let mut render_pass_static = render_pass.forget_lifetime();
        if let Some(renderer) = &config.renderer {
            renderer.render(
                &mut render_pass_static,
                &config.paint_jobs,
                &config.screen_descriptor,
            );
        }

        context.into_sub_graph_commands()
    }
}
