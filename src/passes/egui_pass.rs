use crate::render_graph::{PassExecutionContext, PassNode, ResourceId};
use egui_wgpu::ScreenDescriptor;
use wgpu::{LoadOp, Operations, RenderPassColorAttachment};

pub struct EguiPassData {
    pub renderer: egui_wgpu::Renderer,
    pub paint_jobs: Vec<egui::ClippedPrimitive>,
    pub screen_descriptor: ScreenDescriptor,
}

pub struct EguiPass {
    pub data: EguiPassData,
    color_target: ResourceId,
}

impl EguiPass {
    pub fn new(data: EguiPassData, color_target: ResourceId) -> Self {
        Self { data, color_target }
    }
}

impl PassNode for EguiPass {
    fn name(&self) -> &str {
        "egui_pass"
    }

    fn reads(&self) -> Vec<ResourceId> {
        Vec::new()
    }

    fn writes(&self) -> Vec<ResourceId> {
        Vec::new()
    }

    fn reads_writes(&self) -> Vec<ResourceId> {
        vec![self.color_target]
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn execute(&mut self, context: PassExecutionContext) {
        let (color_view, _, color_store_op) =
            context.resources.get_color_attachment(self.color_target);

        let render_pass = context
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Egui Render Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Load,
                        store: color_store_op,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

        let mut render_pass_static = render_pass.forget_lifetime();
        self.data.renderer.render(
            &mut render_pass_static,
            &self.data.paint_jobs,
            &self.data.screen_descriptor,
        );
    }
}
