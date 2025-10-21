use crate::render_graph::{PassExecutionContext, PassNode, ResourceId};
use std::sync::Arc;
use wgpu::{
    BindGroup, Buffer, IndexFormat, Operations, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPipeline,
};

pub struct ScenePassData {
    pub pipeline: Arc<RenderPipeline>,
    pub vertex_buffer: Arc<Buffer>,
    pub index_buffer: Arc<Buffer>,
    pub index_count: u32,
    pub uniform_bind_group: Arc<BindGroup>,
    pub texture_bind_group: Arc<BindGroup>,
}

pub struct ScenePass {
    pub data: ScenePassData,
    color_output: ResourceId,
    depth_output: ResourceId,
}

impl ScenePass {
    pub fn new(data: ScenePassData, color_output: ResourceId, depth_output: ResourceId) -> Self {
        Self {
            data,
            color_output,
            depth_output,
        }
    }
}

impl PassNode for ScenePass {
    fn name(&self) -> &str {
        "scene_pass"
    }

    fn reads(&self) -> Vec<ResourceId> {
        vec![]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.color_output, self.depth_output]
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn execute(&mut self, context: PassExecutionContext) {
        let (color_view, color_load_op, color_store_op) =
            context.resources.get_color_attachment(self.color_output);
        let (depth_view, depth_load_op, depth_store_op) =
            context.resources.get_depth_attachment(self.depth_output);

        let mut render_pass = context
            .encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Scene Render Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: None,
                    ops: Operations {
                        load: color_load_op,
                        store: color_store_op,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(Operations {
                        load: depth_load_op,
                        store: depth_store_op,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

        render_pass.set_pipeline(&self.data.pipeline);
        render_pass.set_bind_group(0, self.data.uniform_bind_group.as_ref(), &[]);
        render_pass.set_bind_group(1, self.data.texture_bind_group.as_ref(), &[]);
        render_pass.set_vertex_buffer(0, self.data.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.data.index_buffer.slice(..), IndexFormat::Uint32);
        render_pass.draw_indexed(0..self.data.index_count, 0, 0..1);
    }
}
