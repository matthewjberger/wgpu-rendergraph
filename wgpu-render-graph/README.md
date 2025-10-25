# wgpu-render-graph

A dependency-based render graph for wgpu with automatic resource aliasing and pass culling.

## Features

- **Automatic Dependency Tracking**: Declare what each pass reads/writes, and the graph automatically determines execution order
- **Resource Aliasing**: Transient resources automatically reuse GPU memory when their lifetimes don't overlap
- **Dead Pass Culling**: Passes that don't contribute to external outputs are automatically skipped
- **Store Operation Optimization**: Automatically determines when render attachments can use `StoreOp::Discard` to save bandwidth
- **Conditional Execution via Dependencies**: Toggle features (bloom, shadows, etc.) by conditionally declaring dependencies - unused passes auto-cull
- **Resource Version Tracking**: Automatically invalidates bind groups when resources are resized or recreated

## Example

```rust
use wgpu_render_graph::*;

// Create graph
let mut graph = RenderGraph::new();

// Register resources
let hdr_color = graph.add_color_texture("hdr_color")
    .format(TextureFormat::Rgba16Float)
    .size(1920, 1080)
    .transient();

let depth = graph.add_depth_texture("depth")
    .clear_depth(1.0)
    .transient();

let swapchain = graph.add_color_texture("swapchain")
    .external();

// Add passes
graph.add_pass(Box::new(ScenePass {
    hdr_color,
    depth,
}));

graph.add_pass(Box::new(TonemapPass {
    hdr_input: hdr_color,
    output: swapchain,
}));

// Compile and execute
graph.compile()?;
let command_buffers = graph.execute(&device);
queue.submit(command_buffers);
```

## Pass Implementation

```rust
struct ScenePass {
    hdr_color: ResourceId,
    depth: ResourceId,
}

impl PassNode for ScenePass {
    fn name(&self) -> &str { "scene" }

    fn reads(&self) -> Vec<ResourceId> {
        vec![]
    }

    fn writes(&self) -> Vec<ResourceId> {
        vec![self.hdr_color, self.depth]
    }

    fn execute(&mut self, context: PassExecutionContext) {
        let (view, load_op, store_op) = context.resources
            .get_color_attachment(self.hdr_color);

        let mut pass = context.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                ops: wgpu::Operations { load: load_op, store: store_op },
                resolve_target: None,
            })],
            // ...
        });

        // Render scene
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
```

## Conditional Feature Toggling

Toggle rendering features by conditionally declaring dependencies:

```rust
impl FinalCompositePass {
    fn reads(&self) -> Vec<ResourceId> {
        let mut reads = vec![self.hdr_color];

        // Only read bloom if enabled - if not read, entire bloom pipeline auto-culls
        if self.bloom_enabled {
            reads.push(self.bloom_texture);
        }

        reads
    }
}
```

## License

MIT OR Apache-2.0
