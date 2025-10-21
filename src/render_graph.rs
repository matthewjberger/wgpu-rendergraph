#![allow(dead_code)]

use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Extent3d, StoreOp, Texture,
    TextureDescriptor, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId(pub u32);

impl ResourceId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

#[derive(Debug, Clone)]
pub struct RenderGraphTextureDescriptor {
    pub format: TextureFormat,
    pub width: u32,
    pub height: u32,
    pub usage: TextureUsages,
    pub sample_count: u32,
    pub mip_level_count: u32,
}

impl RenderGraphTextureDescriptor {
    pub fn to_wgpu_descriptor<'a>(&self, label: Option<&'a str>) -> TextureDescriptor<'a> {
        TextureDescriptor {
            label,
            size: Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: self.mip_level_count,
            sample_count: self.sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: self.usage,
            view_formats: &[],
        }
    }
}

#[derive(Debug, Clone)]
pub struct RenderGraphBufferDescriptor {
    pub size: u64,
    pub usage: BufferUsages,
    pub mapped_at_creation: bool,
}

impl RenderGraphBufferDescriptor {
    pub fn to_wgpu_descriptor<'a>(&self, label: Option<&'a str>) -> BufferDescriptor<'a> {
        BufferDescriptor {
            label,
            size: self.size,
            usage: self.usage,
            mapped_at_creation: self.mapped_at_creation,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ResourceType {
    ExternalColor {
        clear_color: Option<wgpu::Color>,
    },
    TransientColor {
        descriptor: RenderGraphTextureDescriptor,
        clear_color: Option<wgpu::Color>,
    },
    ExternalDepth {
        clear_depth: Option<f32>,
    },
    TransientDepth {
        descriptor: RenderGraphTextureDescriptor,
        clear_depth: Option<f32>,
    },
    ExternalBuffer,
    TransientBuffer {
        descriptor: RenderGraphBufferDescriptor,
    },
}

#[derive(Debug, Clone)]
pub struct ResourceDescriptor {
    pub name: String,
    pub resource_type: ResourceType,
    pub is_external: bool,
}

impl ResourceDescriptor {
    pub fn color_load_op(&self) -> wgpu::LoadOp<wgpu::Color> {
        match &self.resource_type {
            ResourceType::ExternalColor { clear_color }
            | ResourceType::TransientColor { clear_color, .. } => {
                if let Some(color) = clear_color {
                    wgpu::LoadOp::Clear(*color)
                } else {
                    wgpu::LoadOp::Load
                }
            }
            _ => panic!(
                "color_load_op called on non-color texture resource '{}'",
                self.name
            ),
        }
    }

    pub fn depth_load_op(&self) -> wgpu::LoadOp<f32> {
        match &self.resource_type {
            ResourceType::ExternalDepth { clear_depth }
            | ResourceType::TransientDepth { clear_depth, .. } => {
                if let Some(depth) = clear_depth {
                    wgpu::LoadOp::Clear(*depth)
                } else {
                    wgpu::LoadOp::Load
                }
            }
            _ => panic!(
                "depth_load_op called on non-depth texture resource '{}'",
                self.name
            ),
        }
    }
}

pub enum ResourceHandle {
    ExternalTexture {
        view: TextureView,
        store_op: StoreOp,
    },
    TransientTexture {
        texture: Arc<Texture>,
        view: TextureView,
        store_op: StoreOp,
    },
    ExternalBuffer {
        buffer: Arc<Buffer>,
    },
    TransientBuffer {
        buffer: Arc<Buffer>,
    },
}

impl ResourceHandle {
    pub fn view(&self) -> &TextureView {
        match self {
            ResourceHandle::ExternalTexture { view, .. } => view,
            ResourceHandle::TransientTexture { view, .. } => view,
            _ => panic!("view() called on buffer resource"),
        }
    }

    pub fn texture(&self) -> Option<&Arc<Texture>> {
        match self {
            ResourceHandle::ExternalTexture { .. } => None,
            ResourceHandle::TransientTexture { texture, .. } => Some(texture),
            _ => None,
        }
    }

    pub fn buffer(&self) -> Option<&Arc<Buffer>> {
        match self {
            ResourceHandle::ExternalBuffer { buffer } => Some(buffer),
            ResourceHandle::TransientBuffer { buffer } => Some(buffer),
            _ => None,
        }
    }

    pub fn store_op(&self) -> StoreOp {
        match self {
            ResourceHandle::ExternalTexture { store_op, .. } => *store_op,
            ResourceHandle::TransientTexture { store_op, .. } => *store_op,
            _ => panic!("store_op() called on buffer resource"),
        }
    }
}

pub struct RenderGraphResources {
    descriptors: HashMap<ResourceId, ResourceDescriptor>,
    handles: HashMap<ResourceId, ResourceHandle>,
    next_id: u32,
}

impl RenderGraphResources {
    pub fn new() -> Self {
        Self {
            descriptors: HashMap::new(),
            handles: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn register_external_resource(
        &mut self,
        name: String,
        resource_type: ResourceType,
    ) -> ResourceId {
        let id = ResourceId::new(self.next_id);
        self.next_id += 1;
        self.descriptors.insert(
            id,
            ResourceDescriptor {
                name,
                resource_type,
                is_external: true,
            },
        );
        id
    }

    pub fn set_external_texture(&mut self, id: ResourceId, view: TextureView) {
        self.handles.insert(
            id,
            ResourceHandle::ExternalTexture {
                view,
                store_op: StoreOp::Store,
            },
        );
    }

    pub fn set_external_buffer(&mut self, id: ResourceId, buffer: Arc<Buffer>) {
        self.handles
            .insert(id, ResourceHandle::ExternalBuffer { buffer });
    }

    pub fn register_transient_resource(
        &mut self,
        name: String,
        resource_type: ResourceType,
    ) -> ResourceId {
        let id = ResourceId::new(self.next_id);
        self.next_id += 1;
        self.descriptors.insert(
            id,
            ResourceDescriptor {
                name,
                resource_type,
                is_external: false,
            },
        );
        id
    }

    pub fn get_handle(&self, id: ResourceId) -> Option<&ResourceHandle> {
        self.handles.get(&id)
    }

    pub fn get_descriptor(&self, id: ResourceId) -> Option<&ResourceDescriptor> {
        self.descriptors.get(&id)
    }

    pub fn get_color_attachment(
        &self,
        id: ResourceId,
    ) -> (&TextureView, wgpu::LoadOp<wgpu::Color>, StoreOp) {
        let handle = self
            .get_handle(id)
            .unwrap_or_else(|| panic!("Color attachment {:?} not bound", id));
        let descriptor = self
            .get_descriptor(id)
            .unwrap_or_else(|| panic!("Color attachment {:?} descriptor not found", id));
        (handle.view(), descriptor.color_load_op(), handle.store_op())
    }

    pub fn get_depth_attachment(
        &self,
        id: ResourceId,
    ) -> (&TextureView, wgpu::LoadOp<f32>, StoreOp) {
        let handle = self
            .get_handle(id)
            .unwrap_or_else(|| panic!("Depth attachment {:?} not bound", id));
        let descriptor = self
            .get_descriptor(id)
            .unwrap_or_else(|| panic!("Depth attachment {:?} descriptor not found", id));
        (handle.view(), descriptor.depth_load_op(), handle.store_op())
    }

    pub fn get_texture(&self, id: ResourceId) -> Option<&Arc<Texture>> {
        self.get_handle(id).and_then(|handle| handle.texture())
    }

    pub fn get_texture_view(&self, id: ResourceId) -> Option<&TextureView> {
        self.get_handle(id).map(|handle| handle.view())
    }

    pub fn get_buffer(&self, id: ResourceId) -> Option<&Arc<Buffer>> {
        self.get_handle(id).and_then(|handle| handle.buffer())
    }

    pub fn resize_transient_resource(
        &mut self,
        device: &Device,
        id: ResourceId,
        width: u32,
        height: u32,
    ) {
        let (name, updated_descriptor) = {
            let descriptor = self.get_descriptor(id).unwrap_or_else(|| {
                panic!("Resource {:?} not found", id);
            });

            if descriptor.is_external {
                panic!("Cannot resize external resource '{}'", descriptor.name);
            }

            let name = descriptor.name.clone();

            let updated_descriptor = match &descriptor.resource_type {
                ResourceType::TransientColor {
                    descriptor: tex_desc,
                    clear_color,
                } => ResourceType::TransientColor {
                    descriptor: RenderGraphTextureDescriptor {
                        width,
                        height,
                        ..tex_desc.clone()
                    },
                    clear_color: *clear_color,
                },
                ResourceType::TransientDepth {
                    descriptor: tex_desc,
                    clear_depth,
                } => ResourceType::TransientDepth {
                    descriptor: RenderGraphTextureDescriptor {
                        width,
                        height,
                        ..tex_desc.clone()
                    },
                    clear_depth: *clear_depth,
                },
                ResourceType::TransientBuffer { .. } => {
                    panic!("Cannot resize buffer '{}' with width/height", name)
                }
                _ => panic!("Cannot resize non-transient resource '{}'", name),
            };

            (name, updated_descriptor)
        };

        self.descriptors.insert(
            id,
            ResourceDescriptor {
                name: name.clone(),
                resource_type: updated_descriptor.clone(),
                is_external: false,
            },
        );

        let texture_descriptor = match &updated_descriptor {
            ResourceType::TransientColor {
                descriptor: tex_desc,
                ..
            }
            | ResourceType::TransientDepth {
                descriptor: tex_desc,
                ..
            } => tex_desc.to_wgpu_descriptor(Some(&name)),
            _ => unreachable!(),
        };

        let texture = Arc::new(device.create_texture(&texture_descriptor));
        let view = texture.create_view(&TextureViewDescriptor::default());

        let store_op = self
            .handles
            .get(&id)
            .map(|h| h.store_op())
            .unwrap_or(StoreOp::Store);

        self.handles.insert(
            id,
            ResourceHandle::TransientTexture {
                texture,
                view,
                store_op,
            },
        );
    }

    pub fn resize_transient_buffer(&mut self, device: &Device, id: ResourceId, size: u64) {
        let (name, updated_descriptor) = {
            let descriptor = self.get_descriptor(id).unwrap_or_else(|| {
                panic!("Resource {:?} not found", id);
            });

            if descriptor.is_external {
                panic!("Cannot resize external resource '{}'", descriptor.name);
            }

            let name = descriptor.name.clone();

            let updated_descriptor = match &descriptor.resource_type {
                ResourceType::TransientBuffer {
                    descriptor: buf_desc,
                } => ResourceType::TransientBuffer {
                    descriptor: RenderGraphBufferDescriptor {
                        size,
                        ..buf_desc.clone()
                    },
                },
                _ => panic!("Cannot resize non-buffer resource '{}'", name),
            };

            (name, updated_descriptor)
        };

        self.descriptors.insert(
            id,
            ResourceDescriptor {
                name: name.clone(),
                resource_type: updated_descriptor.clone(),
                is_external: false,
            },
        );

        let buffer_descriptor = match &updated_descriptor {
            ResourceType::TransientBuffer {
                descriptor: buf_desc,
            } => buf_desc.to_wgpu_descriptor(Some(&name)),
            _ => unreachable!(),
        };

        let buffer = Arc::new(device.create_buffer(&buffer_descriptor));

        if let Some(handle) = self.handles.get_mut(&id) {
            *handle = ResourceHandle::TransientBuffer { buffer };
        }
    }

    pub fn allocate_transient_resources(
        &mut self,
        device: &Device,
        store_ops: &HashMap<ResourceId, StoreOp>,
    ) {
        let mut to_allocate = Vec::new();

        for (id, descriptor) in &self.descriptors {
            if !descriptor.is_external && !self.handles.contains_key(id) {
                to_allocate.push((*id, descriptor.clone()));
            }
        }

        for (id, descriptor) in to_allocate {
            match &descriptor.resource_type {
                ResourceType::TransientColor {
                    descriptor: tex_desc,
                    ..
                }
                | ResourceType::TransientDepth {
                    descriptor: tex_desc,
                    ..
                } => {
                    let texture_descriptor = tex_desc.to_wgpu_descriptor(Some(&descriptor.name));
                    let texture = Arc::new(device.create_texture(&texture_descriptor));
                    let view = texture.create_view(&TextureViewDescriptor::default());
                    let store_op = *store_ops.get(&id).unwrap_or(&StoreOp::Store);

                    self.handles.insert(
                        id,
                        ResourceHandle::TransientTexture {
                            texture,
                            view,
                            store_op,
                        },
                    );
                }
                ResourceType::TransientBuffer {
                    descriptor: buf_desc,
                } => {
                    let buffer_descriptor = buf_desc.to_wgpu_descriptor(Some(&descriptor.name));
                    let buffer = Arc::new(device.create_buffer(&buffer_descriptor));

                    self.handles
                        .insert(id, ResourceHandle::TransientBuffer { buffer });
                }
                _ => panic!(
                    "Attempted to allocate non-transient resource '{}'",
                    descriptor.name
                ),
            }
        }
    }

    pub fn clear_transient_handles(&mut self) {
        self.handles.retain(|id, _| {
            if let Some(descriptor) = self.descriptors.get(id) {
                descriptor.is_external
            } else {
                false
            }
        });
    }

    pub fn allocate_transient_resources_with_aliasing(
        &mut self,
        device: &Device,
        store_ops: &HashMap<ResourceId, StoreOp>,
        aliasing_info: &mut ResourceAliasingInfo,
    ) {
        for (pool_index, pool_slot) in aliasing_info.pools.iter_mut().enumerate() {
            if pool_slot.resource.is_some() {
                continue;
            }

            if let Some(descriptor_info) = &pool_slot.descriptor_info {
                let label = format!("pool_{}", pool_index);

                match descriptor_info {
                    PoolDescriptorInfo::Texture(tex_desc) => {
                        let texture_descriptor = tex_desc.to_wgpu_descriptor(Some(&label));
                        let texture = Arc::new(device.create_texture(&texture_descriptor));
                        pool_slot.resource = Some(PooledResource::Texture {
                            texture,
                            descriptor: tex_desc.clone(),
                        });
                    }
                    PoolDescriptorInfo::Buffer(buf_desc) => {
                        let buffer_descriptor = buf_desc.to_wgpu_descriptor(Some(&label));
                        let buffer = Arc::new(device.create_buffer(&buffer_descriptor));
                        pool_slot.resource = Some(PooledResource::Buffer {
                            buffer,
                            descriptor: buf_desc.clone(),
                        });
                    }
                }
            }
        }

        for (resource_id, descriptor) in &self.descriptors {
            if descriptor.is_external || self.handles.contains_key(resource_id) {
                continue;
            }

            if let Some(&pool_index) = aliasing_info.aliases.get(resource_id)
                && let Some(pool_slot) = aliasing_info.pools.get(pool_index)
            {
                match &pool_slot.resource {
                    Some(PooledResource::Texture { texture, .. }) => {
                        let view = texture.create_view(&TextureViewDescriptor::default());
                        let store_op = *store_ops.get(resource_id).unwrap_or(&StoreOp::Store);

                        self.handles.insert(
                            *resource_id,
                            ResourceHandle::TransientTexture {
                                texture: Arc::clone(texture),
                                view,
                                store_op,
                            },
                        );
                    }
                    Some(PooledResource::Buffer { buffer, .. }) => {
                        self.handles.insert(
                            *resource_id,
                            ResourceHandle::TransientBuffer {
                                buffer: Arc::clone(buffer),
                            },
                        );
                    }
                    None => {}
                }
            }
        }
    }
}

pub enum PassType {
    Render,
    Compute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueueType {
    Graphics,
    Compute,
}

pub struct PassExecutionContext<'a> {
    pub encoder: &'a mut CommandEncoder,
    pub resources: &'a RenderGraphResources,
    pub device: &'a Device,
}

pub trait PassNode: Send + Sync {
    fn name(&self) -> &str;
    fn pass_type(&self) -> PassType {
        PassType::Render
    }
    fn queue(&self) -> QueueType {
        QueueType::Graphics
    }
    fn enabled(&self) -> bool {
        true
    }
    fn reads(&self) -> Vec<ResourceId>;
    fn writes(&self) -> Vec<ResourceId>;
    fn reads_writes(&self) -> Vec<ResourceId> {
        Vec::new()
    }
    fn execute(&mut self, context: PassExecutionContext);
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

#[derive(Clone)]
pub struct GraphNode {
    pub name: String,
    pub reads: Vec<ResourceId>,
    pub writes: Vec<ResourceId>,
    pub reads_writes: Vec<ResourceId>,
    pub queue: QueueType,
}

pub struct ColorTextureBuilder<'a> {
    graph: &'a mut RenderGraph,
    name: String,
    descriptor: RenderGraphTextureDescriptor,
    clear_color: Option<wgpu::Color>,
}

impl<'a> ColorTextureBuilder<'a> {
    pub fn format(mut self, format: TextureFormat) -> Self {
        self.descriptor.format = format;
        self
    }

    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.descriptor.width = width;
        self.descriptor.height = height;
        self
    }

    pub fn usage(mut self, usage: TextureUsages) -> Self {
        self.descriptor.usage = usage;
        self
    }

    pub fn sample_count(mut self, count: u32) -> Self {
        self.descriptor.sample_count = count;
        self
    }

    pub fn mip_levels(mut self, levels: u32) -> Self {
        self.descriptor.mip_level_count = levels;
        self
    }

    pub fn clear_color(mut self, color: wgpu::Color) -> Self {
        self.clear_color = Some(color);
        self
    }

    fn validate(&self) {
        if self.descriptor.width == 0 || self.descriptor.height == 0 {
            panic!(
                "Texture '{}' has invalid dimensions: {}x{}. Width and height must be > 0",
                self.name, self.descriptor.width, self.descriptor.height
            );
        }
        if self.descriptor.sample_count == 0 {
            panic!(
                "Texture '{}' has invalid sample_count: {}. Must be >= 1",
                self.name, self.descriptor.sample_count
            );
        }
        if self.descriptor.mip_level_count == 0 {
            panic!(
                "Texture '{}' has invalid mip_level_count: {}. Must be >= 1",
                self.name, self.descriptor.mip_level_count
            );
        }
    }

    pub fn external(self) -> ResourceId {
        self.graph.resources.register_external_resource(
            self.name,
            ResourceType::ExternalColor {
                clear_color: self.clear_color,
            },
        )
    }

    pub fn transient(self) -> ResourceId {
        self.validate();
        self.graph.resources.register_transient_resource(
            self.name,
            ResourceType::TransientColor {
                descriptor: self.descriptor,
                clear_color: self.clear_color,
            },
        )
    }
}

pub struct DepthTextureBuilder<'a> {
    graph: &'a mut RenderGraph,
    name: String,
    descriptor: RenderGraphTextureDescriptor,
    clear_depth: Option<f32>,
}

impl<'a> DepthTextureBuilder<'a> {
    pub fn format(mut self, format: TextureFormat) -> Self {
        self.descriptor.format = format;
        self
    }

    pub fn size(mut self, width: u32, height: u32) -> Self {
        self.descriptor.width = width;
        self.descriptor.height = height;
        self
    }

    pub fn usage(mut self, usage: TextureUsages) -> Self {
        self.descriptor.usage = usage;
        self
    }

    pub fn sample_count(mut self, count: u32) -> Self {
        self.descriptor.sample_count = count;
        self
    }

    pub fn mip_levels(mut self, levels: u32) -> Self {
        self.descriptor.mip_level_count = levels;
        self
    }

    pub fn clear_depth(mut self, depth: f32) -> Self {
        self.clear_depth = Some(depth);
        self
    }

    fn validate(&self) {
        if self.descriptor.width == 0 || self.descriptor.height == 0 {
            panic!(
                "Texture '{}' has invalid dimensions: {}x{}. Width and height must be > 0",
                self.name, self.descriptor.width, self.descriptor.height
            );
        }
        if self.descriptor.sample_count == 0 {
            panic!(
                "Texture '{}' has invalid sample_count: {}. Must be >= 1",
                self.name, self.descriptor.sample_count
            );
        }
        if self.descriptor.mip_level_count == 0 {
            panic!(
                "Texture '{}' has invalid mip_level_count: {}. Must be >= 1",
                self.name, self.descriptor.mip_level_count
            );
        }
    }

    pub fn external(self) -> ResourceId {
        self.graph.resources.register_external_resource(
            self.name,
            ResourceType::ExternalDepth {
                clear_depth: self.clear_depth,
            },
        )
    }

    pub fn transient(self) -> ResourceId {
        self.validate();
        self.graph.resources.register_transient_resource(
            self.name,
            ResourceType::TransientDepth {
                descriptor: self.descriptor,
                clear_depth: self.clear_depth,
            },
        )
    }
}

pub struct BufferBuilder<'a> {
    graph: &'a mut RenderGraph,
    name: String,
    descriptor: RenderGraphBufferDescriptor,
}

impl<'a> BufferBuilder<'a> {
    pub fn size(mut self, size: u64) -> Self {
        self.descriptor.size = size;
        self
    }

    pub fn usage(mut self, usage: BufferUsages) -> Self {
        self.descriptor.usage = usage;
        self
    }

    pub fn mapped_at_creation(mut self, mapped: bool) -> Self {
        self.descriptor.mapped_at_creation = mapped;
        self
    }

    fn validate(&self) {
        if self.descriptor.size == 0 {
            panic!(
                "Buffer '{}' has invalid size: {}. Size must be > 0",
                self.name, self.descriptor.size
            );
        }
    }

    pub fn external(self) -> ResourceId {
        self.graph
            .resources
            .register_external_resource(self.name, ResourceType::ExternalBuffer)
    }

    pub fn transient(self) -> ResourceId {
        self.validate();
        self.graph.resources.register_transient_resource(
            self.name,
            ResourceType::TransientBuffer {
                descriptor: self.descriptor,
            },
        )
    }
}

pub struct RenderGraph {
    graph: DiGraph<GraphNode, ResourceId>,
    pass_nodes: HashMap<String, NodeIndex>,
    passes: HashMap<String, Box<dyn PassNode>>,
    resources: RenderGraphResources,
    execution_order: Vec<NodeIndex>,
    store_ops: HashMap<ResourceId, StoreOp>,
    statistics: Vec<PassStatistics>,
    profiling_enabled: bool,
    aliasing_info: Option<ResourceAliasingInfo>,
    aliasing_enabled: bool,
    needs_recompile: bool,
}

impl RenderGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            pass_nodes: HashMap::new(),
            passes: HashMap::new(),
            resources: RenderGraphResources::new(),
            execution_order: Vec::new(),
            store_ops: HashMap::new(),
            statistics: Vec::new(),
            profiling_enabled: false,
            aliasing_info: None,
            aliasing_enabled: true,
            needs_recompile: true,
        }
    }

    pub fn add_pass(&mut self, pass: Box<dyn PassNode>) -> NodeIndex {
        let name = pass.name().to_string();
        let reads = pass.reads();
        let writes = pass.writes();
        let reads_writes = pass.reads_writes();
        let queue = pass.queue();

        let graph_node = GraphNode {
            name: name.clone(),
            reads,
            writes,
            reads_writes,
            queue,
        };

        let index = self.graph.add_node(graph_node);
        self.pass_nodes.insert(name.clone(), index);
        self.passes.insert(name, pass);
        self.needs_recompile = true;
        index
    }

    pub fn remove_pass(&mut self, pass_name: &str) -> Option<Box<dyn PassNode>> {
        let node_index = self.pass_nodes.remove(pass_name)?;
        self.graph.remove_node(node_index);
        let pass = self.passes.remove(pass_name);
        self.needs_recompile = true;
        pass
    }

    pub fn add_color_texture(&mut self, name: &str) -> ColorTextureBuilder<'_> {
        ColorTextureBuilder {
            graph: self,
            name: name.to_string(),
            descriptor: RenderGraphTextureDescriptor {
                format: TextureFormat::Rgba8UnormSrgb,
                width: 1,
                height: 1,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                sample_count: 1,
                mip_level_count: 1,
            },
            clear_color: None,
        }
    }

    pub fn add_depth_texture(&mut self, name: &str) -> DepthTextureBuilder<'_> {
        DepthTextureBuilder {
            graph: self,
            name: name.to_string(),
            descriptor: RenderGraphTextureDescriptor {
                format: TextureFormat::Depth32Float,
                width: 1,
                height: 1,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                sample_count: 1,
                mip_level_count: 1,
            },
            clear_depth: None,
        }
    }

    pub fn add_buffer(&mut self, name: &str) -> BufferBuilder<'_> {
        BufferBuilder {
            graph: self,
            name: name.to_string(),
            descriptor: RenderGraphBufferDescriptor {
                size: 256,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        }
    }

    fn build_dependency_edges(&mut self) {
        let mut resource_writers: HashMap<ResourceId, NodeIndex> = HashMap::new();

        let node_indices: Vec<NodeIndex> = self.graph.node_indices().collect();
        let mut edges_to_add: Vec<(NodeIndex, NodeIndex, ResourceId)> = Vec::new();

        for &node_index in &node_indices {
            let node = &self.graph[node_index];
            let reads = node.reads.clone();
            let writes = node.writes.clone();
            let reads_writes = node.reads_writes.clone();

            for &read_resource in &reads {
                if let Some(&writer_index) = resource_writers.get(&read_resource)
                    && !self.graph.contains_edge(writer_index, node_index)
                {
                    edges_to_add.push((writer_index, node_index, read_resource));
                }
            }

            for &rw_resource in &reads_writes {
                if let Some(&writer_index) = resource_writers.get(&rw_resource)
                    && !self.graph.contains_edge(writer_index, node_index)
                {
                    edges_to_add.push((writer_index, node_index, rw_resource));
                }
            }

            for &write_resource in &writes {
                resource_writers.insert(write_resource, node_index);
            }

            for &rw_resource in &reads_writes {
                resource_writers.insert(rw_resource, node_index);
            }
        }

        for (from, to, resource) in edges_to_add {
            self.graph.add_edge(from, to, resource);
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        self.validate_multiple_writers()?;
        self.validate_missing_resources()?;
        self.validate_no_cycles()?;
        Ok(())
    }

    fn validate_multiple_writers(&self) -> Result<(), String> {
        let mut resource_writers: HashMap<ResourceId, Vec<String>> = HashMap::new();

        for node_index in self.graph.node_indices() {
            let node = &self.graph[node_index];

            for &write_resource in &node.writes {
                resource_writers
                    .entry(write_resource)
                    .or_default()
                    .push(node.name.clone());
            }
            for &rw_resource in &node.reads_writes {
                resource_writers
                    .entry(rw_resource)
                    .or_default()
                    .push(node.name.clone());
            }
        }

        for (resource_id, writers) in &resource_writers {
            if writers.len() > 1 {
                let writer_indices: Vec<NodeIndex> = self
                    .graph
                    .node_indices()
                    .filter(|&idx| writers.contains(&self.graph[idx].name))
                    .collect();

                let has_valid_dependencies = if writer_indices.len() == 2 {
                    self.graph
                        .contains_edge(writer_indices[0], writer_indices[1])
                        || self
                            .graph
                            .contains_edge(writer_indices[1], writer_indices[0])
                } else {
                    let mut all_connected = true;
                    for i in 0..writer_indices.len() {
                        let mut has_connection = false;
                        for j in 0..writer_indices.len() {
                            if i != j
                                && (self
                                    .graph
                                    .contains_edge(writer_indices[i], writer_indices[j])
                                    || self
                                        .graph
                                        .contains_edge(writer_indices[j], writer_indices[i])
                                    || petgraph::algo::has_path_connecting(
                                        &self.graph,
                                        writer_indices[i],
                                        writer_indices[j],
                                        None,
                                    )
                                    || petgraph::algo::has_path_connecting(
                                        &self.graph,
                                        writer_indices[j],
                                        writer_indices[i],
                                        None,
                                    ))
                            {
                                has_connection = true;
                                break;
                            }
                        }
                        if !has_connection {
                            all_connected = false;
                            break;
                        }
                    }
                    all_connected
                };

                if !has_valid_dependencies {
                    return Err(format!(
                        "Resource {:?} written by multiple passes without dependency: {:?}",
                        resource_id, writers
                    ));
                }
            }
        }

        Ok(())
    }

    fn validate_missing_resources(&self) -> Result<(), String> {
        let mut resource_writers: HashMap<ResourceId, Vec<String>> = HashMap::new();
        let mut resource_readers: HashMap<ResourceId, Vec<String>> = HashMap::new();

        for node_index in self.graph.node_indices() {
            let node = &self.graph[node_index];

            for &read_resource in &node.reads {
                resource_readers
                    .entry(read_resource)
                    .or_default()
                    .push(node.name.clone());
            }

            for &write_resource in &node.writes {
                resource_writers
                    .entry(write_resource)
                    .or_default()
                    .push(node.name.clone());
            }

            for &rw_resource in &node.reads_writes {
                resource_readers
                    .entry(rw_resource)
                    .or_default()
                    .push(node.name.clone());
                resource_writers
                    .entry(rw_resource)
                    .or_default()
                    .push(node.name.clone());
            }
        }

        for (resource_id, readers) in &resource_readers {
            let descriptor = self
                .resources
                .get_descriptor(*resource_id)
                .ok_or_else(|| format!("Resource {:?} used but not registered", resource_id))?;

            if !descriptor.is_external && !resource_writers.contains_key(resource_id) {
                return Err(format!(
                    "Resource '{}' ({:?}) is read by {:?} but never written",
                    descriptor.name, resource_id, readers
                ));
            }
        }

        Ok(())
    }

    fn validate_no_cycles(&self) -> Result<(), String> {
        if petgraph::algo::toposort(&self.graph, None).is_err() {
            return Err("Render graph contains cycles".to_string());
        }
        Ok(())
    }

    fn compute_resource_lifetimes(&self, execution_order: &[NodeIndex]) -> Vec<ResourceLifetime> {
        let mut lifetimes: HashMap<ResourceId, ResourceLifetime> = HashMap::new();

        for (pass_index, &node_index) in execution_order.iter().enumerate() {
            let node = &self.graph[node_index];

            for &resource_id in &node.writes {
                lifetimes.entry(resource_id).or_insert(ResourceLifetime {
                    resource_id,
                    first_use: pass_index,
                    last_use: pass_index,
                });
            }

            for &resource_id in &node.reads {
                let lifetime = lifetimes.entry(resource_id).or_insert(ResourceLifetime {
                    resource_id,
                    first_use: pass_index,
                    last_use: pass_index,
                });
                lifetime.last_use = pass_index;
            }

            for &resource_id in &node.reads_writes {
                let lifetime = lifetimes.entry(resource_id).or_insert(ResourceLifetime {
                    resource_id,
                    first_use: pass_index,
                    last_use: pass_index,
                });
                lifetime.last_use = pass_index;
            }
        }

        lifetimes
            .into_iter()
            .filter(|(id, _)| {
                if let Some(desc) = self.resources.get_descriptor(*id) {
                    !desc.is_external
                } else {
                    false
                }
            })
            .map(|(_, lifetime)| lifetime)
            .collect()
    }

    fn can_alias_textures(
        desc1: &RenderGraphTextureDescriptor,
        desc2: &RenderGraphTextureDescriptor,
    ) -> bool {
        desc1.format == desc2.format
            && desc1.width == desc2.width
            && desc1.height == desc2.height
            && desc1.sample_count == desc2.sample_count
            && desc1.mip_level_count == desc2.mip_level_count
            && desc1.usage == desc2.usage
    }

    fn can_alias_buffers(
        desc1: &RenderGraphBufferDescriptor,
        desc2: &RenderGraphBufferDescriptor,
    ) -> bool {
        desc1.size >= desc2.size && desc1.usage == desc2.usage
    }

    fn compute_resource_aliasing(
        &self,
        mut lifetimes: Vec<ResourceLifetime>,
    ) -> ResourceAliasingInfo {
        lifetimes.sort_by_key(|lt| lt.first_use);

        let mut aliasing_info = ResourceAliasingInfo {
            aliases: HashMap::new(),
            pools: Vec::new(),
        };

        let mut available_pools: BinaryHeap<PoolHeapEntry> = BinaryHeap::new();

        for lifetime in lifetimes {
            let descriptor = self.resources.get_descriptor(lifetime.resource_id).unwrap();

            let mut reused_candidates = Vec::new();
            while let Some(entry) = available_pools.peek() {
                if entry.lifetime_end < lifetime.first_use {
                    reused_candidates.push(available_pools.pop().unwrap());
                } else {
                    break;
                }
            }

            let mut assigned_slot = None;

            for candidate in reused_candidates.iter_mut() {
                let can_reuse = match (&candidate.descriptor_info, &descriptor.resource_type) {
                    (
                        PoolDescriptorInfo::Texture(pool_desc),
                        ResourceType::TransientColor {
                            descriptor: res_desc,
                            ..
                        }
                        | ResourceType::TransientDepth {
                            descriptor: res_desc,
                            ..
                        },
                    ) => Self::can_alias_textures(pool_desc, res_desc),
                    (
                        PoolDescriptorInfo::Buffer(pool_desc),
                        ResourceType::TransientBuffer {
                            descriptor: res_desc,
                        },
                    ) => Self::can_alias_buffers(pool_desc, res_desc),
                    _ => false,
                };

                if can_reuse {
                    if let (
                        PoolDescriptorInfo::Buffer(pool_desc),
                        ResourceType::TransientBuffer {
                            descriptor: res_desc,
                        },
                    ) = (&mut candidate.descriptor_info, &descriptor.resource_type)
                        && res_desc.size > pool_desc.size
                    {
                        *pool_desc = res_desc.clone();
                    }

                    let pool_slot = &mut aliasing_info.pools[candidate.pool_index];
                    pool_slot.lifetime_end = lifetime.last_use;
                    pool_slot.descriptor_info = Some(candidate.descriptor_info.clone());

                    candidate.lifetime_end = lifetime.last_use;
                    assigned_slot = Some(candidate.pool_index);
                    break;
                }
            }

            for candidate in reused_candidates {
                available_pools.push(candidate);
            }

            if assigned_slot.is_none() {
                let slot_index = aliasing_info.pools.len();

                let descriptor_info = match &descriptor.resource_type {
                    ResourceType::TransientColor {
                        descriptor: tex_desc,
                        ..
                    }
                    | ResourceType::TransientDepth {
                        descriptor: tex_desc,
                        ..
                    } => PoolDescriptorInfo::Texture(tex_desc.clone()),
                    ResourceType::TransientBuffer {
                        descriptor: buf_desc,
                    } => PoolDescriptorInfo::Buffer(buf_desc.clone()),
                    _ => continue,
                };

                aliasing_info.pools.push(PoolSlot {
                    resource: None,
                    descriptor_info: Some(descriptor_info.clone()),
                    lifetime_end: lifetime.last_use,
                });

                available_pools.push(PoolHeapEntry {
                    pool_index: slot_index,
                    lifetime_end: lifetime.last_use,
                    descriptor_info,
                });

                assigned_slot = Some(slot_index);
            }

            aliasing_info
                .aliases
                .insert(lifetime.resource_id, assigned_slot.unwrap());
        }

        aliasing_info
    }

    fn compute_store_ops(&self, execution_order: &[NodeIndex]) -> HashMap<ResourceId, StoreOp> {
        let mut store_ops = HashMap::new();

        for &resource_id in self.resources.descriptors.keys() {
            let descriptor = self.resources.get_descriptor(resource_id).unwrap();

            if descriptor.is_external {
                store_ops.insert(resource_id, StoreOp::Store);
                continue;
            }

            let mut is_read = false;
            for &node_index in execution_order {
                let node = &self.graph[node_index];

                if node.reads.contains(&resource_id) || node.reads_writes.contains(&resource_id) {
                    is_read = true;
                    break;
                }
            }

            let store_op = if is_read {
                StoreOp::Store
            } else {
                StoreOp::Discard
            };
            store_ops.insert(resource_id, store_op);
        }

        store_ops
    }

    pub fn compile(&mut self) -> Result<(), String> {
        self.build_dependency_edges();
        self.validate()?;

        self.execution_order = petgraph::algo::toposort(&self.graph, None)
            .map_err(|_| "Render graph contains cycles")?;

        self.store_ops = self.compute_store_ops(&self.execution_order);

        let lifetimes = self.compute_resource_lifetimes(&self.execution_order);
        self.aliasing_info = Some(self.compute_resource_aliasing(lifetimes));

        self.needs_recompile = false;
        Ok(())
    }

    fn recompile_if_needed(&mut self) {
        if !self.needs_recompile {
            return;
        }

        let edge_indices: Vec<_> = self.graph.edge_indices().collect();
        for edge_index in edge_indices {
            self.graph.remove_edge(edge_index);
        }

        self.build_dependency_edges();

        self.execution_order =
            petgraph::algo::toposort(&self.graph, None).expect("Render graph contains cycles");

        self.store_ops = self.compute_store_ops(&self.execution_order);

        let lifetimes = self.compute_resource_lifetimes(&self.execution_order);
        self.aliasing_info = Some(self.compute_resource_aliasing(lifetimes));

        self.needs_recompile = false;
    }

    pub fn execute(&mut self, device: &Device, encoder: &mut CommandEncoder) {
        self.recompile_if_needed();

        if self.aliasing_enabled {
            if self.aliasing_info.is_none() {
                let lifetimes = self.compute_resource_lifetimes(&self.execution_order);
                self.aliasing_info = Some(self.compute_resource_aliasing(lifetimes));
            }

            if let Some(aliasing_info) = &mut self.aliasing_info {
                self.resources.allocate_transient_resources_with_aliasing(
                    device,
                    &self.store_ops,
                    aliasing_info,
                );
            }
        } else {
            self.resources
                .allocate_transient_resources(device, &self.store_ops);
        }

        if self.profiling_enabled {
            self.statistics.clear();
        }

        for &node_index in &self.execution_order {
            let node = &self.graph[node_index];

            if let Some(pass) = self.passes.get_mut(&node.name) {
                let enabled = pass.enabled();

                if self.profiling_enabled {
                    let start = Instant::now();

                    if enabled {
                        let context = PassExecutionContext {
                            encoder,
                            resources: &self.resources,
                            device,
                        };
                        pass.execute(context);
                    }

                    let execution_time = start.elapsed();
                    self.statistics.push(PassStatistics {
                        pass_name: node.name.clone(),
                        execution_time,
                        enabled,
                    });
                } else if enabled {
                    let context = PassExecutionContext {
                        encoder,
                        resources: &self.resources,
                        device,
                    };
                    pass.execute(context);
                }
            }
        }
    }

    pub fn enable_profiling(&mut self, enabled: bool) {
        self.profiling_enabled = enabled;
    }

    pub fn get_statistics(&self) -> &[PassStatistics] {
        &self.statistics
    }

    pub fn enable_aliasing(&mut self, enabled: bool) {
        self.aliasing_enabled = enabled;
    }

    pub fn resources_mut(&mut self) -> &mut RenderGraphResources {
        &mut self.resources
    }

    pub fn get_execution_order(&self) -> Vec<String> {
        self.execution_order
            .iter()
            .map(|&index| self.graph[index].name.clone())
            .collect()
    }

    pub fn get_pass_mut<T: PassNode + 'static>(&mut self, pass_name: &str) -> Option<&mut T> {
        self.passes
            .get_mut(pass_name)
            .and_then(|pass| pass.as_any_mut().downcast_mut::<T>())
    }

    pub fn resize_transient_resource(
        &mut self,
        device: &Device,
        id: ResourceId,
        width: u32,
        height: u32,
    ) {
        self.resources
            .resize_transient_resource(device, id, width, height);

        if self.aliasing_enabled {
            self.aliasing_info = None;
            self.resources.clear_transient_handles();
        }
    }
}

#[derive(Debug, Clone)]
pub struct PassStatistics {
    pub pass_name: String,
    pub execution_time: Duration,
    pub enabled: bool,
}

#[derive(Debug, Clone)]
struct ResourceLifetime {
    resource_id: ResourceId,
    first_use: usize,
    last_use: usize,
}

enum PooledResource {
    Texture {
        texture: Arc<Texture>,
        descriptor: RenderGraphTextureDescriptor,
    },
    Buffer {
        buffer: Arc<Buffer>,
        descriptor: RenderGraphBufferDescriptor,
    },
}

#[derive(Clone)]
enum PoolDescriptorInfo {
    Texture(RenderGraphTextureDescriptor),
    Buffer(RenderGraphBufferDescriptor),
}

pub(crate) struct PoolSlot {
    resource: Option<PooledResource>,
    descriptor_info: Option<PoolDescriptorInfo>,
    lifetime_end: usize,
}

#[derive(Clone)]
struct PoolHeapEntry {
    pool_index: usize,
    lifetime_end: usize,
    descriptor_info: PoolDescriptorInfo,
}

impl PartialEq for PoolHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.lifetime_end == other.lifetime_end
    }
}

impl Eq for PoolHeapEntry {}

impl PartialOrd for PoolHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PoolHeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.lifetime_end.cmp(&self.lifetime_end)
    }
}

pub(crate) struct ResourceAliasingInfo {
    aliases: HashMap<ResourceId, usize>,
    pools: Vec<PoolSlot>,
}
