use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandBuffer, CommandEncoder, Device, Extent3d,
    StoreOp, Texture, TextureDescriptor, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor,
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
    versions: HashMap<ResourceId, u64>,
    next_id: u32,
}

impl RenderGraphResources {
    pub fn new() -> Self {
        Self {
            descriptors: HashMap::new(),
            handles: HashMap::new(),
            versions: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn get_version(&self, id: ResourceId) -> u64 {
        *self.versions.get(&id).unwrap_or(&0)
    }

    fn increment_version(&mut self, id: ResourceId) {
        let version = self.versions.entry(id).or_insert(0);
        *version += 1;
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

        let load_op = match &descriptor.resource_type {
            ResourceType::ExternalColor { clear_color }
            | ResourceType::TransientColor { clear_color, .. } => {
                if let Some(color) = clear_color {
                    wgpu::LoadOp::Clear(*color)
                } else {
                    wgpu::LoadOp::Load
                }
            }
            _ => panic!(
                "get_color_attachment called on non-color texture resource '{}'",
                descriptor.name
            ),
        };

        (handle.view(), load_op, handle.store_op())
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

        let load_op = match &descriptor.resource_type {
            ResourceType::ExternalDepth { clear_depth }
            | ResourceType::TransientDepth { clear_depth, .. } => {
                if let Some(depth) = clear_depth {
                    wgpu::LoadOp::Clear(*depth)
                } else {
                    wgpu::LoadOp::Load
                }
            }
            _ => panic!(
                "get_depth_attachment called on non-depth texture resource '{}'",
                descriptor.name
            ),
        };

        (handle.view(), load_op, handle.store_op())
    }

    pub fn get_texture_view(&self, id: ResourceId) -> Option<&TextureView> {
        self.get_handle(id).map(|handle| handle.view())
    }

    pub fn update_transient_descriptor(&mut self, id: ResourceId, width: u32, height: u32) {
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

        self.descriptors.insert(
            id,
            ResourceDescriptor {
                name,
                resource_type: updated_descriptor,
                is_external: false,
            },
        );
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
                        pool_slot.resource = Some(PooledResource::Texture { texture });
                    }
                    PoolDescriptorInfo::Buffer(buf_desc) => {
                        let buffer_descriptor = buf_desc.to_wgpu_descriptor(Some(&label));
                        let buffer = Arc::new(device.create_buffer(&buffer_descriptor));
                        pool_slot.resource = Some(PooledResource::Buffer { buffer });
                    }
                }
            }
        }

        let mut allocated_resources = Vec::new();

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
                        allocated_resources.push(*resource_id);
                    }
                    Some(PooledResource::Buffer { buffer, .. }) => {
                        self.handles.insert(
                            *resource_id,
                            ResourceHandle::TransientBuffer {
                                buffer: Arc::clone(buffer),
                            },
                        );
                        allocated_resources.push(*resource_id);
                    }
                    None => {}
                }
            }
        }

        for resource_id in allocated_resources {
            self.increment_version(resource_id);
        }
    }
}

impl Default for RenderGraphResources {
    fn default() -> Self {
        Self::new()
    }
}

pub struct PassExecutionContext<'a, C = ()> {
    pub encoder: &'a mut CommandEncoder,
    pub resources: &'a RenderGraphResources,
    pub device: &'a Device,
    slot_mappings: &'a HashMap<String, ResourceId>,
    pub configs: &'a C,
}

impl<'a, C> PassExecutionContext<'a, C> {
    pub fn get_slot(&self, slot: &str) -> ResourceId {
        *self
            .slot_mappings
            .get(slot)
            .unwrap_or_else(|| panic!("Slot '{}' not found in pass resource mappings", slot))
    }

    pub fn get_texture_view(&self, slot: &str) -> &'a wgpu::TextureView {
        let resource_id = self.get_slot(slot);
        self.resources
            .get_texture_view(resource_id)
            .unwrap_or_else(|| panic!("Texture view for slot '{}' not allocated", slot))
    }

    pub fn get_color_attachment(
        &self,
        slot: &str,
    ) -> (
        &'a wgpu::TextureView,
        wgpu::LoadOp<wgpu::Color>,
        wgpu::StoreOp,
    ) {
        let resource_id = self.get_slot(slot);
        self.resources.get_color_attachment(resource_id)
    }

    pub fn get_depth_attachment(
        &self,
        slot: &str,
    ) -> (&'a wgpu::TextureView, wgpu::LoadOp<f32>, wgpu::StoreOp) {
        let resource_id = self.get_slot(slot);
        self.resources.get_depth_attachment(resource_id)
    }
}

pub trait PassNode<C = ()>: Send + Sync {
    fn name(&self) -> &str;
    fn reads(&self) -> Vec<&str>;
    fn writes(&self) -> Vec<&str>;
    fn reads_writes(&self) -> Vec<&str> {
        Vec::new()
    }
    fn prepare(&mut self, _device: &Device, _queue: &wgpu::Queue, _configs: &C) {}
    fn invalidate_bind_groups(&mut self) {}
    fn execute(&mut self, context: PassExecutionContext<C>);
}

pub struct GraphNode<C> {
    pub name: String,
    pub reads: Vec<ResourceId>,
    pub writes: Vec<ResourceId>,
    pub reads_writes: Vec<ResourceId>,
    pub pass: Box<dyn PassNode<C>>,
}

pub struct ColorTextureBuilder<'a, C = ()> {
    graph: &'a mut RenderGraph<C>,
    name: String,
    descriptor: RenderGraphTextureDescriptor,
    clear_color: Option<wgpu::Color>,
}

impl<'a, C> ColorTextureBuilder<'a, C> {
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

    pub fn external(self) -> ResourceId {
        self.graph.resources.register_external_resource(
            self.name,
            ResourceType::ExternalColor {
                clear_color: self.clear_color,
            },
        )
    }

    pub fn transient(self) -> ResourceId {
        self.graph.resources.register_transient_resource(
            self.name,
            ResourceType::TransientColor {
                descriptor: self.descriptor,
                clear_color: self.clear_color,
            },
        )
    }
}

pub struct DepthTextureBuilder<'a, C = ()> {
    graph: &'a mut RenderGraph<C>,
    name: String,
    descriptor: RenderGraphTextureDescriptor,
    clear_depth: Option<f32>,
}

impl<'a, C> DepthTextureBuilder<'a, C> {
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

    pub fn external(self) -> ResourceId {
        self.graph.resources.register_external_resource(
            self.name,
            ResourceType::ExternalDepth {
                clear_depth: self.clear_depth,
            },
        )
    }

    pub fn transient(self) -> ResourceId {
        self.graph.resources.register_transient_resource(
            self.name,
            ResourceType::TransientDepth {
                descriptor: self.descriptor,
                clear_depth: self.clear_depth,
            },
        )
    }
}

pub struct BufferBuilder<'a, C = ()> {
    graph: &'a mut RenderGraph<C>,
    name: String,
    descriptor: RenderGraphBufferDescriptor,
}

impl<'a, C> BufferBuilder<'a, C> {
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

    pub fn external(self) -> ResourceId {
        self.graph
            .resources
            .register_external_resource(self.name, ResourceType::ExternalBuffer)
    }

    pub fn transient(self) -> ResourceId {
        self.graph.resources.register_transient_resource(
            self.name,
            ResourceType::TransientBuffer {
                descriptor: self.descriptor,
            },
        )
    }
}

pub struct RenderGraph<C = ()> {
    graph: DiGraph<GraphNode<C>, ResourceId>,
    pass_nodes: HashMap<String, NodeIndex>,
    pass_resource_mappings: HashMap<String, HashMap<String, ResourceId>>,
    resources: RenderGraphResources,
    execution_order: Vec<NodeIndex>,
    store_ops: HashMap<ResourceId, StoreOp>,
    aliasing_info: Option<ResourceAliasingInfo>,
    needs_recompile: bool,
    needs_resource_reallocation: bool,
    culled_passes: std::collections::HashSet<NodeIndex>,
    resource_versions: HashMap<ResourceId, u64>,
}

impl<C> RenderGraph<C> {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            pass_nodes: HashMap::new(),
            pass_resource_mappings: HashMap::new(),
            resources: RenderGraphResources::new(),
            execution_order: Vec::new(),
            store_ops: HashMap::new(),
            aliasing_info: None,
            needs_recompile: true,
            needs_resource_reallocation: false,
            culled_passes: std::collections::HashSet::new(),
            resource_versions: HashMap::new(),
        }
    }

    pub fn add_pass(
        &mut self,
        pass: Box<dyn PassNode<C>>,
        slot_mappings: &[(&str, ResourceId)],
    ) -> NodeIndex {
        let name = pass.name().to_string();
        let slot_names_reads = pass.reads();
        let slot_names_writes = pass.writes();
        let slot_names_reads_writes = pass.reads_writes();

        let mappings: HashMap<String, ResourceId> = slot_mappings
            .iter()
            .map(|(slot, resource_id)| (slot.to_string(), *resource_id))
            .collect();

        let reads: Vec<ResourceId> = slot_names_reads
            .iter()
            .map(|slot| {
                *mappings.get(*slot).unwrap_or_else(|| {
                    panic!("Pass '{}': slot '{}' not provided in mappings", name, slot)
                })
            })
            .collect();

        let writes: Vec<ResourceId> = slot_names_writes
            .iter()
            .map(|slot| {
                *mappings.get(*slot).unwrap_or_else(|| {
                    panic!("Pass '{}': slot '{}' not provided in mappings", name, slot)
                })
            })
            .collect();

        let reads_writes: Vec<ResourceId> = slot_names_reads_writes
            .iter()
            .map(|slot| {
                *mappings.get(*slot).unwrap_or_else(|| {
                    panic!("Pass '{}': slot '{}' not provided in mappings", name, slot)
                })
            })
            .collect();

        let graph_node = GraphNode {
            name: name.clone(),
            reads,
            writes,
            reads_writes,
            pass,
        };

        let index = self.graph.add_node(graph_node);
        self.pass_nodes.insert(name.clone(), index);
        self.pass_resource_mappings.insert(name, mappings);
        self.needs_recompile = true;
        index
    }

    pub fn add_color_texture(&mut self, name: &str) -> ColorTextureBuilder<'_, C> {
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

    pub fn add_depth_texture(&mut self, name: &str) -> DepthTextureBuilder<'_, C> {
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

    pub fn add_buffer(&mut self, name: &str) -> BufferBuilder<'_, C> {
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
            && desc1.usage.contains(desc2.usage)
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
                    let needs_new_resource = if let (
                        PoolDescriptorInfo::Buffer(pool_desc),
                        ResourceType::TransientBuffer {
                            descriptor: res_desc,
                        },
                    ) =
                        (&mut candidate.descriptor_info, &descriptor.resource_type)
                    {
                        if res_desc.size > pool_desc.size {
                            *pool_desc = res_desc.clone();
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };

                    let pool_slot = &mut aliasing_info.pools[candidate.pool_index];
                    pool_slot.lifetime_end = lifetime.last_use;
                    pool_slot.descriptor_info = Some(candidate.descriptor_info.clone());

                    if needs_new_resource {
                        pool_slot.resource = None;
                    }

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
        let mut last_read: HashMap<ResourceId, usize> = HashMap::new();

        for (index, &node_index) in execution_order.iter().enumerate().rev() {
            let node = &self.graph[node_index];

            for &resource_id in node.reads.iter().chain(&node.reads_writes) {
                last_read.entry(resource_id).or_insert(index);
            }
        }

        let mut store_ops = HashMap::new();

        for (index, &node_index) in execution_order.iter().enumerate() {
            let node = &self.graph[node_index];

            for &resource_id in node.writes.iter().chain(&node.reads_writes) {
                let descriptor = self.resources.get_descriptor(resource_id).unwrap();

                if descriptor.is_external {
                    store_ops.insert(resource_id, StoreOp::Store);
                    continue;
                }

                let store_op = if last_read
                    .get(&resource_id)
                    .is_some_and(|&last| last > index)
                {
                    StoreOp::Store
                } else {
                    StoreOp::Discard
                };
                store_ops.insert(resource_id, store_op);
            }
        }

        for &resource_id in self.resources.descriptors.keys() {
            store_ops.entry(resource_id).or_insert_with(|| {
                let descriptor = self.resources.get_descriptor(resource_id).unwrap();
                if descriptor.is_external {
                    StoreOp::Store
                } else {
                    StoreOp::Discard
                }
            });
        }

        store_ops
    }

    fn compute_dead_passes(&self, execution_order: &[NodeIndex]) -> HashSet<NodeIndex> {
        let mut required_resources: HashSet<ResourceId> = HashSet::new();
        let mut required_passes: HashSet<NodeIndex> = HashSet::new();

        for &resource_id in self.resources.descriptors.keys() {
            let descriptor = self.resources.get_descriptor(resource_id).unwrap();
            if descriptor.is_external {
                required_resources.insert(resource_id);
            }
        }

        for &node_index in execution_order.iter().rev() {
            let node = &self.graph[node_index];

            let has_side_effects = node.writes.is_empty() && node.reads_writes.is_empty();

            let writes_required_resource =
                node.writes.iter().any(|r| required_resources.contains(r))
                    || node
                        .reads_writes
                        .iter()
                        .any(|r| required_resources.contains(r));

            if writes_required_resource || has_side_effects {
                required_passes.insert(node_index);
                required_resources.extend(&node.reads);
                required_resources.extend(&node.reads_writes);
            }
        }

        let all_passes: HashSet<NodeIndex> = execution_order.iter().copied().collect();
        all_passes.difference(&required_passes).copied().collect()
    }

    pub fn compile(&mut self) -> Result<(), String> {
        self.build_dependency_edges();

        self.execution_order = petgraph::algo::toposort(&self.graph, None)
            .map_err(|_| "Render graph contains cycles")?;

        self.store_ops = self.compute_store_ops(&self.execution_order);

        let lifetimes = self.compute_resource_lifetimes(&self.execution_order);
        self.aliasing_info = Some(self.compute_resource_aliasing(lifetimes));

        self.culled_passes = self.compute_dead_passes(&self.execution_order);

        self.needs_recompile = false;
        Ok(())
    }

    fn recompile_if_needed(&mut self) {
        if self.needs_resource_reallocation {
            let lifetimes = self.compute_resource_lifetimes(&self.execution_order);
            self.aliasing_info = Some(self.compute_resource_aliasing(lifetimes));
            self.needs_resource_reallocation = false;
            return;
        }

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

        self.culled_passes = self.compute_dead_passes(&self.execution_order);

        self.needs_recompile = false;
    }

    pub fn execute(
        &mut self,
        device: &Device,
        queue: &wgpu::Queue,
        configs: &C,
    ) -> Vec<CommandBuffer> {
        self.recompile_if_needed();

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

        self.invalidate_bind_groups_for_changed_resources();

        for &node_index in &self.execution_order {
            if self.culled_passes.contains(&node_index) {
                continue;
            }

            let node = &mut self.graph[node_index];
            node.pass.prepare(device, queue, configs);
        }

        self.execute_serial(device, configs)
    }

    fn invalidate_bind_groups_for_changed_resources(&mut self) {
        let mut dirty_resources = HashSet::new();

        for &resource_id in self.resources.descriptors.keys() {
            let current_version = self.resources.get_version(resource_id);
            let stored_version = self
                .resource_versions
                .get(&resource_id)
                .copied()
                .unwrap_or(0);

            if current_version != stored_version {
                dirty_resources.insert(resource_id);
                self.resource_versions.insert(resource_id, current_version);
            }
        }

        if dirty_resources.is_empty() {
            return;
        }

        let mut passes_to_invalidate = HashSet::new();

        for &node_index in &self.execution_order {
            let node = &self.graph[node_index];

            for &resource_id in node
                .reads
                .iter()
                .chain(&node.writes)
                .chain(&node.reads_writes)
            {
                if dirty_resources.contains(&resource_id) {
                    passes_to_invalidate.insert(node_index);
                    break;
                }
            }
        }

        for &node_index in &passes_to_invalidate {
            let node = &mut self.graph[node_index];
            node.pass.invalidate_bind_groups();
        }
    }

    fn execute_serial(&mut self, device: &Device, configs: &C) -> Vec<CommandBuffer> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RenderGraph Serial Encoder"),
        });

        for &node_index in &self.execution_order {
            if self.culled_passes.contains(&node_index) {
                continue;
            }

            let node = &mut self.graph[node_index];
            let slot_mappings = self
                .pass_resource_mappings
                .get(&node.name)
                .expect("Pass resource mappings not found");
            let context = PassExecutionContext {
                encoder: &mut encoder,
                resources: &self.resources,
                device,
                slot_mappings,
                configs,
            };
            node.pass.execute(context);
        }

        vec![encoder.finish()]
    }

    pub fn resources_mut(&mut self) -> &mut RenderGraphResources {
        &mut self.resources
    }

    pub fn resize_transient_resource(
        &mut self,
        _device: &Device,
        id: ResourceId,
        width: u32,
        height: u32,
    ) {
        self.resources
            .update_transient_descriptor(id, width, height);

        self.aliasing_info = None;
        self.resources.handles.retain(|id, _| {
            if let Some(descriptor) = self.resources.descriptors.get(id) {
                descriptor.is_external
            } else {
                false
            }
        });
        self.needs_resource_reallocation = true;
    }
}

impl<C> Default for RenderGraph<C> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
struct ResourceLifetime {
    resource_id: ResourceId,
    first_use: usize,
    last_use: usize,
}

enum PooledResource {
    Texture { texture: Arc<Texture> },
    Buffer { buffer: Arc<Buffer> },
}

#[derive(Clone)]
enum PoolDescriptorInfo {
    Texture(RenderGraphTextureDescriptor),
    Buffer(RenderGraphBufferDescriptor),
}

pub struct PoolSlot {
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

pub struct ResourceAliasingInfo {
    pub aliases: HashMap<ResourceId, usize>,
    pub pools: Vec<PoolSlot>,
}
