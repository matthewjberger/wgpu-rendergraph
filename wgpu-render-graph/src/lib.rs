use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use wgpu::{
    Buffer, BufferDescriptor, BufferUsages, CommandBuffer, CommandEncoder, Device, Extent3d,
    StoreOp, Texture, TextureDescriptor, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor,
};

#[derive(Debug, thiserror::Error)]
pub enum RenderGraphError {
    #[error("Slot '{slot}' not found in pass '{pass}' resource mappings")]
    SlotNotFound { slot: String, pass: String },

    #[error("Resource '{resource}' (id: {id:?}) not bound")]
    ResourceNotBound { resource: String, id: ResourceId },

    #[error("Resource '{resource}' descriptor not found (id: {id:?})")]
    DescriptorNotFound { resource: String, id: ResourceId },

    #[error("Type mismatch: {operation} called on {actual_type} resource '{resource}'")]
    TypeMismatch {
        operation: String,
        actual_type: String,
        resource: String,
    },

    #[error("Pass '{pass}': slot '{slot}' not provided in mappings")]
    SlotNotMapped { pass: String, slot: String },

    #[error("Cannot resize external resource '{resource}'")]
    CannotResizeExternal { resource: String },

    #[error("Cannot resize buffer '{resource}' with width/height")]
    CannotResizeBuffer { resource: String },

    #[error("Cannot resize non-transient resource '{resource}'")]
    CannotResizeNonTransient { resource: String },

    #[error("Render graph contains cycles")]
    CyclicDependency,

    #[error("Sub-graph '{sub_graph}' not found")]
    SubGraphNotFound { sub_graph: String },

    #[error("Sub-graph input '{input}' expects {expected} but received {received}")]
    SubGraphInputTypeMismatch {
        input: String,
        expected: String,
        received: String,
    },

    #[error("Resource '{resource}' (id: {id:?}) not found")]
    ResourceNotFound { resource: String, id: ResourceId },
}

pub type Result<T> = std::result::Result<T, RenderGraphError>;

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
    pub dimension: wgpu::TextureDimension,
    pub depth_or_array_layers: u32,
}

impl RenderGraphTextureDescriptor {
    pub fn to_wgpu_descriptor<'a>(&self, label: Option<&'a str>) -> TextureDescriptor<'a> {
        TextureDescriptor {
            label,
            size: Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: self.depth_or_array_layers,
            },
            mip_level_count: self.mip_level_count,
            sample_count: self.sample_count,
            dimension: self.dimension,
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
        force_store: bool,
    },
    TransientColor {
        descriptor: RenderGraphTextureDescriptor,
        clear_color: Option<wgpu::Color>,
    },
    ExternalDepth {
        clear_depth: Option<f32>,
        force_store: bool,
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
    ) -> Result<(&TextureView, wgpu::LoadOp<wgpu::Color>, StoreOp)> {
        let handle = self
            .get_handle(id)
            .ok_or_else(|| RenderGraphError::ResourceNotBound {
                resource: format!("color_attachment_{:?}", id),
                id,
            })?;
        let descriptor =
            self.get_descriptor(id)
                .ok_or_else(|| RenderGraphError::DescriptorNotFound {
                    resource: format!("color_attachment_{:?}", id),
                    id,
                })?;

        let load_op = match &descriptor.resource_type {
            ResourceType::ExternalColor { clear_color, .. }
            | ResourceType::TransientColor { clear_color, .. } => {
                if let Some(color) = clear_color {
                    wgpu::LoadOp::Clear(*color)
                } else {
                    wgpu::LoadOp::Load
                }
            }
            _ => {
                return Err(RenderGraphError::TypeMismatch {
                    operation: "get_color_attachment".to_string(),
                    actual_type: match &descriptor.resource_type {
                        ResourceType::ExternalDepth { .. }
                        | ResourceType::TransientDepth { .. } => "depth".to_string(),
                        ResourceType::ExternalBuffer | ResourceType::TransientBuffer { .. } => {
                            "buffer".to_string()
                        }
                        _ => "unknown".to_string(),
                    },
                    resource: descriptor.name.clone(),
                });
            }
        };

        Ok((handle.view(), load_op, handle.store_op()))
    }

    pub fn get_depth_attachment(
        &self,
        id: ResourceId,
    ) -> Result<(&TextureView, wgpu::LoadOp<f32>, StoreOp)> {
        let handle = self
            .get_handle(id)
            .ok_or_else(|| RenderGraphError::ResourceNotBound {
                resource: format!("depth_attachment_{:?}", id),
                id,
            })?;
        let descriptor =
            self.get_descriptor(id)
                .ok_or_else(|| RenderGraphError::DescriptorNotFound {
                    resource: format!("depth_attachment_{:?}", id),
                    id,
                })?;

        let load_op = match &descriptor.resource_type {
            ResourceType::ExternalDepth { clear_depth, .. }
            | ResourceType::TransientDepth { clear_depth, .. } => {
                if let Some(depth) = clear_depth {
                    wgpu::LoadOp::Clear(*depth)
                } else {
                    wgpu::LoadOp::Load
                }
            }
            _ => {
                return Err(RenderGraphError::TypeMismatch {
                    operation: "get_depth_attachment".to_string(),
                    actual_type: match &descriptor.resource_type {
                        ResourceType::ExternalColor { .. }
                        | ResourceType::TransientColor { .. } => "color".to_string(),
                        ResourceType::ExternalBuffer | ResourceType::TransientBuffer { .. } => {
                            "buffer".to_string()
                        }
                        _ => "unknown".to_string(),
                    },
                    resource: descriptor.name.clone(),
                });
            }
        };

        Ok((handle.view(), load_op, handle.store_op()))
    }

    pub fn get_texture_view(&self, id: ResourceId) -> Option<&TextureView> {
        self.get_handle(id).map(|handle| handle.view())
    }

    pub fn update_transient_descriptor(
        &mut self,
        id: ResourceId,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let descriptor =
            self.get_descriptor(id)
                .ok_or_else(|| RenderGraphError::ResourceNotFound {
                    resource: format!("resource_{:?}", id),
                    id,
                })?;

        if descriptor.is_external {
            return Err(RenderGraphError::CannotResizeExternal {
                resource: descriptor.name.clone(),
            });
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
                return Err(RenderGraphError::CannotResizeBuffer {
                    resource: name.clone(),
                });
            }
            _ => {
                return Err(RenderGraphError::CannotResizeNonTransient {
                    resource: name.clone(),
                });
            }
        };

        self.descriptors.insert(
            id,
            ResourceDescriptor {
                name,
                resource_type: updated_descriptor,
                is_external: false,
            },
        );

        Ok(())
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

pub struct SubGraphRunCommand<'a> {
    pub sub_graph_name: String,
    pub inputs: Vec<SlotValue<'a>>,
}

pub struct PassExecutionContext<'r, 'e, C = ()> {
    pub encoder: &'e mut CommandEncoder,
    pub resources: &'r RenderGraphResources,
    pub device: &'r Device,
    slot_mappings: &'r HashMap<String, ResourceId>,
    pub configs: &'r C,
    pub(crate) sub_graph_commands: Vec<SubGraphRunCommand<'r>>,
}

impl<'r, 'e, C> PassExecutionContext<'r, 'e, C> {
    pub fn get_slot(&self, slot: &str) -> Result<ResourceId> {
        self.slot_mappings
            .get(slot)
            .copied()
            .ok_or_else(|| RenderGraphError::SlotNotFound {
                slot: slot.to_string(),
                pass: "unknown".to_string(),
            })
    }

    pub fn get_texture_view(&self, slot: &str) -> Result<&'r wgpu::TextureView> {
        let resource_id = self.get_slot(slot)?;
        self.resources.get_texture_view(resource_id).ok_or_else(|| {
            RenderGraphError::ResourceNotBound {
                resource: slot.to_string(),
                id: resource_id,
            }
        })
    }

    pub fn get_color_attachment(
        &self,
        slot: &str,
    ) -> Result<(
        &'r wgpu::TextureView,
        wgpu::LoadOp<wgpu::Color>,
        wgpu::StoreOp,
    )> {
        let resource_id = self.get_slot(slot)?;
        self.resources.get_color_attachment(resource_id)
    }

    pub fn get_depth_attachment(
        &self,
        slot: &str,
    ) -> Result<(&'r wgpu::TextureView, wgpu::LoadOp<f32>, wgpu::StoreOp)> {
        let resource_id = self.get_slot(slot)?;
        self.resources.get_depth_attachment(resource_id)
    }

    pub fn get_buffer(&self, slot: &str) -> Result<&'r std::sync::Arc<wgpu::Buffer>> {
        let resource_id = self.get_slot(slot)?;
        let handle = self.resources.get_handle(resource_id).ok_or_else(|| {
            RenderGraphError::ResourceNotBound {
                resource: slot.to_string(),
                id: resource_id,
            }
        })?;

        match handle {
            ResourceHandle::ExternalBuffer { buffer }
            | ResourceHandle::TransientBuffer { buffer } => Ok(buffer),
            _ => Err(RenderGraphError::TypeMismatch {
                operation: "get_buffer".to_string(),
                actual_type: "texture".to_string(),
                resource: slot.to_string(),
            }),
        }
    }

    pub fn get_texture_size(&self, slot: &str) -> Result<(u32, u32)> {
        let resource_id = self.get_slot(slot)?;
        let descriptor = self.resources.get_descriptor(resource_id).ok_or_else(|| {
            RenderGraphError::DescriptorNotFound {
                resource: slot.to_string(),
                id: resource_id,
            }
        })?;

        match &descriptor.resource_type {
            ResourceType::TransientColor {
                descriptor: texture_desc,
                ..
            }
            | ResourceType::TransientDepth {
                descriptor: texture_desc,
                ..
            } => Ok((texture_desc.width, texture_desc.height)),
            ResourceType::ExternalColor { .. } | ResourceType::ExternalDepth { .. } => {
                let handle = self.resources.get_handle(resource_id).ok_or_else(|| {
                    RenderGraphError::ResourceNotBound {
                        resource: slot.to_string(),
                        id: resource_id,
                    }
                })?;

                match handle {
                    ResourceHandle::TransientTexture { texture, .. } => {
                        Ok((texture.width(), texture.height()))
                    }
                    _ => Err(RenderGraphError::TypeMismatch {
                        operation: "get_texture_size".to_string(),
                        actual_type: "external_texture".to_string(),
                        resource: slot.to_string(),
                    }),
                }
            }
            _ => Err(RenderGraphError::TypeMismatch {
                operation: "get_texture_size".to_string(),
                actual_type: "buffer".to_string(),
                resource: slot.to_string(),
            }),
        }
    }

    pub fn run_sub_graph(&mut self, sub_graph_name: String, inputs: Vec<SlotValue<'r>>) {
        self.sub_graph_commands.push(SubGraphRunCommand {
            sub_graph_name,
            inputs,
        });
    }

    pub fn into_sub_graph_commands(self) -> Vec<SubGraphRunCommand<'r>> {
        self.sub_graph_commands
    }
}

pub trait PassNode<C = ()>: Send + Sync {
    fn name(&self) -> &str;
    fn reads(&self) -> Vec<&str>;
    fn writes(&self) -> Vec<&str>;
    fn reads_writes(&self) -> Vec<&str> {
        Vec::new()
    }
    fn is_enabled(&self, _configs: &C) -> bool {
        true
    }
    fn prepare(&mut self, _device: &Device, _queue: &wgpu::Queue, _configs: &C) {}
    fn invalidate_bind_groups(&mut self) {}
    fn execute<'r, 'e>(
        &mut self,
        context: PassExecutionContext<'r, 'e, C>,
    ) -> Result<Vec<SubGraphRunCommand<'r>>>;
}

pub struct GraphNode<C> {
    pub name: String,
    pub reads: Vec<ResourceId>,
    pub writes: Vec<ResourceId>,
    pub reads_writes: Vec<ResourceId>,
    pub pass: Box<dyn PassNode<C>>,
}

pub enum SlotValue<'a> {
    TextureView(&'a TextureView),
    Buffer(&'a Arc<Buffer>),
}

#[derive(Clone)]
pub struct SubGraphInputSlot {
    pub name: String,
}

pub struct ColorTextureBuilder<'a, C = ()> {
    graph: &'a mut RenderGraph<C>,
    name: String,
    descriptor: RenderGraphTextureDescriptor,
    clear_color: Option<wgpu::Color>,
    force_store: bool,
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

    pub fn no_store(mut self) -> Self {
        self.force_store = false;
        self
    }

    pub fn external(self) -> ResourceId {
        self.graph.resources.register_external_resource(
            self.name,
            ResourceType::ExternalColor {
                clear_color: self.clear_color,
                force_store: self.force_store,
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
    force_store: bool,
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

    pub fn no_store(mut self) -> Self {
        self.force_store = false;
        self
    }

    pub fn external(self) -> ResourceId {
        self.graph.resources.register_external_resource(
            self.name,
            ResourceType::ExternalDepth {
                clear_depth: self.clear_depth,
                force_store: self.force_store,
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

#[derive(Clone)]
pub struct ResourceTemplate {
    format: TextureFormat,
    width: u32,
    height: u32,
    usage: TextureUsages,
    sample_count: u32,
    mip_level_count: u32,
    dimension: wgpu::TextureDimension,
    depth_or_array_layers: u32,
}

impl ResourceTemplate {
    pub fn new(format: TextureFormat, width: u32, height: u32) -> Self {
        Self {
            format,
            width,
            height,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            sample_count: 1,
            mip_level_count: 1,
            dimension: wgpu::TextureDimension::D2,
            depth_or_array_layers: 1,
        }
    }

    pub fn usage(mut self, usage: TextureUsages) -> Self {
        self.usage = usage;
        self
    }

    pub fn sample_count(mut self, count: u32) -> Self {
        self.sample_count = count;
        self
    }

    pub fn mip_levels(mut self, levels: u32) -> Self {
        self.mip_level_count = levels;
        self
    }

    pub fn cube_map(mut self) -> Self {
        self.dimension = wgpu::TextureDimension::D2;
        self.depth_or_array_layers = 6;
        self
    }

    pub fn array_layers(mut self, layers: u32) -> Self {
        self.depth_or_array_layers = layers;
        self
    }

    pub fn dimension_3d(mut self, depth: u32) -> Self {
        self.dimension = wgpu::TextureDimension::D3;
        self.depth_or_array_layers = depth;
        self
    }
}

pub struct PassBuilder<'a, C = ()> {
    graph: &'a mut RenderGraph<C>,
    pass: Option<Box<dyn PassNode<C>>>,
    slots: Vec<(&'static str, ResourceId)>,
}

impl<'a, C> PassBuilder<'a, C> {
    pub fn read(mut self, slot: &'static str, resource: ResourceId) -> Self {
        self.slots.push((slot, resource));
        self
    }

    pub fn write(mut self, slot: &'static str, resource: ResourceId) -> Self {
        self.slots.push((slot, resource));
        self
    }

    pub fn slot(mut self, slot: &'static str, resource: ResourceId) -> Self {
        self.slots.push((slot, resource));
        self
    }
}

impl<'a, C> Drop for PassBuilder<'a, C> {
    fn drop(&mut self) {
        if let Some(pass) = self.pass.take() {
            let result = self.graph.add_pass(pass, &self.slots);
            if let Err(e) = result {
                panic!("Failed to add render pass: {}", e);
            }
        }
    }
}

pub struct ResourcePool<'a, C = ()> {
    graph: &'a mut RenderGraph<C>,
    template: ResourceTemplate,
}

impl<'a, C> ResourcePool<'a, C> {
    pub fn transient(&mut self, name: &str) -> ResourceId {
        self.graph.transient_color_from_template(name, &self.template)
    }

    pub fn transient_many(&mut self, names: &[&str]) -> Vec<ResourceId> {
        names.iter().map(|name| self.transient(name)).collect()
    }

    pub fn external(&mut self, name: &str) -> ResourceId {
        self.graph.external_color_from_template(name, &self.template)
    }
}

pub struct RenderGraph<C = ()> {
    graph: DiGraph<GraphNode<C>, ResourceId>,
    pass_nodes: HashMap<String, NodeIndex>,
    pass_resource_mappings: HashMap<String, HashMap<String, ResourceId>>,
    sub_graphs: HashMap<String, RenderGraph<C>>,
    sub_graph_inputs: HashMap<String, Vec<SubGraphInputSlot>>,
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
            sub_graphs: HashMap::new(),
            sub_graph_inputs: HashMap::new(),
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
    ) -> Result<NodeIndex> {
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
                mappings
                    .get(*slot)
                    .copied()
                    .ok_or_else(|| RenderGraphError::SlotNotMapped {
                        pass: name.clone(),
                        slot: slot.to_string(),
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        let writes: Vec<ResourceId> = slot_names_writes
            .iter()
            .map(|slot| {
                mappings
                    .get(*slot)
                    .copied()
                    .ok_or_else(|| RenderGraphError::SlotNotMapped {
                        pass: name.clone(),
                        slot: slot.to_string(),
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        let reads_writes: Vec<ResourceId> = slot_names_reads_writes
            .iter()
            .map(|slot| {
                mappings
                    .get(*slot)
                    .copied()
                    .ok_or_else(|| RenderGraphError::SlotNotMapped {
                        pass: name.clone(),
                        slot: slot.to_string(),
                    })
            })
            .collect::<Result<Vec<_>>>()?;

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
        Ok(index)
    }

    pub fn add_sub_graph(
        &mut self,
        name: String,
        sub_graph: RenderGraph<C>,
        input_slots: Vec<SubGraphInputSlot>,
    ) {
        self.sub_graphs.insert(name.clone(), sub_graph);
        self.sub_graph_inputs.insert(name, input_slots);
    }

    pub fn get_sub_graph(&self, name: &str) -> Option<&RenderGraph<C>> {
        self.sub_graphs.get(name)
    }

    pub fn get_sub_graph_mut(&mut self, name: &str) -> Option<&mut RenderGraph<C>> {
        self.sub_graphs.get_mut(name)
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
                dimension: wgpu::TextureDimension::D2,
                depth_or_array_layers: 1,
            },
            clear_color: None,
            force_store: true,
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
                dimension: wgpu::TextureDimension::D2,
                depth_or_array_layers: 1,
            },
            clear_depth: None,
            force_store: true,
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

    pub fn transient_color_from_template(
        &mut self,
        name: &str,
        template: &ResourceTemplate,
    ) -> ResourceId {
        self.resources.register_transient_resource(
            name.to_string(),
            ResourceType::TransientColor {
                descriptor: RenderGraphTextureDescriptor {
                    format: template.format,
                    width: template.width,
                    height: template.height,
                    usage: template.usage,
                    sample_count: template.sample_count,
                    mip_level_count: template.mip_level_count,
                    dimension: template.dimension,
                    depth_or_array_layers: template.depth_or_array_layers,
                },
                clear_color: None,
            },
        )
    }

    pub fn transient_color_from_template_with_clear(
        &mut self,
        name: &str,
        template: &ResourceTemplate,
        clear_color: wgpu::Color,
    ) -> ResourceId {
        self.resources.register_transient_resource(
            name.to_string(),
            ResourceType::TransientColor {
                descriptor: RenderGraphTextureDescriptor {
                    format: template.format,
                    width: template.width,
                    height: template.height,
                    usage: template.usage,
                    sample_count: template.sample_count,
                    mip_level_count: template.mip_level_count,
                    dimension: template.dimension,
                    depth_or_array_layers: template.depth_or_array_layers,
                },
                clear_color: Some(clear_color),
            },
        )
    }

    pub fn external_color_from_template(
        &mut self,
        name: &str,
        _template: &ResourceTemplate,
    ) -> ResourceId {
        self.resources.register_external_resource(
            name.to_string(),
            ResourceType::ExternalColor {
                clear_color: None,
                force_store: true,
            },
        )
    }

    pub fn add_pass_with_slots(
        &mut self,
        pass: Box<dyn PassNode<C>>,
        slots: Vec<(&str, ResourceId)>,
    ) -> Result<NodeIndex> {
        self.add_pass(pass, &slots)
    }

    pub fn pass(&mut self, pass: Box<dyn PassNode<C>>) -> PassBuilder<'_, C> {
        PassBuilder {
            graph: self,
            pass: Some(pass),
            slots: Vec::new(),
        }
    }

    pub fn resource_pool(&mut self, template: &ResourceTemplate) -> ResourcePool<'_, C> {
        ResourcePool {
            graph: self,
            template: template.clone(),
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
                    let needs_new_resource =
                        match (&mut candidate.descriptor_info, &descriptor.resource_type) {
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
                            ) => {
                                if !pool_desc.usage.contains(res_desc.usage) {
                                    pool_desc.usage = pool_desc.usage | res_desc.usage;
                                    true
                                } else {
                                    false
                                }
                            }
                            (
                                PoolDescriptorInfo::Buffer(pool_desc),
                                ResourceType::TransientBuffer {
                                    descriptor: res_desc,
                                },
                            ) => {
                                if res_desc.size > pool_desc.size {
                                    *pool_desc = res_desc.clone();
                                    true
                                } else {
                                    false
                                }
                            }
                            _ => false,
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
                    let store_op = match &descriptor.resource_type {
                        ResourceType::ExternalColor { force_store, .. }
                        | ResourceType::ExternalDepth { force_store, .. } => {
                            if *force_store {
                                StoreOp::Store
                            } else if last_read
                                .get(&resource_id)
                                .is_some_and(|&last| last > index)
                            {
                                StoreOp::Store
                            } else {
                                StoreOp::Discard
                            }
                        }
                        ResourceType::ExternalBuffer => StoreOp::Store,
                        _ => StoreOp::Store,
                    };
                    store_ops.insert(resource_id, store_op);
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
                    match &descriptor.resource_type {
                        ResourceType::ExternalColor { force_store, .. }
                        | ResourceType::ExternalDepth { force_store, .. } => {
                            if *force_store {
                                StoreOp::Store
                            } else {
                                StoreOp::Discard
                            }
                        }
                        _ => StoreOp::Store,
                    }
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

    pub fn compile(&mut self) -> Result<()> {
        self.build_dependency_edges();

        self.execution_order = petgraph::algo::toposort(&self.graph, None)
            .map_err(|_| RenderGraphError::CyclicDependency)?;

        self.store_ops = self.compute_store_ops(&self.execution_order);

        let lifetimes = self.compute_resource_lifetimes(&self.execution_order);
        self.aliasing_info = Some(self.compute_resource_aliasing(lifetimes));

        self.culled_passes = self.compute_dead_passes(&self.execution_order);

        self.needs_recompile = false;
        Ok(())
    }

    fn recompile_if_needed(&mut self) -> Result<()> {
        if self.needs_resource_reallocation {
            let lifetimes = self.compute_resource_lifetimes(&self.execution_order);
            self.aliasing_info = Some(self.compute_resource_aliasing(lifetimes));
            self.needs_resource_reallocation = false;
            return Ok(());
        }

        if !self.needs_recompile {
            return Ok(());
        }

        let edge_indices: Vec<_> = self.graph.edge_indices().collect();
        for edge_index in edge_indices {
            self.graph.remove_edge(edge_index);
        }

        self.build_dependency_edges();

        self.execution_order = petgraph::algo::toposort(&self.graph, None)
            .map_err(|_| RenderGraphError::CyclicDependency)?;

        self.store_ops = self.compute_store_ops(&self.execution_order);

        let lifetimes = self.compute_resource_lifetimes(&self.execution_order);
        self.aliasing_info = Some(self.compute_resource_aliasing(lifetimes));

        self.culled_passes = self.compute_dead_passes(&self.execution_order);

        self.needs_recompile = false;
        Ok(())
    }

    pub fn execute(
        &mut self,
        device: &Device,
        queue: &wgpu::Queue,
        configs: &C,
    ) -> Result<Vec<CommandBuffer>> {
        self.recompile_if_needed()?;

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

            if !node.pass.is_enabled(configs) {
                continue;
            }

            node.pass.prepare(device, queue, configs);
        }

        self.execute_serial(device, queue, configs)
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

    fn execute_serial(
        &mut self,
        device: &Device,
        queue: &wgpu::Queue,
        configs: &C,
    ) -> Result<Vec<CommandBuffer>> {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RenderGraph Serial Encoder"),
        });

        let mut command_buffers = Vec::new();

        for &node_index in &self.execution_order {
            if self.culled_passes.contains(&node_index) {
                continue;
            }

            let node = &mut self.graph[node_index];

            if !node.pass.is_enabled(configs) {
                continue;
            }

            let slot_mappings = self.pass_resource_mappings.get(&node.name).ok_or_else(|| {
                RenderGraphError::ResourceNotFound {
                    resource: format!("pass_{}_mappings", node.name),
                    id: ResourceId(0),
                }
            })?;

            let sub_graph_commands = {
                let context = PassExecutionContext {
                    encoder: &mut encoder,
                    resources: &self.resources,
                    device,
                    slot_mappings,
                    configs,
                    sub_graph_commands: Vec::new(),
                };

                node.pass.execute(context)?
            };

            for command in sub_graph_commands {
                command_buffers.push(encoder.finish());

                let sub_graph = self
                    .sub_graphs
                    .get_mut(&command.sub_graph_name)
                    .ok_or_else(|| RenderGraphError::SubGraphNotFound {
                        sub_graph: command.sub_graph_name.clone(),
                    })?;

                let input_slots = self
                    .sub_graph_inputs
                    .get(&command.sub_graph_name)
                    .cloned()
                    .unwrap_or_default();

                for (index, slot_value) in command.inputs.iter().enumerate() {
                    if let Some(input_slot) = input_slots.get(index) {
                        match slot_value {
                            SlotValue::TextureView(view) => {
                                if let Some(resource_id) = sub_graph
                                    .resources
                                    .descriptors
                                    .iter()
                                    .find(|(_, desc)| {
                                        desc.name == input_slot.name && desc.is_external
                                    })
                                    .map(|(id, _)| *id)
                                {
                                    let descriptor = sub_graph
                                        .resources
                                        .get_descriptor(resource_id)
                                        .ok_or_else(|| RenderGraphError::DescriptorNotFound {
                                            resource: input_slot.name.clone(),
                                            id: resource_id,
                                        })?;
                                    match &descriptor.resource_type {
                                        ResourceType::ExternalColor { .. }
                                        | ResourceType::ExternalDepth { .. } => {
                                            sub_graph
                                                .resources
                                                .set_external_texture(resource_id, (*view).clone());
                                        }
                                        _ => {
                                            return Err(
                                                RenderGraphError::SubGraphInputTypeMismatch {
                                                    input: input_slot.name.clone(),
                                                    expected: "buffer".to_string(),
                                                    received: "texture".to_string(),
                                                },
                                            );
                                        }
                                    }
                                }
                            }
                            SlotValue::Buffer(buffer) => {
                                if let Some(resource_id) = sub_graph
                                    .resources
                                    .descriptors
                                    .iter()
                                    .find(|(_, desc)| {
                                        desc.name == input_slot.name && desc.is_external
                                    })
                                    .map(|(id, _)| *id)
                                {
                                    let descriptor = sub_graph
                                        .resources
                                        .get_descriptor(resource_id)
                                        .ok_or_else(|| RenderGraphError::DescriptorNotFound {
                                            resource: input_slot.name.clone(),
                                            id: resource_id,
                                        })?;
                                    match &descriptor.resource_type {
                                        ResourceType::ExternalBuffer => {
                                            sub_graph.resources.set_external_buffer(
                                                resource_id,
                                                (*buffer).clone(),
                                            );
                                        }
                                        _ => {
                                            return Err(
                                                RenderGraphError::SubGraphInputTypeMismatch {
                                                    input: input_slot.name.clone(),
                                                    expected: "texture".to_string(),
                                                    received: "buffer".to_string(),
                                                },
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                let sub_graph_buffers = sub_graph.execute(device, queue, configs)?;
                command_buffers.extend(sub_graph_buffers);

                encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("RenderGraph Serial Encoder"),
                });
            }
        }

        command_buffers.push(encoder.finish());
        Ok(command_buffers)
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
    ) -> Result<()> {
        self.resources
            .update_transient_descriptor(id, width, height)?;

        self.aliasing_info = None;
        self.resources.handles.retain(|id, _| {
            if let Some(descriptor) = self.resources.descriptors.get(id) {
                descriptor.is_external
            } else {
                false
            }
        });
        self.needs_resource_reallocation = true;
        Ok(())
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

#[macro_export]
macro_rules! pass_slots {
    ($($slot:ident: $resource:expr),* $(,)?) => {
        vec![
            $((stringify!($slot), $resource),)*
        ]
    };
}

#[macro_export]
macro_rules! arc_data {
    ($struct_name:ident { $($field:ident: $value:expr),* $(,)? }) => {
        $struct_name {
            $(
                $field: std::sync::Arc::clone(&$value),
            )*
        }
    };
}
