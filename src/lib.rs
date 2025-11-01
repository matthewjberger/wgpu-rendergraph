#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
use wgpu::InstanceDescriptor;

mod pass_configs;
mod passes;

use std::sync::Arc;
use web_time::{Duration, Instant};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    window::{Theme, Window},
};

use kira::{
    AudioManager, AudioManagerSettings, DefaultBackend, sound::static_sound::StaticSoundData,
};
use std::collections::HashMap;

#[derive(Clone)]
pub struct Camera {
    pub id: usize,
    pub name: String,
    pub position: nalgebra_glm::Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov: f32,
}

struct ViewportPane {
    name: String,
    selected_camera_id: Option<usize>,
}

struct TileTreeBehavior {
    viewport_dimensions: HashMap<egui_tiles::TileId, (f32, f32)>,
    viewport_texture_id: Option<egui::TextureId>,
    viewport_texture_ids: HashMap<egui_tiles::TileId, egui::TextureId>,
    max_viewport_size: (f32, f32),
    add_viewport_requested: bool,
    cameras: Vec<Camera>,
    camera_texture_aspect: f32,
}

impl egui_tiles::Behavior<ViewportPane> for TileTreeBehavior {
    fn simplification_options(&self) -> egui_tiles::SimplificationOptions {
        egui_tiles::SimplificationOptions {
            all_panes_must_have_tabs: true,
            ..Default::default()
        }
    }

    fn tab_title_for_pane(&mut self, pane: &ViewportPane) -> egui::WidgetText {
        pane.name.as_str().into()
    }

    fn top_bar_right_ui(
        &mut self,
        _tiles: &egui_tiles::Tiles<ViewportPane>,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
        _tabs: &egui_tiles::Tabs,
        _scroll_offset: &mut f32,
    ) {
        if ui.button("➕").clicked() {
            self.add_viewport_requested = true;
        }
    }

    fn pane_ui(
        &mut self,
        ui: &mut egui::Ui,
        _tile_id: egui_tiles::TileId,
        pane: &mut ViewportPane,
    ) -> egui_tiles::UiResponse {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.label("Camera:");
                egui::ComboBox::from_id_salt((_tile_id, "camera_selector"))
                    .selected_text(
                        pane.selected_camera_id
                            .and_then(|id| self.cameras.iter().find(|c| c.id == id))
                            .map(|c| c.name.as_str())
                            .unwrap_or("None"),
                    )
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut pane.selected_camera_id, None, "None");
                        for camera in &self.cameras {
                            ui.selectable_value(
                                &mut pane.selected_camera_id,
                                Some(camera.id),
                                &camera.name,
                            );
                        }
                    });
            });

            let available_size = ui.available_size();
            self.viewport_dimensions
                .insert(_tile_id, (available_size.x, available_size.y));

            if pane.selected_camera_id.is_none() {
                ui.centered_and_justified(|ui| {
                    ui.label("No camera selected");
                });
            } else if let Some(texture_id) = self.viewport_texture_ids.get(&_tile_id) {
                let camera = pane
                    .selected_camera_id
                    .and_then(|id| self.cameras.iter().find(|c| c.id == id));

                let display_size = if let Some(_camera) = camera {
                    let camera_aspect = self.camera_texture_aspect;
                    let viewport_aspect = available_size.x / available_size.y.max(0.1);

                    if camera_aspect > viewport_aspect {
                        let width = available_size.y * camera_aspect;
                        egui::vec2(width, available_size.y)
                    } else {
                        let height = available_size.x / camera_aspect;
                        egui::vec2(available_size.x, height)
                    }
                } else {
                    available_size
                };

                ui.centered_and_justified(|ui| {
                    ui.add(egui::Image::new(egui::load::SizedTexture {
                        id: *texture_id,
                        size: display_size,
                    }));
                });
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label(format!("Viewport: {}", pane.name));
                });
            }
        });

        egui_tiles::UiResponse::None
    }
}

pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    gui_state: Option<egui_winit::State>,
    last_render_time: Option<Instant>,
    #[cfg(target_arch = "wasm32")]
    renderer_receiver: Option<futures::channel::oneshot::Receiver<Renderer>>,
    audio_manager: Option<AudioManager>,
    last_size: (u32, u32),
    tile_tree: Option<egui_tiles::Tree<ViewportPane>>,
    tile_tree_behavior: TileTreeBehavior,
}

impl Default for App {
    fn default() -> Self {
        let origin = nalgebra_glm::vec3(0.0_f32, 0.0_f32, 0.0_f32);

        let camera1_pos = nalgebra_glm::vec3(0.0_f32, 2.0_f32, 5.0_f32);
        let camera1_dir = nalgebra_glm::normalize(&(origin - camera1_pos));
        let camera1_yaw = camera1_dir.z.atan2(camera1_dir.x);
        let camera1_pitch = camera1_dir.y.asin();

        let camera2_pos = nalgebra_glm::vec3(5.0_f32, 2.0_f32, 0.0_f32);
        let camera2_dir = nalgebra_glm::normalize(&(origin - camera2_pos));
        let camera2_yaw = camera2_dir.z.atan2(camera2_dir.x);
        let camera2_pitch = camera2_dir.y.asin();

        let camera3_pos = nalgebra_glm::vec3(0.5_f32, 5.0_f32, 0.5_f32);
        let camera3_dir = nalgebra_glm::normalize(&(origin - camera3_pos));
        let camera3_yaw = camera3_dir.z.atan2(camera3_dir.x);
        let camera3_pitch = camera3_dir.y.asin();

        let cameras = vec![
            Camera {
                id: 0,
                name: "Camera 1".to_string(),
                position: camera1_pos,
                yaw: camera1_yaw,
                pitch: camera1_pitch,
                fov: 45.0,
            },
            Camera {
                id: 1,
                name: "Camera 2".to_string(),
                position: camera2_pos,
                yaw: camera2_yaw,
                pitch: camera2_pitch,
                fov: 45.0,
            },
            Camera {
                id: 2,
                name: "Camera 3".to_string(),
                position: camera3_pos,
                yaw: camera3_yaw,
                pitch: camera3_pitch,
                fov: 60.0,
            },
        ];

        let mut tiles = egui_tiles::Tiles::default();

        let viewport1 = ViewportPane {
            name: "Viewport 1".to_string(),
            selected_camera_id: Some(0),
        };
        let viewport2 = ViewportPane {
            name: "Viewport 2".to_string(),
            selected_camera_id: Some(1),
        };
        let viewport3 = ViewportPane {
            name: "Viewport 3".to_string(),
            selected_camera_id: Some(2),
        };

        let pane1 = tiles.insert_pane(viewport1);
        let pane2 = tiles.insert_pane(viewport2);
        let pane3 = tiles.insert_pane(viewport3);

        let root = tiles.insert_tab_tile(vec![pane1, pane2, pane3]);

        let tree = egui_tiles::Tree::new("viewport_tree", root, tiles);

        Self {
            window: None,
            renderer: None,
            gui_state: None,
            last_render_time: None,
            #[cfg(target_arch = "wasm32")]
            renderer_receiver: None,
            audio_manager: None,
            last_size: (0, 0),
            tile_tree: Some(tree),
            tile_tree_behavior: TileTreeBehavior {
                viewport_dimensions: HashMap::new(),
                viewport_texture_id: None,
                viewport_texture_ids: HashMap::new(),
                max_viewport_size: (0.0, 0.0),
                add_viewport_requested: false,
                cameras,
                camera_texture_aspect: 1.0,
            },
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let mut attributes = Window::default_attributes();

        #[cfg(not(target_arch = "wasm32"))]
        {
            attributes = attributes.with_title("Standalone Winit/Wgpu Example");
        }

        #[allow(unused_assignments)]
        #[cfg(target_arch = "wasm32")]
        let (mut canvas_width, mut canvas_height) = (0, 0);

        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowAttributesExtWebSys;
            let canvas = wgpu::web_sys::window()
                .unwrap()
                .document()
                .unwrap()
                .get_element_by_id("canvas")
                .unwrap()
                .dyn_into::<wgpu::web_sys::HtmlCanvasElement>()
                .unwrap();
            canvas_width = canvas.width();
            canvas_height = canvas.height();
            self.last_size = (canvas_width, canvas_height);
            attributes = attributes.with_canvas(Some(canvas));
        }

        let Ok(window) = event_loop.create_window(attributes) else {
            return;
        };

        let first_window_handle = self.window.is_none();
        let window_handle = Arc::new(window);
        self.window = Some(window_handle.clone());
        if !first_window_handle {
            return;
        }
        let gui_context = egui::Context::default();

        #[cfg(not(target_arch = "wasm32"))]
        {
            let inner_size = window_handle.inner_size();
            self.last_size = (inner_size.width, inner_size.height);
        }

        #[cfg(target_arch = "wasm32")]
        {
            gui_context.set_pixels_per_point(window_handle.scale_factor() as f32);
        }

        let viewport_id = gui_context.viewport_id();
        let gui_state = egui_winit::State::new(
            gui_context,
            viewport_id,
            &window_handle,
            Some(window_handle.scale_factor() as _),
            Some(Theme::Dark),
            None,
        );

        #[cfg(not(target_arch = "wasm32"))]
        let (width, height) = (
            window_handle.inner_size().width,
            window_handle.inner_size().height,
        );

        #[cfg(not(target_arch = "wasm32"))]
        {
            env_logger::init();
            let renderer = pollster::block_on(async move {
                Renderer::new(window_handle.clone(), width, height).await
            });
            self.renderer = Some(renderer);
        }

        #[cfg(target_arch = "wasm32")]
        {
            let (sender, receiver) = futures::channel::oneshot::channel();
            self.renderer_receiver = Some(receiver);
            std::panic::set_hook(Box::new(console_error_panic_hook::hook));
            console_log::init().expect("Failed to initialize logger!");
            log::info!("Canvas dimensions: ({canvas_width} x {canvas_height})");
            wasm_bindgen_futures::spawn_local(async move {
                let renderer =
                    Renderer::new(window_handle.clone(), canvas_width, canvas_height).await;
                if sender.send(renderer).is_err() {
                    log::error!("Failed to create and send renderer!");
                }
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            match AudioManager::<DefaultBackend>::new(AudioManagerSettings::default()) {
                Ok(manager) => {
                    log::info!("Audio manager initialized successfully");
                    self.audio_manager = Some(manager);
                }
                Err(error) => {
                    log::error!("Failed to initialize audio manager: {:?}", error);
                }
            }
        }

        self.gui_state = Some(gui_state);
        self.last_render_time = Some(Instant::now());
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        #[cfg(target_arch = "wasm32")]
        {
            let mut renderer_received = false;
            if let Some(receiver) = self.renderer_receiver.as_mut() {
                if let Ok(Some(renderer)) = receiver.try_recv() {
                    self.renderer = Some(renderer);
                    renderer_received = true;
                }
            }
            if renderer_received {
                self.renderer_receiver = None;
            }
        }

        let (Some(gui_state), Some(renderer), Some(window), Some(last_render_time)) = (
            self.gui_state.as_mut(),
            self.renderer.as_mut(),
            self.window.as_ref(),
            self.last_render_time.as_mut(),
        ) else {
            return;
        };

        if gui_state.on_window_event(window, &event).consumed {
            return;
        }

        match event {
            WindowEvent::KeyboardInput {
                event:
                    winit::event::KeyEvent {
                        physical_key: winit::keyboard::PhysicalKey::Code(key_code),
                        state: winit::event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if matches!(key_code, winit::keyboard::KeyCode::Escape) {
                    event_loop.exit();
                }

                #[cfg(target_arch = "wasm32")]
                {
                    if self.audio_manager.is_none() {
                        log::info!("Initializing audio manager on first key press (WASM)");
                        match AudioManager::<DefaultBackend>::new(AudioManagerSettings::default()) {
                            Ok(manager) => {
                                log::info!("Audio manager initialized successfully on WASM");
                                self.audio_manager = Some(manager);
                            }
                            Err(error) => {
                                log::error!(
                                    "Failed to initialize audio manager on WASM: {:?}",
                                    error
                                );
                            }
                        }
                    }
                }

                if let Some(audio_manager) = self.audio_manager.as_mut() {
                    #[cfg(not(target_arch = "wasm32"))]
                    let sound_result = match key_code {
                        winit::keyboard::KeyCode::Digit1 => {
                            Some(StaticSoundData::from_file("assets/blip.ogg"))
                        }
                        winit::keyboard::KeyCode::Digit2 => {
                            Some(StaticSoundData::from_file("assets/drums.ogg"))
                        }
                        winit::keyboard::KeyCode::Digit3 => {
                            Some(StaticSoundData::from_file("assets/score.ogg"))
                        }
                        winit::keyboard::KeyCode::Digit4 => {
                            Some(StaticSoundData::from_file("assets/sine.wav"))
                        }
                        winit::keyboard::KeyCode::KeyQ => {
                            Some(StaticSoundData::from_file("assets/dynamic/arp.ogg"))
                        }
                        winit::keyboard::KeyCode::KeyW => {
                            Some(StaticSoundData::from_file("assets/dynamic/bass.ogg"))
                        }
                        winit::keyboard::KeyCode::KeyE => {
                            Some(StaticSoundData::from_file("assets/dynamic/drums.ogg"))
                        }
                        winit::keyboard::KeyCode::KeyR => {
                            Some(StaticSoundData::from_file("assets/dynamic/lead.ogg"))
                        }
                        winit::keyboard::KeyCode::KeyT => {
                            Some(StaticSoundData::from_file("assets/dynamic/pad.ogg"))
                        }
                        _ => None,
                    };

                    #[cfg(target_arch = "wasm32")]
                    let sound_result = match key_code {
                        winit::keyboard::KeyCode::Digit1 => Some(StaticSoundData::from_cursor(
                            std::io::Cursor::new(include_bytes!("../assets/blip.ogg")),
                        )),
                        winit::keyboard::KeyCode::Digit2 => Some(StaticSoundData::from_cursor(
                            std::io::Cursor::new(include_bytes!("../assets/drums.ogg")),
                        )),
                        winit::keyboard::KeyCode::Digit3 => Some(StaticSoundData::from_cursor(
                            std::io::Cursor::new(include_bytes!("../assets/score.ogg")),
                        )),
                        winit::keyboard::KeyCode::Digit4 => Some(StaticSoundData::from_cursor(
                            std::io::Cursor::new(include_bytes!("../assets/sine.wav")),
                        )),
                        winit::keyboard::KeyCode::KeyQ => Some(StaticSoundData::from_cursor(
                            std::io::Cursor::new(include_bytes!("../assets/dynamic/arp.ogg")),
                        )),
                        winit::keyboard::KeyCode::KeyW => Some(StaticSoundData::from_cursor(
                            std::io::Cursor::new(include_bytes!("../assets/dynamic/bass.ogg")),
                        )),
                        winit::keyboard::KeyCode::KeyE => Some(StaticSoundData::from_cursor(
                            std::io::Cursor::new(include_bytes!("../assets/dynamic/drums.ogg")),
                        )),
                        winit::keyboard::KeyCode::KeyR => Some(StaticSoundData::from_cursor(
                            std::io::Cursor::new(include_bytes!("../assets/dynamic/lead.ogg")),
                        )),
                        winit::keyboard::KeyCode::KeyT => Some(StaticSoundData::from_cursor(
                            std::io::Cursor::new(include_bytes!("../assets/dynamic/pad.ogg")),
                        )),
                        _ => None,
                    };

                    if let Some(result) = sound_result {
                        log::info!("Attempting to play sound");
                        match result {
                            Ok(sound_data) => {
                                log::info!("Sound data loaded successfully");
                                match audio_manager.play(sound_data) {
                                    Ok(_handle) => {
                                        log::info!("Sound playback started successfully");
                                    }
                                    Err(error) => {
                                        log::error!("Failed to play sound: {:?}", error);
                                    }
                                }
                            }
                            Err(error) => {
                                log::error!("Failed to load sound data: {:?}", error);
                            }
                        }
                    } else {
                        log::info!("No sound mapped to this key");
                    }
                }
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                gui_state
                    .egui_ctx()
                    .set_pixels_per_point(scale_factor as f32);
            }
            WindowEvent::Resized(PhysicalSize { width, height }) => {
                if width == 0 || height == 0 {
                    return;
                }

                log::info!("Resizing renderer surface to: ({width}, {height})");
                renderer.resize(width, height);
                self.last_size = (width, height);

                let scale_factor = window.scale_factor() as f32;
                gui_state.egui_ctx().set_pixels_per_point(scale_factor);
            }
            WindowEvent::CloseRequested => {
                log::info!("Close requested. Exiting...");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let delta_time = now - *last_render_time;
                *last_render_time = now;

                self.tile_tree_behavior.viewport_texture_id = renderer.viewport_texture_id();
                self.tile_tree_behavior.max_viewport_size = (
                    renderer.gpu.surface_config.width as f32,
                    renderer.gpu.surface_config.height as f32,
                );

                let gui_input = gui_state.take_egui_input(window);
                gui_state.egui_ctx().begin_pass(gui_input);

                #[cfg(feature = "webgl")]
                let title = "Rust/Wgpu/Webgl";

                #[cfg(all(feature = "webgpu", not(feature = "webgl")))]
                let title = "Rust/Wgpu/Webgpu";

                #[cfg(all(not(feature = "webgpu"), not(feature = "webgl")))]
                let title = "Rust/Wgpu";

                {
                    egui::TopBottomPanel::top("top").show(gui_state.egui_ctx(), |ui| {
                        ui.horizontal(|ui| {
                            egui::MenuBar::new().ui(ui, |ui| {
                                ui.menu_button("File", |ui| {
                                    if ui.button("Load").clicked() {
                                        ui.close();
                                    }
                                    if ui.button("Save").clicked() {
                                        ui.close();
                                    }
                                    ui.separator();
                                    if ui.button("Import").clicked() {
                                        ui.close();
                                    }
                                });

                                ui.menu_button("Edit", |ui| {
                                    if ui.button("Clear").clicked() {
                                        ui.close();
                                    }
                                    if ui.button("Reset").clicked() {
                                        ui.close();
                                    }
                                });

                                ui.separator();

                                ui.label(
                                    egui::RichText::new(title).color(egui::Color32::LIGHT_GREEN),
                                );

                                ui.separator();
                            });

                            ui.with_layout(egui::Layout::right_to_left(egui::Align::RIGHT), |ui| {
                                ui.add_space(10.0);
                                ui.label(
                                    egui::RichText::new("v0.1.0").color(egui::Color32::ORANGE),
                                );
                                ui.separator();
                            });
                        });
                    });

                    egui::SidePanel::left("left").show(gui_state.egui_ctx(), |ui| {
                        ui.heading("Scene Tree");
                    });

                    egui::SidePanel::right("right").show(gui_state.egui_ctx(), |ui| {
                        ui.heading("Inspector");
                        ui.separator();

                        ui.collapsing("Render Graph", |ui| {
                            let mut edge_detection_enabled = renderer.is_edge_detection_enabled();
                            if ui
                                .checkbox(&mut edge_detection_enabled, "Edge Detection")
                                .changed()
                            {
                                renderer.set_edge_detection_enabled(edge_detection_enabled);
                            }

                            let mut brightness_contrast_enabled =
                                renderer.is_brightness_contrast_enabled();
                            if ui
                                .checkbox(&mut brightness_contrast_enabled, "Brightness/Contrast")
                                .changed()
                            {
                                renderer
                                    .set_brightness_contrast_enabled(brightness_contrast_enabled);
                            }

                            if brightness_contrast_enabled {
                                let mut brightness = renderer.get_brightness();
                                if ui
                                    .add(
                                        egui::Slider::new(&mut brightness, -0.5..=0.5)
                                            .text("Brightness"),
                                    )
                                    .changed()
                                {
                                    renderer.set_brightness(brightness);
                                }

                                let mut contrast = renderer.get_contrast();
                                if ui
                                    .add(
                                        egui::Slider::new(&mut contrast, 0.0..=2.0)
                                            .text("Contrast"),
                                    )
                                    .changed()
                                {
                                    renderer.set_contrast(contrast);
                                }
                            }

                            let mut gaussian_blur_enabled = renderer.is_gaussian_blur_enabled();
                            if ui
                                .checkbox(&mut gaussian_blur_enabled, "Gaussian Blur")
                                .changed()
                            {
                                renderer.set_gaussian_blur_enabled(gaussian_blur_enabled);
                            }

                            let mut sharpen_enabled = renderer.is_sharpen_enabled();
                            if ui.checkbox(&mut sharpen_enabled, "Sharpen").changed() {
                                renderer.set_sharpen_enabled(sharpen_enabled);
                            }

                            if sharpen_enabled {
                                let mut sharpen_strength = renderer.get_sharpen_strength();
                                if ui
                                    .add(
                                        egui::Slider::new(&mut sharpen_strength, 0.0..=2.0)
                                            .text("Strength"),
                                    )
                                    .changed()
                                {
                                    renderer.set_sharpen_strength(sharpen_strength);
                                }
                            }

                            let mut convolution_enabled = renderer.is_convolution_enabled();
                            if ui
                                .checkbox(&mut convolution_enabled, "Convolution")
                                .changed()
                            {
                                renderer.set_convolution_enabled(convolution_enabled);
                            }

                            if convolution_enabled {
                                ui.label("Kernel:");
                                if ui.button("Identity").clicked() {
                                    renderer.set_convolution_kernel([
                                        0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                    ]);
                                }
                                if ui.button("Sharpen").clicked() {
                                    renderer.set_convolution_kernel([
                                        0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0,
                                    ]);
                                }
                                if ui.button("Edge Detect").clicked() {
                                    renderer.set_convolution_kernel([
                                        -1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0,
                                    ]);
                                }
                                if ui.button("Box Blur").clicked() {
                                    renderer.set_convolution_kernel([
                                        1.0 / 9.0,
                                        1.0 / 9.0,
                                        1.0 / 9.0,
                                        1.0 / 9.0,
                                        1.0 / 9.0,
                                        1.0 / 9.0,
                                        1.0 / 9.0,
                                        1.0 / 9.0,
                                        1.0 / 9.0,
                                    ]);
                                }
                                if ui.button("Emboss").clicked() {
                                    renderer.set_convolution_kernel([
                                        -2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0,
                                    ]);
                                }
                            }

                            let mut vignette_enabled = renderer.is_vignette_enabled();
                            if ui.checkbox(&mut vignette_enabled, "Vignette").changed() {
                                renderer.set_vignette_enabled(vignette_enabled);
                            }

                            if vignette_enabled {
                                let mut strength = renderer.get_vignette_strength();
                                if ui
                                    .add(
                                        egui::Slider::new(&mut strength, 0.5..=3.0)
                                            .text("Strength"),
                                    )
                                    .changed()
                                {
                                    renderer.set_vignette_strength(strength);
                                }

                                let mut radius = renderer.get_vignette_radius();
                                if ui
                                    .add(egui::Slider::new(&mut radius, 0.0..=0.8).text("Radius"))
                                    .changed()
                                {
                                    renderer.set_vignette_radius(radius);
                                }

                                let mut color_tint = renderer.get_vignette_color_tint();
                                let mut egui_color_bytes = [
                                    (color_tint[0] * 255.0) as u8,
                                    (color_tint[1] * 255.0) as u8,
                                    (color_tint[2] * 255.0) as u8,
                                ];
                                if ui.color_edit_button_srgb(&mut egui_color_bytes).changed() {
                                    color_tint = [
                                        egui_color_bytes[0] as f32 / 255.0,
                                        egui_color_bytes[1] as f32 / 255.0,
                                        egui_color_bytes[2] as f32 / 255.0,
                                    ];
                                    renderer.set_vignette_color_tint(color_tint);
                                }
                            }

                            let mut grayscale_enabled = renderer.is_grayscale_enabled();
                            if ui.checkbox(&mut grayscale_enabled, "Grayscale").changed() {
                                renderer.set_grayscale_enabled(grayscale_enabled);
                            }

                            let mut color_invert_enabled = renderer.is_color_invert_enabled();
                            if ui
                                .checkbox(&mut color_invert_enabled, "Color Invert")
                                .changed()
                            {
                                renderer.set_color_invert_enabled(color_invert_enabled);
                            }
                        });
                    });

                    egui::TopBottomPanel::bottom("Console").show(gui_state.egui_ctx(), |ui| {
                        ui.heading("Console");
                    });

                    egui::CentralPanel::default().show(gui_state.egui_ctx(), |ui| {
                        if let Some(tree) = &mut self.tile_tree {
                            tree.ui(&mut self.tile_tree_behavior, ui);
                        }
                    });

                    if self.tile_tree_behavior.add_viewport_requested {
                        self.tile_tree_behavior.add_viewport_requested = false;

                        if let Some(tree) = &mut self.tile_tree {
                            let viewport_count = tree
                                .tiles
                                .tiles()
                                .filter(|tile| matches!(tile, egui_tiles::Tile::Pane(_)))
                                .count();

                            let new_pane = ViewportPane {
                                name: format!("Viewport {}", viewport_count + 1),
                                selected_camera_id: None,
                            };
                            let new_tile = tree.tiles.insert_pane(new_pane);

                            if let Some(root) = tree.root() {
                                if let Some(egui_tiles::Tile::Container(container)) =
                                    tree.tiles.get_mut(root)
                                {
                                    if matches!(container.kind(), egui_tiles::ContainerKind::Tabs) {
                                        container.add_child(new_tile);
                                    } else {
                                        let old_root = root;
                                        let tabs = egui_tiles::Container::new_tabs(vec![
                                            old_root, new_tile,
                                        ]);
                                        let tabs_id = tree.tiles.insert_container(tabs);
                                        tree.root = Some(tabs_id);
                                    }
                                } else {
                                    let tabs =
                                        egui_tiles::Container::new_tabs(vec![root, new_tile]);
                                    let tabs_id = tree.tiles.insert_container(tabs);
                                    tree.root = Some(tabs_id);
                                }
                            }
                        }
                    }
                }

                let egui_winit::egui::FullOutput {
                    textures_delta,
                    shapes,
                    pixels_per_point,
                    platform_output,
                    ..
                } = gui_state.egui_ctx().end_pass();

                gui_state.handle_platform_output(window, platform_output);

                let paint_jobs = gui_state.egui_ctx().tessellate(shapes, pixels_per_point);

                let screen_descriptor = {
                    let (width, height) = self.last_size;
                    if width == 0 || height == 0 {
                        return;
                    }
                    egui_wgpu::ScreenDescriptor {
                        size_in_pixels: [width, height],
                        pixels_per_point: window.scale_factor() as f32,
                    }
                };

                let viewports = if let Some(tree) = &self.tile_tree {
                    tree.tiles
                        .iter()
                        .filter_map(|(tile_id, tile)| {
                            if let egui_tiles::Tile::Pane(pane) = tile {
                                if let Some(camera_id) = pane.selected_camera_id {
                                    let (width, height) = self
                                        .tile_tree_behavior
                                        .viewport_dimensions
                                        .get(tile_id)
                                        .copied()
                                        .unwrap_or((0.0, 0.0));

                                    Some(ViewportCamera {
                                        tile_id: *tile_id,
                                        camera_id,
                                        viewport_width: width,
                                        viewport_height: height,
                                    })
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                        .collect()
                } else {
                    vec![]
                };

                self.tile_tree_behavior.camera_texture_aspect = renderer.gpu.surface_config.width
                    as f32
                    / renderer.gpu.surface_config.height.max(1) as f32;

                renderer.render_frame(
                    screen_descriptor,
                    paint_jobs,
                    textures_delta,
                    delta_time,
                    viewports,
                    &self.tile_tree_behavior.cameras,
                );

                if let Some(tree) = &self.tile_tree {
                    for (tile_id, texture_id) in tree.tiles.iter().filter_map(|(tile_id, tile)| {
                        if matches!(tile, egui_tiles::Tile::Pane(_)) {
                            renderer
                                .get_viewport_texture_id(*tile_id)
                                .map(|tex_id| (*tile_id, tex_id))
                        } else {
                            None
                        }
                    }) {
                        self.tile_tree_behavior
                            .viewport_texture_ids
                            .insert(tile_id, texture_id);
                    }
                }
            }
            _ => (),
        }

        window.request_redraw();
    }
}

use pass_configs::PassConfigs;
use passes::{
    BlitPass, BlitPassData, BrightnessContrastPass, BrightnessContrastPassData, ColorInvertPass,
    ColorInvertPassData, ConvolutionPass, ConvolutionPassData, EdgeDetectionPass,
    EdgeDetectionPassData, EguiPass, GaussianBlurHorizontalPass, GaussianBlurPassData,
    GaussianBlurVerticalPass, GrayscalePass, GrayscalePassData, PostProcessPass,
    PostProcessPassData, ScenePass, ScenePassData, SharpenPass, SharpenPassData, VignettePass,
    VignettePassData,
};
use wgpu_render_graph::{RenderGraph, ResourceId};

pub struct ViewportRenderTarget {
    pub tile_id: egui_tiles::TileId,
    pub color_texture: wgpu::Texture,
    pub color_view: wgpu::TextureView,
    pub egui_texture_id: Option<egui::TextureId>,
}

pub struct ViewportCamera {
    pub tile_id: egui_tiles::TileId,
    pub camera_id: usize,
    pub viewport_width: f32,
    pub viewport_height: f32,
}

pub struct Renderer {
    gpu: Gpu,
    depth_texture_view: wgpu::TextureView,
    scene: Scene,
    render_graph: RenderGraph<PassConfigs>,
    pass_configs: PassConfigs,
    surface_resource_id: ResourceId,
    depth_resource_id: ResourceId,
    hdr_resource_id: ResourceId,
    output_resource_id: ResourceId,
    output_with_edges_resource_id: ResourceId,
    output_with_brightness_contrast_resource_id: ResourceId,
    blur_horizontal_resource_id: ResourceId,
    blur_vertical_resource_id: ResourceId,
    convolution_resource_id: ResourceId,
    vignette_resource_id: ResourceId,
    grayscale_resource_id: ResourceId,
    color_invert_resource_id: ResourceId,
    viewport_display_resource_id: ResourceId,
    viewport_display_texture: wgpu::Texture,
    viewport_display_view: wgpu::TextureView,
    sharpen_resource_id: ResourceId,
    egui_output_resource_id: ResourceId,
    viewport_texture_id: Option<egui::TextureId>,
    _sharpen_uniform_buffer: Arc<wgpu::Buffer>,
    viewport_targets: HashMap<egui_tiles::TileId, ViewportRenderTarget>,
    camera_render_targets: HashMap<usize, (wgpu::Texture, wgpu::TextureView)>,
}

impl Renderer {
    const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    pub async fn new(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Self {
        let gpu = Gpu::new_async(window, width, height).await;
        let depth_texture_view = gpu.create_depth_texture(width, height);

        let egui_renderer =
            egui_wgpu::Renderer::new(&gpu.device, gpu.surface_config.format, None, 1, false);

        let scene = Scene::new(&gpu.device, wgpu::TextureFormat::Rgba16Float, &gpu.queue);

        let (post_process_pipeline, post_process_bind_group_layout) =
            PostProcessPass::create_pipeline(&gpu.device, gpu.surface_format);

        let post_process_sampler = Arc::new(gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Post Process Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        let post_process_data = PostProcessPassData {
            pipeline: post_process_pipeline,
            bind_group_layout: post_process_bind_group_layout,
            sampler: post_process_sampler,
        };

        let (edge_detection_pipeline, edge_detection_bind_group_layout) =
            EdgeDetectionPass::create_pipeline(&gpu.device, gpu.surface_format);

        let edge_detection_sampler =
            Arc::new(gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Edge Detection Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            }));

        let (blit_pipeline, blit_bind_group_layout) =
            BlitPass::create_pipeline(&gpu.device, gpu.surface_format);

        let edge_detection_data = EdgeDetectionPassData {
            pipeline: edge_detection_pipeline,
            blit_pipeline: Arc::clone(&blit_pipeline),
            bind_group_layout: edge_detection_bind_group_layout,
            sampler: edge_detection_sampler,
        };

        let blit_sampler = Arc::new(gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Blit Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        let (brightness_contrast_pipeline, brightness_contrast_bind_group_layout) =
            BrightnessContrastPass::create_pipeline(&gpu.device, gpu.surface_format);

        let brightness_contrast_uniform_buffer =
            Arc::new(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Brightness/Contrast Uniform Buffer"),
                size: std::mem::size_of::<[f32; 2]>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

        let brightness_contrast_sampler =
            Arc::new(gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Brightness/Contrast Sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                ..Default::default()
            }));

        let brightness_contrast_data = BrightnessContrastPassData {
            pipeline: brightness_contrast_pipeline,
            blit_pipeline: Arc::clone(&blit_pipeline),
            bind_group_layout: brightness_contrast_bind_group_layout,
            sampler: brightness_contrast_sampler,
        };

        let (gaussian_blur_pipeline, gaussian_blur_bind_group_layout) =
            GaussianBlurHorizontalPass::create_pipeline(&gpu.device, gpu.surface_format);

        let gaussian_blur_horizontal_uniform_buffer =
            Arc::new(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Gaussian Blur Horizontal Uniform Buffer"),
                size: std::mem::size_of::<[f32; 2]>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

        let gaussian_blur_vertical_uniform_buffer =
            Arc::new(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Gaussian Blur Vertical Uniform Buffer"),
                size: std::mem::size_of::<[f32; 2]>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

        let gaussian_blur_sampler = Arc::new(gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Gaussian Blur Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        let gaussian_blur_data = GaussianBlurPassData {
            pipeline: gaussian_blur_pipeline,
            blit_pipeline: Arc::clone(&blit_pipeline),
            bind_group_layout: gaussian_blur_bind_group_layout,
            sampler: gaussian_blur_sampler,
        };

        let (sharpen_pipeline, sharpen_bind_group_layout) =
            SharpenPass::create_pipeline(&gpu.device, gpu.surface_format);

        let sharpen_sampler = Arc::new(gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Sharpen Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        let sharpen_uniform_buffer = Arc::new(gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sharpen Uniform Buffer"),
            size: std::mem::size_of::<[f32; 4]>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let sharpen_data = SharpenPassData {
            pipeline: sharpen_pipeline,
            blit_pipeline: Arc::clone(&blit_pipeline),
            bind_group_layout: sharpen_bind_group_layout,
            sampler: sharpen_sampler,
        };

        let (convolution_pipeline, convolution_bind_group_layout) =
            ConvolutionPass::create_pipeline(&gpu.device, gpu.surface_format);

        let convolution_kernel_buffer =
            Arc::new(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Convolution Kernel Buffer"),
                size: std::mem::size_of::<[[f32; 4]; 3]>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

        let convolution_sampler = Arc::new(gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Convolution Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        let convolution_data = ConvolutionPassData {
            pipeline: convolution_pipeline,
            blit_pipeline: Arc::clone(&blit_pipeline),
            bind_group_layout: convolution_bind_group_layout,
            sampler: convolution_sampler,
        };

        let (vignette_pipeline, vignette_bind_group_layout) =
            VignettePass::create_pipeline(&gpu.device, gpu.surface_format);

        let vignette_uniform_buffer = Arc::new(gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Vignette Uniform Buffer"),
            size: std::mem::size_of::<[f32; 8]>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let vignette_sampler = Arc::new(gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Vignette Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        let vignette_data = VignettePassData {
            pipeline: vignette_pipeline,
            blit_pipeline: Arc::clone(&blit_pipeline),
            bind_group_layout: vignette_bind_group_layout,
            sampler: vignette_sampler,
        };

        let (grayscale_pipeline, grayscale_bind_group_layout) =
            GrayscalePass::create_pipeline(&gpu.device, gpu.surface_format);

        let grayscale_sampler = Arc::new(gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Grayscale Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        let grayscale_data = GrayscalePassData {
            pipeline: grayscale_pipeline,
            blit_pipeline: Arc::clone(&blit_pipeline),
            bind_group_layout: grayscale_bind_group_layout,
            sampler: grayscale_sampler,
        };

        let (color_invert_pipeline, color_invert_bind_group_layout) =
            ColorInvertPass::create_pipeline(&gpu.device, gpu.surface_format);

        let color_invert_sampler = Arc::new(gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Color Invert Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        }));

        let color_invert_data = ColorInvertPassData {
            pipeline: color_invert_pipeline,
            blit_pipeline: Arc::clone(&blit_pipeline),
            bind_group_layout: color_invert_bind_group_layout,
            sampler: color_invert_sampler,
        };

        let blit_data = BlitPassData {
            pipeline: blit_pipeline,
            bind_group_layout: blit_bind_group_layout,
            sampler: blit_sampler,
        };

        let mut graph = RenderGraph::new();

        let output_resource_id = graph
            .add_color_texture("output")
            .format(gpu.surface_format)
            .size(gpu.surface_config.width, gpu.surface_config.height)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
            .sample_count(1)
            .mip_levels(1)
            .transient();

        let output_with_edges_resource_id = graph
            .add_color_texture("output_with_edges")
            .format(gpu.surface_format)
            .size(gpu.surface_config.width, gpu.surface_config.height)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
            .sample_count(1)
            .mip_levels(1)
            .transient();

        let output_with_brightness_contrast_resource_id = graph
            .add_color_texture("output_with_brightness_contrast")
            .format(gpu.surface_format)
            .size(gpu.surface_config.width, gpu.surface_config.height)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
            .sample_count(1)
            .mip_levels(1)
            .transient();

        let blur_horizontal_resource_id = graph
            .add_color_texture("blur_horizontal")
            .format(gpu.surface_format)
            .size(gpu.surface_config.width, gpu.surface_config.height)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
            .sample_count(1)
            .mip_levels(1)
            .transient();

        let blur_vertical_resource_id = graph
            .add_color_texture("blur_vertical")
            .format(gpu.surface_format)
            .size(gpu.surface_config.width, gpu.surface_config.height)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
            .sample_count(1)
            .mip_levels(1)
            .transient();

        let convolution_resource_id = graph
            .add_color_texture("convolution")
            .format(gpu.surface_format)
            .size(gpu.surface_config.width, gpu.surface_config.height)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
            .sample_count(1)
            .mip_levels(1)
            .transient();

        let vignette_resource_id = graph
            .add_color_texture("vignette")
            .format(gpu.surface_format)
            .size(gpu.surface_config.width, gpu.surface_config.height)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
            .sample_count(1)
            .mip_levels(1)
            .transient();

        let grayscale_resource_id = graph
            .add_color_texture("grayscale")
            .format(gpu.surface_format)
            .size(gpu.surface_config.width, gpu.surface_config.height)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
            .sample_count(1)
            .mip_levels(1)
            .transient();

        let color_invert_resource_id = graph
            .add_color_texture("color_invert")
            .format(gpu.surface_format)
            .size(gpu.surface_config.width, gpu.surface_config.height)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
            .sample_count(1)
            .mip_levels(1)
            .transient();

        let viewport_display_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Viewport Display Texture"),
            size: wgpu::Extent3d {
                width: gpu.surface_config.width,
                height: gpu.surface_config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: gpu.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let viewport_display_view =
            viewport_display_texture.create_view(&wgpu::TextureViewDescriptor::default());

        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &viewport_display_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &vec![0u8; (gpu.surface_config.width * gpu.surface_config.height * 4) as usize],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(gpu.surface_config.width * 4),
                rows_per_image: Some(gpu.surface_config.height),
            },
            wgpu::Extent3d {
                width: gpu.surface_config.width,
                height: gpu.surface_config.height,
                depth_or_array_layers: 1,
            },
        );

        let viewport_display_resource_id = graph
            .add_color_texture("viewport_display")
            .format(gpu.surface_format)
            .size(gpu.surface_config.width, gpu.surface_config.height)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
            .sample_count(1)
            .mip_levels(1)
            .external();

        let sharpen_resource_id = graph
            .add_color_texture("sharpen")
            .format(gpu.surface_format)
            .size(gpu.surface_config.width, gpu.surface_config.height)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
            .sample_count(1)
            .mip_levels(1)
            .transient();

        let egui_output_resource_id = graph
            .add_color_texture("egui_output")
            .format(gpu.surface_format)
            .size(gpu.surface_config.width, gpu.surface_config.height)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
            .sample_count(1)
            .mip_levels(1)
            .clear_color(wgpu::Color {
                r: 0.3,
                g: 0.3,
                b: 0.3,
                a: 1.0,
            })
            .transient();

        let surface_resource_id = graph
            .add_color_texture("surface")
            .format(gpu.surface_format)
            .size(gpu.surface_config.width, gpu.surface_config.height)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT)
            .sample_count(1)
            .mip_levels(1)
            .external();

        let hdr_resource_id = graph
            .add_color_texture("hdr_buffer")
            .format(wgpu::TextureFormat::Rgba16Float)
            .size(gpu.surface_config.width, gpu.surface_config.height)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
            .sample_count(1)
            .mip_levels(1)
            .clear_color(wgpu::Color {
                r: 0.5,
                g: 0.5,
                b: 0.5,
                a: 1.0,
            })
            .transient();

        let depth_resource_id = graph
            .add_depth_texture("depth")
            .format(wgpu::TextureFormat::Depth32Float)
            .size(gpu.surface_config.width, gpu.surface_config.height)
            .usage(wgpu::TextureUsages::RENDER_ATTACHMENT)
            .sample_count(1)
            .mip_levels(1)
            .clear_depth(1.0)
            .external();

        graph
            .add_pass(
                Box::new(ScenePass::new(ScenePassData {
                    pipeline: Arc::clone(&scene.pipeline),
                    vertex_buffer: Arc::clone(&scene.vertex_buffer),
                    index_buffer: Arc::clone(&scene.index_buffer),
                    index_count: INDICES.len() as u32,
                    uniform_bind_group: Arc::clone(&scene.uniform.bind_group),
                    texture_bind_group: Arc::clone(&scene.texture_bind_group),
                })),
                &[
                    ("color_output", hdr_resource_id),
                    ("depth_output", depth_resource_id),
                ],
            )
            .expect("Failed to add scene pass");

        graph
            .add_pass(
                Box::new(PostProcessPass::new(PostProcessPassData {
                    pipeline: Arc::clone(&post_process_data.pipeline),
                    bind_group_layout: Arc::clone(&post_process_data.bind_group_layout),
                    sampler: Arc::clone(&post_process_data.sampler),
                })),
                &[
                    ("hdr_input", hdr_resource_id),
                    ("color_output", output_resource_id),
                ],
            )
            .expect("Failed to add post process pass");

        graph
            .add_pass(
                Box::new(EdgeDetectionPass::new(EdgeDetectionPassData {
                    pipeline: Arc::clone(&edge_detection_data.pipeline),
                    blit_pipeline: Arc::clone(&edge_detection_data.blit_pipeline),
                    bind_group_layout: Arc::clone(&edge_detection_data.bind_group_layout),
                    sampler: Arc::clone(&edge_detection_data.sampler),
                })),
                &[
                    ("input", output_resource_id),
                    ("output", output_with_edges_resource_id),
                ],
            )
            .expect("Failed to add edge detection pass");

        graph
            .add_pass(
                Box::new(BrightnessContrastPass::new(
                    BrightnessContrastPassData {
                        pipeline: Arc::clone(&brightness_contrast_data.pipeline),
                        blit_pipeline: Arc::clone(&brightness_contrast_data.blit_pipeline),
                        bind_group_layout: Arc::clone(&brightness_contrast_data.bind_group_layout),
                        sampler: Arc::clone(&brightness_contrast_data.sampler),
                    },
                    brightness_contrast_uniform_buffer,
                )),
                &[
                    ("input", output_with_edges_resource_id),
                    ("output", output_with_brightness_contrast_resource_id),
                ],
            )
            .expect("Failed to add brightness contrast pass");

        graph
            .add_pass(
                Box::new(GaussianBlurHorizontalPass::new(
                    GaussianBlurPassData {
                        pipeline: Arc::clone(&gaussian_blur_data.pipeline),
                        blit_pipeline: Arc::clone(&gaussian_blur_data.blit_pipeline),
                        bind_group_layout: Arc::clone(&gaussian_blur_data.bind_group_layout),
                        sampler: Arc::clone(&gaussian_blur_data.sampler),
                    },
                    gaussian_blur_horizontal_uniform_buffer,
                )),
                &[
                    ("input", output_with_brightness_contrast_resource_id),
                    ("output", blur_horizontal_resource_id),
                ],
            )
            .expect("Failed to add gaussian blur horizontal pass");

        graph
            .add_pass(
                Box::new(GaussianBlurVerticalPass::new(
                    GaussianBlurPassData {
                        pipeline: Arc::clone(&gaussian_blur_data.pipeline),
                        blit_pipeline: Arc::clone(&gaussian_blur_data.blit_pipeline),
                        bind_group_layout: Arc::clone(&gaussian_blur_data.bind_group_layout),
                        sampler: Arc::clone(&gaussian_blur_data.sampler),
                    },
                    gaussian_blur_vertical_uniform_buffer,
                )),
                &[
                    ("input", blur_horizontal_resource_id),
                    ("output", blur_vertical_resource_id),
                ],
            )
            .expect("Failed to add gaussian blur vertical pass");

        graph
            .add_pass(
                Box::new(SharpenPass::new(
                    SharpenPassData {
                        pipeline: Arc::clone(&sharpen_data.pipeline),
                        blit_pipeline: Arc::clone(&sharpen_data.blit_pipeline),
                        bind_group_layout: Arc::clone(&sharpen_data.bind_group_layout),
                        sampler: Arc::clone(&sharpen_data.sampler),
                    },
                    Arc::clone(&sharpen_uniform_buffer),
                )),
                &[
                    ("input", blur_vertical_resource_id),
                    ("output", sharpen_resource_id),
                ],
            )
            .expect("Failed to add sharpen pass");

        graph
            .add_pass(
                Box::new(ConvolutionPass::new(
                    ConvolutionPassData {
                        pipeline: Arc::clone(&convolution_data.pipeline),
                        blit_pipeline: Arc::clone(&convolution_data.blit_pipeline),
                        bind_group_layout: Arc::clone(&convolution_data.bind_group_layout),
                        sampler: Arc::clone(&convolution_data.sampler),
                    },
                    convolution_kernel_buffer,
                )),
                &[
                    ("input", sharpen_resource_id),
                    ("output", convolution_resource_id),
                ],
            )
            .expect("Failed to add convolution pass");

        graph
            .add_pass(
                Box::new(VignettePass::new(
                    VignettePassData {
                        pipeline: Arc::clone(&vignette_data.pipeline),
                        blit_pipeline: Arc::clone(&vignette_data.blit_pipeline),
                        bind_group_layout: Arc::clone(&vignette_data.bind_group_layout),
                        sampler: Arc::clone(&vignette_data.sampler),
                    },
                    vignette_uniform_buffer,
                )),
                &[
                    ("input", convolution_resource_id),
                    ("output", vignette_resource_id),
                ],
            )
            .expect("Failed to add vignette pass");

        graph
            .add_pass(
                Box::new(GrayscalePass::new(GrayscalePassData {
                    pipeline: Arc::clone(&grayscale_data.pipeline),
                    blit_pipeline: Arc::clone(&grayscale_data.blit_pipeline),
                    bind_group_layout: Arc::clone(&grayscale_data.bind_group_layout),
                    sampler: Arc::clone(&grayscale_data.sampler),
                })),
                &[
                    ("input", vignette_resource_id),
                    ("output", grayscale_resource_id),
                ],
            )
            .expect("Failed to add grayscale pass");

        graph
            .add_pass(
                Box::new(ColorInvertPass::new(ColorInvertPassData {
                    pipeline: Arc::clone(&color_invert_data.pipeline),
                    blit_pipeline: Arc::clone(&color_invert_data.blit_pipeline),
                    bind_group_layout: Arc::clone(&color_invert_data.bind_group_layout),
                    sampler: Arc::clone(&color_invert_data.sampler),
                })),
                &[
                    ("input", grayscale_resource_id),
                    ("output", color_invert_resource_id),
                ],
            )
            .expect("Failed to add color invert pass");

        graph
            .add_pass(
                Box::new(BlitPass::new(
                    BlitPassData {
                        pipeline: Arc::clone(&blit_data.pipeline),
                        bind_group_layout: Arc::clone(&blit_data.bind_group_layout),
                        sampler: Arc::clone(&blit_data.sampler),
                    },
                    "blit_to_viewport_display".to_string(),
                )),
                &[
                    ("input", color_invert_resource_id),
                    ("output", viewport_display_resource_id),
                ],
            )
            .expect("Failed to add blit to viewport pass");

        graph
            .add_pass(
                Box::new(EguiPass::new()),
                &[("color_target", egui_output_resource_id)],
            )
            .expect("Failed to add egui pass");

        graph
            .add_pass(
                Box::new(BlitPass::new(
                    BlitPassData {
                        pipeline: Arc::clone(&blit_data.pipeline),
                        bind_group_layout: Arc::clone(&blit_data.bind_group_layout),
                        sampler: Arc::clone(&blit_data.sampler),
                    },
                    "blit_to_surface".to_string(),
                )),
                &[
                    ("input", egui_output_resource_id),
                    ("output", surface_resource_id),
                ],
            )
            .expect("Failed to add blit to surface pass");

        graph.compile().expect("Failed to compile render graph");

        log::info!("Render graph compiled successfully");

        let mut pass_configs = PassConfigs::default();
        pass_configs.egui.renderer = Some(egui_renderer);

        Self {
            gpu,
            depth_texture_view,
            scene,
            render_graph: graph,
            pass_configs,
            surface_resource_id,
            depth_resource_id,
            hdr_resource_id,
            output_resource_id,
            output_with_edges_resource_id,
            output_with_brightness_contrast_resource_id,
            blur_horizontal_resource_id,
            blur_vertical_resource_id,
            convolution_resource_id,
            vignette_resource_id,
            grayscale_resource_id,
            color_invert_resource_id,
            viewport_display_resource_id,
            viewport_display_texture,
            viewport_display_view,
            sharpen_resource_id,
            egui_output_resource_id,
            viewport_texture_id: None,
            _sharpen_uniform_buffer: sharpen_uniform_buffer,
            viewport_targets: HashMap::new(),
            camera_render_targets: HashMap::new(),
        }
    }

    pub fn set_edge_detection_enabled(&mut self, enabled: bool) {
        self.pass_configs.edge_detection.enabled = enabled;
    }

    pub fn is_edge_detection_enabled(&mut self) -> bool {
        self.pass_configs.edge_detection.enabled
    }

    pub fn set_brightness_contrast_enabled(&mut self, enabled: bool) {
        self.pass_configs.brightness_contrast.enabled = enabled;
    }

    pub fn is_brightness_contrast_enabled(&mut self) -> bool {
        self.pass_configs.brightness_contrast.enabled
    }

    pub fn set_brightness(&mut self, brightness: f32) {
        self.pass_configs.brightness_contrast.brightness = brightness;
    }

    pub fn get_brightness(&mut self) -> f32 {
        self.pass_configs.brightness_contrast.brightness
    }

    pub fn set_contrast(&mut self, contrast: f32) {
        self.pass_configs.brightness_contrast.contrast = contrast;
    }

    pub fn get_contrast(&mut self) -> f32 {
        self.pass_configs.brightness_contrast.contrast
    }

    pub fn set_gaussian_blur_enabled(&mut self, enabled: bool) {
        self.pass_configs.gaussian_blur.enabled = enabled;
    }

    pub fn is_gaussian_blur_enabled(&mut self) -> bool {
        self.pass_configs.gaussian_blur.enabled
    }

    pub fn set_sharpen_enabled(&mut self, enabled: bool) {
        self.pass_configs.sharpen.enabled = enabled;
    }

    pub fn is_sharpen_enabled(&mut self) -> bool {
        self.pass_configs.sharpen.enabled
    }

    pub fn set_sharpen_strength(&mut self, strength: f32) {
        self.pass_configs.sharpen.strength = strength;
    }

    pub fn get_sharpen_strength(&mut self) -> f32 {
        self.pass_configs.sharpen.strength
    }

    pub fn set_convolution_enabled(&mut self, enabled: bool) {
        self.pass_configs.convolution.enabled = enabled;
    }

    pub fn is_convolution_enabled(&mut self) -> bool {
        self.pass_configs.convolution.enabled
    }

    pub fn set_convolution_kernel(&mut self, kernel: [f32; 9]) {
        self.pass_configs.convolution.kernel = kernel;
    }

    pub fn get_convolution_kernel(&mut self) -> [f32; 9] {
        self.pass_configs.convolution.kernel
    }

    pub fn set_vignette_enabled(&mut self, enabled: bool) {
        self.pass_configs.vignette.enabled = enabled;
    }

    pub fn is_vignette_enabled(&mut self) -> bool {
        self.pass_configs.vignette.enabled
    }

    pub fn set_vignette_strength(&mut self, strength: f32) {
        self.pass_configs.vignette.strength = strength;
    }

    pub fn get_vignette_strength(&mut self) -> f32 {
        self.pass_configs.vignette.strength
    }

    pub fn set_vignette_radius(&mut self, radius: f32) {
        self.pass_configs.vignette.radius = radius;
    }

    pub fn get_vignette_radius(&mut self) -> f32 {
        self.pass_configs.vignette.radius
    }

    pub fn set_vignette_color_tint(&mut self, color_tint: [f32; 3]) {
        self.pass_configs.vignette.color_tint = color_tint;
    }

    pub fn get_vignette_color_tint(&mut self) -> [f32; 3] {
        self.pass_configs.vignette.color_tint
    }

    pub fn set_grayscale_enabled(&mut self, enabled: bool) {
        self.pass_configs.grayscale.enabled = enabled;
    }

    pub fn is_grayscale_enabled(&mut self) -> bool {
        self.pass_configs.grayscale.enabled
    }

    pub fn set_color_invert_enabled(&mut self, enabled: bool) {
        self.pass_configs.color_invert.enabled = enabled;
    }

    pub fn is_color_invert_enabled(&mut self) -> bool {
        self.pass_configs.color_invert.enabled
    }

    pub fn viewport_texture_id(&self) -> Option<egui::TextureId> {
        self.viewport_texture_id
    }

    pub fn get_viewport_texture_id(&self, tile_id: egui_tiles::TileId) -> Option<egui::TextureId> {
        self.viewport_targets
            .get(&tile_id)
            .and_then(|target| target.egui_texture_id)
    }

    fn ensure_viewport_target(&mut self, tile_id: egui_tiles::TileId, width: u32, height: u32) {
        let needs_create = if let Some(existing) = self.viewport_targets.get(&tile_id) {
            existing.color_texture.width() != width || existing.color_texture.height() != height
        } else {
            true
        };

        if needs_create {
            let color_texture = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Viewport Color Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.gpu.surface_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

            let target = ViewportRenderTarget {
                tile_id,
                color_texture,
                color_view,
                egui_texture_id: None,
            };

            self.viewport_targets.insert(tile_id, target);
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.gpu.resize(width, height);
        self.depth_texture_view = self.gpu.create_depth_texture(width, height);

        self.render_graph
            .resize_transient_resource(&self.gpu.device, self.hdr_resource_id, width, height)
            .expect("Failed to resize hdr resource");

        self.render_graph
            .resize_transient_resource(&self.gpu.device, self.output_resource_id, width, height)
            .expect("Failed to resize output resource");

        self.render_graph
            .resize_transient_resource(
                &self.gpu.device,
                self.output_with_edges_resource_id,
                width,
                height,
            )
            .expect("Failed to resize output_with_edges resource");

        self.render_graph
            .resize_transient_resource(
                &self.gpu.device,
                self.output_with_brightness_contrast_resource_id,
                width,
                height,
            )
            .expect("Failed to resize output_with_brightness_contrast resource");

        self.render_graph
            .resize_transient_resource(
                &self.gpu.device,
                self.blur_horizontal_resource_id,
                width,
                height,
            )
            .expect("Failed to resize blur_horizontal resource");

        self.render_graph
            .resize_transient_resource(
                &self.gpu.device,
                self.blur_vertical_resource_id,
                width,
                height,
            )
            .expect("Failed to resize blur_vertical resource");

        self.render_graph
            .resize_transient_resource(
                &self.gpu.device,
                self.convolution_resource_id,
                width,
                height,
            )
            .expect("Failed to resize convolution resource");

        self.render_graph
            .resize_transient_resource(&self.gpu.device, self.vignette_resource_id, width, height)
            .expect("Failed to resize vignette resource");

        self.render_graph
            .resize_transient_resource(&self.gpu.device, self.grayscale_resource_id, width, height)
            .expect("Failed to resize grayscale resource");

        self.render_graph
            .resize_transient_resource(
                &self.gpu.device,
                self.color_invert_resource_id,
                width,
                height,
            )
            .expect("Failed to resize color_invert resource");

        self.render_graph
            .resize_transient_resource(&self.gpu.device, self.sharpen_resource_id, width, height)
            .expect("Failed to resize sharpen resource");

        self.render_graph
            .resize_transient_resource(
                &self.gpu.device,
                self.egui_output_resource_id,
                width,
                height,
            )
            .expect("Failed to resize egui_output resource");

        self.viewport_display_texture = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Viewport Display Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.gpu.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.viewport_display_view = self
            .viewport_display_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.viewport_display_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &vec![0u8; (width * height * 4) as usize],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        if let Some(old_texture_id) = self.viewport_texture_id
            && let Some(renderer) = &mut self.pass_configs.egui.renderer
        {
            renderer.free_texture(&old_texture_id);
            self.viewport_texture_id = Some(renderer.register_native_texture(
                &self.gpu.device,
                &self.viewport_display_view,
                wgpu::FilterMode::Linear,
            ));
        }
    }

    pub fn render_frame(
        &mut self,
        screen_descriptor: egui_wgpu::ScreenDescriptor,
        paint_jobs: Vec<egui::epaint::ClippedPrimitive>,
        textures_delta: egui::TexturesDelta,
        delta_time: crate::Duration,
        viewports: Vec<ViewportCamera>,
        cameras: &[Camera],
    ) {
        let delta_time = delta_time.as_secs_f32();

        if let Some(renderer) = &mut self.pass_configs.egui.renderer {
            for (id, image_delta) in &textures_delta.set {
                renderer.update_texture(&self.gpu.device, &self.gpu.queue, *id, image_delta);
            }

            for id in &textures_delta.free {
                renderer.free_texture(id);
            }
        }

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        if let Some(renderer) = &mut self.pass_configs.egui.renderer {
            renderer.update_buffers(
                &self.gpu.device,
                &self.gpu.queue,
                &mut encoder,
                &paint_jobs,
                &screen_descriptor,
            );
        }

        let camera_render_size = (
            self.gpu.surface_config.width as f32,
            self.gpu.surface_config.height as f32,
        );

        let mut unique_cameras: HashMap<usize, (Camera, f32, f32)> = HashMap::new();

        for camera in cameras {
            unique_cameras.insert(
                camera.id,
                (camera.clone(), camera_render_size.0, camera_render_size.1),
            );
        }

        for (camera_id, (_camera, width, height)) in &unique_cameras {
            let width = width.ceil() as u32;
            let height = height.ceil() as u32;

            let needs_create =
                if let Some((existing_texture, _)) = self.camera_render_targets.get(camera_id) {
                    existing_texture.width() != width || existing_texture.height() != height
                } else {
                    true
                };

            if needs_create {
                let camera_texture = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(&format!("Camera {} Texture", camera_id)),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.gpu.surface_format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                });
                let camera_texture_view =
                    camera_texture.create_view(&wgpu::TextureViewDescriptor::default());

                self.camera_render_targets
                    .insert(*camera_id, (camera_texture, camera_texture_view));
            }
        }

        self.scene.model = nalgebra_glm::rotate(
            &self.scene.model,
            30_f32.to_radians() * delta_time,
            &nalgebra_glm::Vec3::y(),
        );

        for (camera_id, (camera, width, height)) in unique_cameras {
            let width = width.ceil() as u32;
            let height = height.ceil() as u32;

            let (_, camera_texture_view) = self.camera_render_targets.get(&camera_id).unwrap();

            let camera_depth = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Camera {} Depth", camera_id)),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Self::DEPTH_FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let camera_depth_view =
                camera_depth.create_view(&wgpu::TextureViewDescriptor::default());

            let aspect_ratio = width as f32 / height as f32;
            self.scene
                .update_with_camera(&self.gpu.queue, aspect_ratio, 0.0, &camera);

            let dummy_surface = self.gpu.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Dummy Surface for Camera Rendering"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.gpu.surface_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let dummy_surface_view =
                dummy_surface.create_view(&wgpu::TextureViewDescriptor::default());

            self.render_graph
                .resources_mut()
                .set_external_texture(self.depth_resource_id, camera_depth_view);
            self.render_graph.resources_mut().set_external_texture(
                self.viewport_display_resource_id,
                camera_texture_view.clone(),
            );
            self.render_graph
                .resources_mut()
                .set_external_texture(self.surface_resource_id, dummy_surface_view);

            let saved_paint_jobs = std::mem::take(&mut self.pass_configs.egui.paint_jobs);
            let saved_screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: self.pass_configs.egui.screen_descriptor.size_in_pixels,
                pixels_per_point: self.pass_configs.egui.screen_descriptor.pixels_per_point,
            };
            self.pass_configs.egui.screen_descriptor = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [width, height],
                pixels_per_point: 1.0,
            };
            let camera_command_buffers = self
                .render_graph
                .execute(&self.gpu.device, &self.gpu.queue, &self.pass_configs)
                .expect("Failed to execute render graph for camera");
            self.pass_configs.egui.paint_jobs = saved_paint_jobs;
            self.pass_configs.egui.screen_descriptor = saved_screen_descriptor;
            self.gpu.queue.submit(camera_command_buffers);
        }

        for viewport in &viewports {
            self.ensure_viewport_target(
                viewport.tile_id,
                self.gpu.surface_config.width,
                self.gpu.surface_config.height,
            );

            if let Some((camera_texture, _)) = self.camera_render_targets.get(&viewport.camera_id) {
                let viewport_target = self.viewport_targets.get(&viewport.tile_id).unwrap();

                let mut copy_encoder =
                    self.gpu
                        .device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Copy Camera to Viewport Encoder"),
                        });

                copy_encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: camera_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &viewport_target.color_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: camera_texture.width(),
                        height: camera_texture.height(),
                        depth_or_array_layers: 1,
                    },
                );

                self.gpu.queue.submit(Some(copy_encoder.finish()));
            }
        }

        let surface_texture = match self.gpu.surface.get_current_texture() {
            Ok(texture) => texture,
            Err(wgpu::SurfaceError::Outdated) => {
                self.gpu
                    .surface
                    .configure(&self.gpu.device, &self.gpu.surface_config);
                self.gpu
                    .surface
                    .get_current_texture()
                    .expect("Failed to get surface texture after reconfiguration!")
            }
            Err(error) => panic!("Failed to get surface texture: {:?}", error),
        };

        let surface_texture_view =
            surface_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor {
                    label: wgpu::Label::default(),
                    aspect: wgpu::TextureAspect::default(),
                    format: Some(self.gpu.surface_format),
                    dimension: None,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: None,
                    usage: None,
                });

        self.render_graph
            .resources_mut()
            .set_external_texture(self.surface_resource_id, surface_texture_view);
        self.render_graph
            .resources_mut()
            .set_external_texture(self.depth_resource_id, self.depth_texture_view.clone());
        self.render_graph.resources_mut().set_external_texture(
            self.viewport_display_resource_id,
            self.viewport_display_view.clone(),
        );

        self.pass_configs.egui.paint_jobs = paint_jobs;
        self.pass_configs.egui.screen_descriptor = screen_descriptor;

        let mut command_buffers = self
            .render_graph
            .execute(&self.gpu.device, &self.gpu.queue, &self.pass_configs)
            .expect("Failed to execute render graph");
        command_buffers.push(encoder.finish());

        self.gpu.queue.submit(command_buffers);

        if let Some(renderer) = &mut self.pass_configs.egui.renderer {
            for viewport_target in self.viewport_targets.values_mut() {
                if viewport_target.egui_texture_id.is_none() {
                    viewport_target.egui_texture_id = Some(renderer.register_native_texture(
                        &self.gpu.device,
                        &viewport_target.color_view,
                        wgpu::FilterMode::Linear,
                    ));
                }
            }

            if self.viewport_texture_id.is_none() {
                self.viewport_texture_id = Some(renderer.register_native_texture(
                    &self.gpu.device,
                    &self.viewport_display_view,
                    wgpu::FilterMode::Linear,
                ));
            }
        }

        surface_texture.present();
    }
}

pub struct Gpu {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub surface_format: wgpu::TextureFormat,
}

impl Gpu {
    pub fn aspect_ratio(&self) -> f32 {
        self.surface_config.width as f32 / self.surface_config.height.max(1) as f32
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    pub fn create_depth_texture(&self, width: u32, height: u32) -> wgpu::TextureView {
        let texture = self.device.create_texture(
            &(wgpu::TextureDescriptor {
                label: Some("Depth Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            }),
        );
        texture.create_view(&wgpu::TextureViewDescriptor {
            label: None,
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            base_array_layer: 0,
            array_layer_count: None,
            mip_level_count: None,
            usage: None,
        })
    }

    pub async fn new_async(
        window: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Self {
        let instance = wgpu::Instance::new(&InstanceDescriptor::default());
        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to request adapter!");
        let (device, queue) = {
            log::info!("WGPU Adapter Features: {:#?}", adapter.features());
            adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("WGPU Device"),
                    memory_hints: wgpu::MemoryHints::default(),
                    required_features: wgpu::Features::default(),
                    #[cfg(not(target_arch = "wasm32"))]
                    required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
                    required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                    #[cfg(all(target_arch = "wasm32", feature = "webgl"))]
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
                    trace: wgpu::Trace::Off,
                })
                .await
                .expect("Failed to request a device!")
        };

        let surface_capabilities = surface.get_capabilities(&adapter);

        let surface_format = surface_capabilities
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(surface_capabilities.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: surface_capabilities.present_modes[0],
            alpha_mode: surface_capabilities.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_config);

        Self {
            surface,
            device,
            queue,
            surface_config,
            surface_format,
        }
    }
}

struct TextureAtlas {
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    width: u32,
    height: u32,
    current_y: u32,
    regions: Vec<TextureRegion>,
}

#[derive(Debug, Clone, Copy)]
struct TextureRegion {
    min_u: f32,
    min_v: f32,
    max_u: f32,
    max_v: f32,
}

impl TextureAtlas {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Texture Atlas"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            texture,
            texture_view,
            sampler,
            width,
            height,
            current_y: 0,
            regions: Vec::new(),
        }
    }

    pub fn add_texture(
        &mut self,
        queue: &wgpu::Queue,
        texture_data: &[u8],
        texture_width: u32,
        texture_height: u32,
    ) -> usize {
        let region_index = self.regions.len();

        if self.current_y + texture_height > self.height {
            panic!("Texture atlas is full");
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: self.current_y,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            texture_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * texture_width),
                rows_per_image: Some(texture_height),
            },
            wgpu::Extent3d {
                width: texture_width,
                height: texture_height,
                depth_or_array_layers: 1,
            },
        );

        let min_u = 0.0;
        let max_u = texture_width as f32 / self.width as f32;
        let min_v = self.current_y as f32 / self.height as f32;
        let max_v = (self.current_y + texture_height) as f32 / self.height as f32;

        self.regions.push(TextureRegion {
            min_u,
            min_v,
            max_u,
            max_v,
        });

        self.current_y += texture_height;

        region_index
    }
}

struct Scene {
    pub model: nalgebra_glm::Mat4,
    pub vertex_buffer: Arc<wgpu::Buffer>,
    pub index_buffer: Arc<wgpu::Buffer>,
    pub uniform: UniformBinding,
    _texture_atlas: TextureAtlas,
    pub texture_bind_group: Arc<wgpu::BindGroup>,
    _atlas_regions_buffer: wgpu::Buffer,
    pub pipeline: Arc<wgpu::RenderPipeline>,
}

impl Scene {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        queue: &wgpu::Queue,
    ) -> Self {
        let vertex_buffer = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("Vertex Buffer"),
                contents: bytemuck::cast_slice(&VERTICES),
                usage: wgpu::BufferUsages::VERTEX,
            },
        );
        let index_buffer = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("index Buffer"),
                contents: bytemuck::cast_slice(&INDICES),
                usage: wgpu::BufferUsages::INDEX,
            },
        );
        let uniform = UniformBinding::new(device);

        let mut texture_atlas = TextureAtlas::new(device, 512, 1024);

        #[cfg(not(target_arch = "wasm32"))]
        let awesomeface_data = {
            let img = image::open("assets/images/awesomeface.png")
                .expect("Failed to load awesomeface.png")
                .to_rgba8();
            let dimensions = img.dimensions();
            (img.into_raw(), dimensions.0, dimensions.1)
        };

        #[cfg(target_arch = "wasm32")]
        let awesomeface_data = {
            let img = image::load_from_memory(include_bytes!("../assets/images/awesomeface.png"))
                .expect("Failed to load awesomeface.png from memory")
                .to_rgba8();
            let dimensions = img.dimensions();
            (img.into_raw(), dimensions.0, dimensions.1)
        };

        texture_atlas.add_texture(
            queue,
            &awesomeface_data.0,
            awesomeface_data.1,
            awesomeface_data.2,
        );

        let checkerboard = Self::create_checkerboard_texture(8);
        texture_atlas.add_texture(queue, &checkerboard, 8, 8);

        let stripes = Self::create_stripes_texture(16);
        texture_atlas.add_texture(queue, &stripes, 16, 16);

        let gradient = Self::create_gradient_texture(32);
        texture_atlas.add_texture(queue, &gradient, 32, 32);

        let regions_data: Vec<[f32; 4]> = texture_atlas
            .regions
            .iter()
            .map(|r| [r.min_u, r.min_v, r.max_u, r.max_v])
            .collect();

        let atlas_regions_buffer = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("Atlas Regions Buffer"),
                contents: bytemuck::cast_slice(&regions_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        );

        let texture_bind_group_layout = Self::create_texture_bind_group_layout(device);
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Atlas Bind Group"),
            layout: &texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_atlas.texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&texture_atlas.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: atlas_regions_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline =
            Self::create_pipeline(device, surface_format, &uniform, &texture_bind_group_layout);

        Self {
            model: nalgebra_glm::Mat4::identity(),
            uniform,
            _texture_atlas: texture_atlas,
            texture_bind_group: Arc::new(texture_bind_group),
            _atlas_regions_buffer: atlas_regions_buffer,
            pipeline: Arc::new(pipeline),
            vertex_buffer: Arc::new(vertex_buffer),
            index_buffer: Arc::new(index_buffer),
        }
    }

    pub fn update_with_camera(
        &mut self,
        queue: &wgpu::Queue,
        aspect_ratio: f32,
        delta_time: f32,
        camera: &Camera,
    ) {
        let projection =
            nalgebra_glm::perspective_lh_zo(aspect_ratio, camera.fov.to_radians(), 0.1, 1000.0);

        let forward = nalgebra_glm::vec3(
            camera.yaw.cos() * camera.pitch.cos(),
            camera.pitch.sin(),
            camera.yaw.sin() * camera.pitch.cos(),
        );
        let target = camera.position + forward;

        let view = nalgebra_glm::look_at_lh(&camera.position, &target, &nalgebra_glm::Vec3::y());

        self.model = nalgebra_glm::rotate(
            &self.model,
            30_f32.to_radians() * delta_time,
            &nalgebra_glm::Vec3::y(),
        );

        self.uniform.update_buffer(
            queue,
            0,
            UniformBuffer {
                mvp: projection * view * self.model,
            },
        );
    }

    fn create_checkerboard_texture(size: u32) -> Vec<u8> {
        let mut texture_data = Vec::with_capacity((size * size * 4) as usize);
        for y in 0..size {
            for x in 0..size {
                let is_white = (x + y) % 2 == 0;
                let color = if is_white { 255u8 } else { 0u8 };
                texture_data.extend_from_slice(&[color, color, color, 255]);
            }
        }
        texture_data
    }

    fn create_stripes_texture(size: u32) -> Vec<u8> {
        let mut texture_data = Vec::with_capacity((size * size * 4) as usize);
        for _y in 0..size {
            for x in 0..size {
                let is_red = x % 2 == 0;
                let color = if is_red {
                    [255u8, 0, 0, 255]
                } else {
                    [0, 0, 255, 255]
                };
                texture_data.extend_from_slice(&color);
            }
        }
        texture_data
    }

    fn create_gradient_texture(size: u32) -> Vec<u8> {
        let mut texture_data = Vec::with_capacity((size * size * 4) as usize);
        for y in 0..size {
            for x in 0..size {
                let r = ((x as f32 / size as f32) * 255.0) as u8;
                let g = ((y as f32 / size as f32) * 255.0) as u8;
                texture_data.extend_from_slice(&[r, g, 0, 255]);
            }
        }
        texture_data
    }

    fn create_texture_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Bind Group Layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    fn create_pipeline(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        uniform: &UniformBinding,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_SOURCE)),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&uniform.bind_group_layout, texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vertex_main"),
                buffers: &[Vertex::description(&Vertex::vertex_attributes())],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Cw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
                unclipped_depth: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Renderer::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
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
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        })
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 4],
    texture_coordinates: [f32; 2],
    texture_index: u32,
    padding: u32,
}

impl Vertex {
    pub fn vertex_attributes() -> Vec<wgpu::VertexAttribute> {
        wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x2, 2 => Uint32, 3 => Uint32].to_vec()
    }

    pub fn description(attributes: &[wgpu::VertexAttribute]) -> wgpu::VertexBufferLayout<'_> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes,
        }
    }
}

#[repr(C)]
#[derive(Default, Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UniformBuffer {
    mvp: nalgebra_glm::Mat4,
}

struct UniformBinding {
    pub buffer: wgpu::Buffer,
    pub bind_group: Arc<wgpu::BindGroup>,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl UniformBinding {
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::cast_slice(&[UniformBuffer::default()]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            },
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("uniform_bind_group_layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group"),
        });

        Self {
            buffer,
            bind_group: Arc::new(bind_group),
            bind_group_layout,
        }
    }

    pub fn update_buffer(
        &mut self,
        queue: &wgpu::Queue,
        offset: wgpu::BufferAddress,
        uniform_buffer: UniformBuffer,
    ) {
        queue.write_buffer(
            &self.buffer,
            offset,
            bytemuck::cast_slice(&[uniform_buffer]),
        )
    }
}

const VERTICES: [Vertex; 10] = [
    Vertex {
        position: [-1.5, -1.0, 0.0, 1.0],
        texture_coordinates: [0.0, 1.0],
        texture_index: 0,
        padding: 0,
    },
    Vertex {
        position: [-0.5, -1.0, 0.0, 1.0],
        texture_coordinates: [1.0, 1.0],
        texture_index: 0,
        padding: 0,
    },
    Vertex {
        position: [-0.5, 0.0, 0.0, 1.0],
        texture_coordinates: [1.0, 0.0],
        texture_index: 0,
        padding: 0,
    },
    Vertex {
        position: [-1.5, 0.0, 0.0, 1.0],
        texture_coordinates: [0.0, 0.0],
        texture_index: 0,
        padding: 0,
    },
    Vertex {
        position: [0.5, -1.0, 0.0, 1.0],
        texture_coordinates: [1.0, 1.0],
        texture_index: 2,
        padding: 0,
    },
    Vertex {
        position: [-0.5, -1.0, 0.0, 1.0],
        texture_coordinates: [0.0, 1.0],
        texture_index: 2,
        padding: 0,
    },
    Vertex {
        position: [0.0, 0.0, 0.0, 1.0],
        texture_coordinates: [0.5, 0.0],
        texture_index: 2,
        padding: 0,
    },
    Vertex {
        position: [1.5, -1.0, 0.0, 1.0],
        texture_coordinates: [1.0, 1.0],
        texture_index: 3,
        padding: 0,
    },
    Vertex {
        position: [0.5, -1.0, 0.0, 1.0],
        texture_coordinates: [0.0, 1.0],
        texture_index: 3,
        padding: 0,
    },
    Vertex {
        position: [1.0, 0.0, 0.0, 1.0],
        texture_coordinates: [0.5, 0.0],
        texture_index: 3,
        padding: 0,
    },
];

const INDICES: [u32; 12] = [0, 1, 2, 0, 2, 3, 4, 5, 6, 7, 8, 9];

const SHADER_SOURCE: &str = "
struct Uniform {
    mvp: mat4x4<f32>,
};

struct AtlasRegion {
    min_u: f32,
    min_v: f32,
    max_u: f32,
    max_v: f32,
};

@group(0) @binding(0)
var<uniform> ubo: Uniform;

@group(1) @binding(0)
var texture_atlas: texture_2d<f32>;

@group(1) @binding(1)
var texture_sampler: sampler;

@group(1) @binding(2)
var<storage, read> atlas_regions: array<AtlasRegion>;

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) texture_coordinates: vec2<f32>,
    @location(2) texture_index: u32,
};
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) texture_coordinates: vec2<f32>,
    @location(1) @interpolate(flat) texture_index: u32,
};

@vertex
fn vertex_main(vert: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.texture_coordinates = vert.texture_coordinates;
    out.texture_index = vert.texture_index;
    out.position = ubo.mvp * vert.position;
    return out;
};

@fragment
fn fragment_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let region = atlas_regions[in.texture_index];
    let atlas_uv = vec2<f32>(
        mix(region.min_u, region.max_u, in.texture_coordinates.x),
        mix(region.min_v, region.max_v, in.texture_coordinates.y)
    );
    return textureSample(texture_atlas, texture_sampler, atlas_uv);
}
";
