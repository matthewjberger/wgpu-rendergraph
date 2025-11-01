pub struct BrightnessContrastConfig {
    pub enabled: bool,
    pub brightness: f32,
    pub contrast: f32,
}

impl Default for BrightnessContrastConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            brightness: 0.0,
            contrast: 1.0,
        }
    }
}

pub struct VignetteConfig {
    pub enabled: bool,
    pub strength: f32,
    pub radius: f32,
    pub color_tint: [f32; 3],
}

impl Default for VignetteConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strength: 1.5,
            radius: 0.3,
            color_tint: [0.0, 0.0, 0.0],
        }
    }
}

pub struct SharpenConfig {
    pub enabled: bool,
    pub strength: f32,
}

impl Default for SharpenConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            strength: 0.5,
        }
    }
}

#[derive(Default)]
pub struct GaussianBlurConfig {
    pub enabled: bool,
}

pub struct ConvolutionConfig {
    pub enabled: bool,
    pub kernel: [f32; 9],
}

impl Default for ConvolutionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            kernel: [0.0; 9],
        }
    }
}

#[derive(Default)]
pub struct GrayscaleConfig {
    pub enabled: bool,
}

#[derive(Default)]
pub struct ColorInvertConfig {
    pub enabled: bool,
}

#[derive(Default)]
pub struct EdgeDetectionConfig {
    pub enabled: bool,
}

#[derive(Default)]
pub struct ComputeGrayscaleConfig {
    pub enabled: bool,
}

pub struct EguiConfig {
    pub paint_jobs: Vec<egui::ClippedPrimitive>,
    pub screen_descriptor: egui_wgpu::ScreenDescriptor,
    pub renderer: Option<egui_wgpu::Renderer>,
}

impl Default for EguiConfig {
    fn default() -> Self {
        Self {
            paint_jobs: Vec::new(),
            screen_descriptor: egui_wgpu::ScreenDescriptor {
                size_in_pixels: [0, 0],
                pixels_per_point: 1.0,
            },
            renderer: None,
        }
    }
}

#[derive(Default)]
pub struct PassConfigs {
    pub brightness_contrast: BrightnessContrastConfig,
    pub vignette: VignetteConfig,
    pub sharpen: SharpenConfig,
    pub gaussian_blur: GaussianBlurConfig,
    pub convolution: ConvolutionConfig,
    pub grayscale: GrayscaleConfig,
    pub color_invert: ColorInvertConfig,
    pub edge_detection: EdgeDetectionConfig,
    pub compute_grayscale: ComputeGrayscaleConfig,
    pub egui: EguiConfig,
}
