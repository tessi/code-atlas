use std::{
    fmt, fs,
    path::Path,
    str::FromStr,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use fontdue::{Font, FontSettings};
use serde::{Deserialize, Serialize};
use tiny_skia::{
    BlendMode, Color, FillRule, LineCap, LineJoin, Paint, Path as SkiaPath, PathBuilder, Pixmap,
    PixmapPaint, Stroke, Transform,
};

use crate::{
    layout::{LayoutOptions, call_control_points, call_curve, sample_rect},
    model::{Atlas, NodeKind, Point, Rect},
};

pub(crate) const DENSITY_KNEE_MULTIPLIER: usize = 4;
pub(crate) const DENSE_EXPOSURE_EXPONENT: f64 = 1.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderBackend {
    Software,
    Wgpu,
}

impl FromStr for RenderBackend {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "software" | "cpu" => Ok(Self::Software),
            "wgpu" | "gpu" => Ok(Self::Wgpu),
            _ => bail!("unknown render backend {value:?}; expected software or wgpu"),
        }
    }
}

impl fmt::Display for RenderBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Software => "software",
            Self::Wgpu => "wgpu",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Theme {
    Architect,
    Night,
    Ink,
    SolarizedDark,
    SolarizedLight,
}

impl FromStr for Theme {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "architect" | "architect-pencil" | "pencil" => Ok(Self::Architect),
            "night" | "night-transit" => Ok(Self::Night),
            "ink" | "ink-transit" => Ok(Self::Ink),
            "solarized-dark" => Ok(Self::SolarizedDark),
            "solarized-light" => Ok(Self::SolarizedLight),
            _ => bail!(
                "unknown theme {value:?}; expected architect, night, ink, solarized-dark, or solarized-light"
            ),
        }
    }
}

impl fmt::Display for Theme {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Architect => "architect",
            Self::Night => "night",
            Self::Ink => "ink",
            Self::SolarizedDark => "solarized-dark",
            Self::SolarizedLight => "solarized-light",
        })
    }
}

#[derive(Debug, Clone)]
pub struct RenderOptions {
    pub backend: RenderBackend,
    pub theme: Theme,
    /// Optional HTTP(S) template used by interactive HTML endpoint links.
    /// It must contain `{revision}`, `{path}`, and `{line}` placeholders.
    pub source_url_template: Option<String>,
    pub call_opacity: u8,
    pub call_width: f32,
    pub boundary_width: f32,
    /// Reduce per-call pigment on very large graphs while preserving every
    /// independently composited stroke and its cumulative darkening.
    pub density_aware_exposure: bool,
    pub density_reference_calls: usize,
    /// Physical resolution used to convert raster pixels to PDF points.
    pub pdf_dpi: f64,
    /// Optional independent resolution for the raster call layer embedded in
    /// PDF output. The page geometry continues to use `pdf_dpi`.
    pub pdf_call_dpi: Option<f64>,
    /// Maximum core width and height of one PDF call-layer tile in pixels.
    pub pdf_call_tile_size: u32,
    /// Retain and reuse checksummed PDF call-layer tiles beside the output.
    pub pdf_resume: bool,
    /// Discard an existing PDF tile cache before starting a resumable render.
    pub pdf_restart: bool,
    /// Internal deterministic interruption hook used by resume tests.
    pub pdf_stop_after_tiles: Option<usize>,
    /// Maximum concurrent PDF tile compression jobs. Zero selects a bounded
    /// hardware-aware default.
    pub pdf_compression_workers: usize,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            backend: RenderBackend::Software,
            theme: Theme::Architect,
            source_url_template: None,
            call_opacity: 34,
            call_width: 1.10,
            boundary_width: 0.62,
            density_aware_exposure: true,
            density_reference_calls: 2_500,
            pdf_dpi: 144.0,
            pdf_call_dpi: None,
            pdf_call_tile_size: 2_048,
            pdf_resume: false,
            pdf_restart: false,
            pdf_stop_after_tiles: None,
            pdf_compression_workers: 0,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PdfTileTimingStats {
    pub planning_ms: u64,
    pub scene_prepare_ms: u64,
    pub spline_evaluation_gpu_ms: Option<f64>,
    pub rasterization_gpu_ms: Option<f64>,
    pub render_and_readback_ms: u64,
    pub rgba_split_ms: u64,
    pub compression_ms: u64,
    pub cache_write_ms: u64,
    pub pdf_assembly_ms: u64,
    /// Wall time of the overlapped render/split/compress/write pipeline.
    pub tile_pipeline_wall_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfTileCacheStats {
    pub enabled: bool,
    pub cache_key: String,
    pub manifest_path: String,
    pub tiles_rendered: usize,
    pub tiles_reused: usize,
    pub compression_workers: usize,
    pub max_in_flight_tiles: usize,
    pub compression_codec: String,
    pub compression_level: i32,
    pub timings: PdfTileTimingStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderStats {
    pub backend: String,
    pub gpu_adapter: Option<String>,
    pub width: u32,
    pub height: u32,
    pub call_layer_width: u32,
    pub call_layer_height: u32,
    pub call_layer_dpi: Option<f64>,
    pub call_layer_tiles: usize,
    pub call_layer_tile_size: Option<u32>,
    pub call_layer_overlap: Option<u32>,
    pub files_drawn: usize,
    pub file_labels_drawn: usize,
    pub directory_labels_drawn: usize,
    pub calls_drawn: usize,
    pub directory_boundaries_drawn: usize,
    pub output_bytes: u64,
    pub effective_call_opacity: u8,
    pub call_compositing: String,
    pub direction_source_color: String,
    pub direction_target_color: String,
    pub render_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pdf_tile_cache: Option<PdfTileCacheStats>,
}

pub fn render_png(
    atlas: &Atlas,
    output: &Path,
    layout_options: &LayoutOptions,
    render_options: &RenderOptions,
) -> Result<RenderStats> {
    match render_options.backend {
        RenderBackend::Software => {
            render_software_png(atlas, output, layout_options, render_options)
        }
        RenderBackend::Wgpu => {
            match crate::render_wgpu::render_wgpu_png(atlas, output, layout_options, render_options)
            {
                Ok(stats) => Ok(stats),
                Err(error) if gpu_is_unavailable(&error) => {
                    eprintln!(
                        "wgpu is unavailable ({error:#}); falling back to the software call renderer"
                    );
                    let mut stats =
                        render_software_png(atlas, output, layout_options, render_options)?;
                    stats.backend = "software (wgpu unavailable)".to_owned();
                    Ok(stats)
                }
                Err(error) => Err(error),
            }
        }
    }
}

pub fn render_output(
    atlas: &Atlas,
    output: &Path,
    layout_options: &LayoutOptions,
    render_options: &RenderOptions,
) -> Result<RenderStats> {
    match output
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "png" => render_png(atlas, output, layout_options, render_options),
        "svg" | "pdf" => {
            crate::export::render_vector_artifact(atlas, output, layout_options, render_options)
        }
        "html" | "htm" => {
            crate::html::render_interactive_html(atlas, output, layout_options, render_options)
        }
        extension => {
            bail!("unsupported output extension {extension:?}; expected .png, .svg, .pdf, or .html")
        }
    }
}

pub(crate) fn render_call_layer_png(
    atlas: &Atlas,
    output: &Path,
    layout_options: &LayoutOptions,
    render_options: &RenderOptions,
) -> Result<RenderStats> {
    match render_options.backend {
        RenderBackend::Software => {
            render_software_call_layer_png(atlas, output, layout_options, render_options)
        }
        RenderBackend::Wgpu => match crate::render_wgpu::render_wgpu_call_layer_png(
            atlas,
            output,
            layout_options,
            render_options,
        ) {
            Ok(stats) => Ok(stats),
            Err(error) if gpu_is_unavailable(&error) => {
                eprintln!(
                    "wgpu is unavailable ({error:#}); falling back to the software call renderer"
                );
                let mut stats =
                    render_software_call_layer_png(atlas, output, layout_options, render_options)?;
                stats.backend = "software (wgpu unavailable)".to_owned();
                Ok(stats)
            }
            Err(error) => Err(error),
        },
    }
}

pub(crate) fn gpu_is_unavailable(error: &anyhow::Error) -> bool {
    let message = format!("{error:#}");
    message.contains("no compatible GPU adapter")
        || message.contains("No suitable graphics adapter")
        || message.contains("does not support blendable RGBA16Float")
        || message.contains("does not support 4x multisampled RGBA16Float")
        || message.contains("exceeds the adapter's")
        || message.contains("GPU spline scene exceeds")
}

pub(crate) fn render_software_png(
    atlas: &Atlas,
    output: &Path,
    layout_options: &LayoutOptions,
    render_options: &RenderOptions,
) -> Result<RenderStats> {
    render_software_png_inner(atlas, output, layout_options, render_options, true, true)
}

pub(crate) fn render_software_base_png(
    atlas: &Atlas,
    output: &Path,
    layout_options: &LayoutOptions,
    render_options: &RenderOptions,
) -> Result<RenderStats> {
    render_software_png_inner(atlas, output, layout_options, render_options, false, false)
}

pub(crate) fn render_software_call_layer_png(
    atlas: &Atlas,
    output: &Path,
    layout_options: &LayoutOptions,
    render_options: &RenderOptions,
) -> Result<RenderStats> {
    let started = Instant::now();
    let palette = Palette::for_theme(render_options.theme);
    let opacity = effective_call_opacity(atlas.calls.len(), render_options);
    let call_indices: Vec<_> = (0..atlas.calls.len()).collect();
    let (pixmap, calls_drawn) = render_software_call_tile(
        atlas,
        layout_options,
        render_options,
        &call_indices,
        CallTileViewport {
            x: 0,
            y: 0,
            width: layout_options.width,
            height: layout_options.height,
        },
    )?;
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("cannot create output directory {}", parent.display()))?;
    }
    pixmap
        .save_png(output)
        .with_context(|| format!("cannot save {}", output.display()))?;
    Ok(RenderStats {
        backend: RenderBackend::Software.to_string(),
        gpu_adapter: None,
        width: layout_options.width,
        height: layout_options.height,
        call_layer_width: layout_options.width,
        call_layer_height: layout_options.height,
        call_layer_dpi: None,
        call_layer_tiles: 1,
        call_layer_tile_size: None,
        call_layer_overlap: None,
        files_drawn: atlas.files().count(),
        file_labels_drawn: 0,
        directory_labels_drawn: 0,
        calls_drawn,
        directory_boundaries_drawn: atlas
            .nodes
            .iter()
            .filter(|node| node.kind == NodeKind::Directory && node.id != 0)
            .count(),
        output_bytes: fs::metadata(output)?.len(),
        effective_call_opacity: opacity,
        call_compositing: optical_density_description(),
        direction_source_color: palette.direction_source.hex(),
        direction_target_color: palette.direction_target.hex(),
        render_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        pdf_tile_cache: None,
    })
}

/// Render one transparent, integer-aligned viewport of the global call layer.
/// Geometry remains in full-frame coordinates until the final translation, so
/// independently rendered overlapping tiles are pixel-equivalent at seams.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CallTileViewport {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug)]
pub(crate) struct CallTile {
    pub core_x: u32,
    pub core_y: u32,
    pub core_width: u32,
    pub core_height: u32,
    pub render_x: u32,
    pub render_y: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub call_indices: Vec<usize>,
}

pub(crate) fn plan_call_tiles(
    atlas: &Atlas,
    layout: &LayoutOptions,
    render: &RenderOptions,
) -> Result<(Vec<CallTile>, usize, u32)> {
    ensure!(
        (64..=8_192).contains(&render.pdf_call_tile_size),
        "--pdf-call-tile-size must be between 64 and 8192 pixels"
    );
    let tile_size = render.pdf_call_tile_size;
    let overlap = call_tile_overlap(layout, render);
    let columns = layout.width.div_ceil(tile_size);
    let rows = layout.height.div_ceil(tile_size);
    let mut tiles = Vec::with_capacity((columns as usize).saturating_mul(rows as usize));
    for row in 0..rows {
        for column in 0..columns {
            let core_x = column * tile_size;
            let core_y = row * tile_size;
            let core_width = tile_size.min(layout.width - core_x);
            let core_height = tile_size.min(layout.height - core_y);
            let render_x = core_x.saturating_sub(overlap);
            let render_y = core_y.saturating_sub(overlap);
            let render_x1 = (core_x + core_width)
                .saturating_add(overlap)
                .min(layout.width);
            let render_y1 = (core_y + core_height)
                .saturating_add(overlap)
                .min(layout.height);
            tiles.push(CallTile {
                core_x,
                core_y,
                core_width,
                core_height,
                render_x,
                render_y,
                render_width: render_x1 - render_x,
                render_height: render_y1 - render_y,
                call_indices: Vec::new(),
            });
        }
    }

    let drawing_scale = (layout.width.min(layout.height) as f64 / 1080.0).max(0.5);
    let stroke_margin = (f64::from(render.call_width) * drawing_scale * 1.28 * 0.5 + 2.0).ceil();
    let mut calls_drawn = 0;
    for (call_index, call) in atlas.calls.iter().enumerate() {
        let mut points = call_control_points(atlas, call);
        if points.len() < 2 {
            continue;
        }
        let first = points[0];
        let last = *points.last().expect("target control point");
        let denominator = points.len().saturating_sub(1).max(1) as f64;
        let bundle_strength = layout.bundle_strength.clamp(0.0, 1.0);
        for (index, point) in points.iter_mut().enumerate() {
            let amount = index as f64 / denominator;
            let direct_x = first.x + (last.x - first.x) * amount;
            let direct_y = first.y + (last.y - first.y) * amount;
            point.x = direct_x + (point.x - direct_x) * bundle_strength;
            point.y = direct_y + (point.y - direct_y) * bundle_strength;
        }
        calls_drawn += 1;
        let Some((x0, y0, x1, y1)) =
            curve_bounds(&points, stroke_margin, layout.width, layout.height)
        else {
            continue;
        };
        let first_column = x0 / tile_size;
        let last_column = x1.saturating_sub(1) / tile_size;
        let first_row = y0 / tile_size;
        let last_row = y1.saturating_sub(1) / tile_size;
        for row in first_row..=last_row {
            for column in first_column..=last_column {
                let index = (row * columns + column) as usize;
                tiles[index].call_indices.push(call_index);
            }
        }
    }
    tiles.retain(|tile| !tile.call_indices.is_empty());
    Ok((tiles, calls_drawn, overlap))
}

fn curve_bounds(
    points: &[Point],
    margin: f64,
    width: u32,
    height: u32,
) -> Option<(u32, u32, u32, u32)> {
    let min_x = points
        .iter()
        .map(|point| point.x)
        .fold(f64::INFINITY, f64::min)
        - margin;
    let min_y = points
        .iter()
        .map(|point| point.y)
        .fold(f64::INFINITY, f64::min)
        - margin;
    let max_x = points
        .iter()
        .map(|point| point.x)
        .fold(f64::NEG_INFINITY, f64::max)
        + margin;
    let max_y = points
        .iter()
        .map(|point| point.y)
        .fold(f64::NEG_INFINITY, f64::max)
        + margin;
    if max_x <= 0.0 || max_y <= 0.0 || min_x >= f64::from(width) || min_y >= f64::from(height) {
        return None;
    }
    let x0 = min_x.floor().clamp(0.0, f64::from(width)) as u32;
    let y0 = min_y.floor().clamp(0.0, f64::from(height)) as u32;
    let x1 = max_x.ceil().clamp(0.0, f64::from(width)) as u32;
    let y1 = max_y.ceil().clamp(0.0, f64::from(height)) as u32;
    (x1 > x0 && y1 > y0).then_some((x0, y0, x1, y1))
}

pub(crate) fn call_tile_overlap(layout: &LayoutOptions, render: &RenderOptions) -> u32 {
    let drawing_scale = (layout.width.min(layout.height) as f64 / 1080.0).max(0.5);
    (f64::from(render.call_width) * drawing_scale * 1.28 * 0.5 + 0.34 + 2.0)
        .ceil()
        .max(4.0) as u32
}

pub(crate) fn render_software_call_tile(
    atlas: &Atlas,
    layout_options: &LayoutOptions,
    render_options: &RenderOptions,
    call_indices: &[usize],
    viewport: CallTileViewport,
) -> Result<(Pixmap, usize)> {
    let palette = Palette::for_theme(render_options.theme);
    let drawing_scale = (layout_options.width.min(layout_options.height) as f32 / 1080.0).max(0.5);
    let opacity = effective_call_opacity(atlas.calls.len(), render_options);
    let mut density = OpticalDensityLayer::new(viewport.width, viewport.height)?;
    let mut ordered_indices = call_indices.to_vec();
    ordered_indices.sort_unstable_by_key(|index| {
        atlas
            .calls
            .get(*index)
            .map(|call| call.id)
            .unwrap_or(u64::MAX)
    });
    let mut calls_drawn = 0;
    for index in ordered_indices {
        let Some(call) = atlas.calls.get(index) else {
            continue;
        };
        let mut points = call_curve(atlas, call, layout_options);
        if points.len() < 2 {
            continue;
        }
        for point in &mut points {
            point.x -= f64::from(viewport.x);
            point.y -= f64::from(viewport.y);
        }
        density.deposit_pencil_call(
            &points,
            palette.direction_source,
            palette.direction_target,
            opacity,
            render_options.call_width * drawing_scale,
            call.id ^ 0xca11_517e,
        );
        calls_drawn += 1;
    }
    Ok((density.resolve()?, calls_drawn))
}

#[derive(Clone, Copy, Default)]
struct OpticalDensityPixel {
    red: f32,
    green: f32,
    blue: f32,
    density: f32,
}

struct OpticalDensityLayer {
    width: u32,
    height: u32,
    pixels: Vec<OpticalDensityPixel>,
    coverage: Vec<f32>,
    path_position: Vec<f32>,
    dirty: Vec<u32>,
}

impl OpticalDensityLayer {
    fn new(width: u32, height: u32) -> Result<Self> {
        let pixel_count = (width as usize)
            .checked_mul(height as usize)
            .context("optical-density call tile dimensions overflow")?;
        Ok(Self {
            width,
            height,
            pixels: vec![OpticalDensityPixel::default(); pixel_count],
            coverage: vec![0.0; pixel_count],
            path_position: vec![0.0; pixel_count],
            dirty: Vec::new(),
        })
    }

    fn deposit_pencil_call(
        &mut self,
        points: &[Point],
        source: Rgba,
        target: Rgba,
        opacity: u8,
        width: f32,
        seed: u64,
    ) {
        if opacity == 0 || points.len() < 2 {
            return;
        }
        let source = linear_rgb(source);
        let target = linear_rgb(target);
        for (pass, (alpha_scale, width_scale, amplitude)) in
            [(0.62, 0.82, 0.14), (0.38, 1.28, 0.34)]
                .into_iter()
                .enumerate()
        {
            let jittered = jitter_points(points, seed ^ pass as u64, amplitude);
            let alpha = (opacity as f32 * alpha_scale).round().max(1.0) / 255.0;
            let unit_density = -(1.0 - alpha.min(0.999_999)).ln();
            self.rasterize_path(&jittered, width * width_scale);
            self.commit_path(unit_density, source, target);
        }
    }

    fn rasterize_path(&mut self, points: &[Point], width: f32) {
        let denominator = points.len().saturating_sub(1).max(1) as f64;
        for (segment, pair) in points.windows(2).enumerate() {
            let start = pair[0];
            let end = pair[1];
            let dx = end.x - start.x;
            let dy = end.y - start.y;
            let length_squared = dx * dx + dy * dy;
            if length_squared <= f64::EPSILON {
                continue;
            }
            let radius = f64::from(width.max(0.05)) * 0.5;
            let x0 = (start.x.min(end.x) - radius - 0.5)
                .floor()
                .clamp(0.0, f64::from(self.width)) as u32;
            let y0 = (start.y.min(end.y) - radius - 0.5)
                .floor()
                .clamp(0.0, f64::from(self.height)) as u32;
            let x1 = (start.x.max(end.x) + radius + 0.5)
                .ceil()
                .clamp(0.0, f64::from(self.width)) as u32;
            let y1 = (start.y.max(end.y) + radius + 0.5)
                .ceil()
                .clamp(0.0, f64::from(self.height)) as u32;
            for y in y0..y1 {
                for x in x0..x1 {
                    let px = f64::from(x) + 0.5;
                    let py = f64::from(y) + 0.5;
                    let along = (((px - start.x) * dx + (py - start.y) * dy) / length_squared)
                        .clamp(0.0, 1.0);
                    let closest_x = start.x + dx * along;
                    let closest_y = start.y + dy * along;
                    let distance = (px - closest_x).hypot(py - closest_y);
                    let coverage = (radius + 0.5 - distance).clamp(0.0, 1.0) as f32;
                    if coverage == 0.0 {
                        continue;
                    }
                    let index = y as usize * self.width as usize + x as usize;
                    if coverage > self.coverage[index] {
                        if self.coverage[index] == 0.0 {
                            self.dirty.push(index as u32);
                        }
                        self.coverage[index] = coverage;
                        self.path_position[index] = ((segment as f64 + along) / denominator) as f32;
                    }
                }
            }
        }
    }

    fn commit_path(&mut self, unit_density: f32, source: [f32; 3], target: [f32; 3]) {
        for index in self.dirty.drain(..) {
            let index = index as usize;
            let weight = self.coverage[index] * unit_density;
            let t = self.path_position[index].clamp(0.0, 1.0);
            let color = [
                source[0] + (target[0] - source[0]) * t,
                source[1] + (target[1] - source[1]) * t,
                source[2] + (target[2] - source[2]) * t,
            ];
            let pixel = &mut self.pixels[index];
            pixel.red += color[0] * weight;
            pixel.green += color[1] * weight;
            pixel.blue += color[2] * weight;
            pixel.density += weight;
            self.coverage[index] = 0.0;
            self.path_position[index] = 0.0;
        }
    }

    fn resolve(self) -> Result<Pixmap> {
        let mut pixmap = Pixmap::new(self.width, self.height)
            .context("optical-density call tile dimensions are too large")?;
        for (output, density) in pixmap.data_mut().chunks_exact_mut(4).zip(self.pixels) {
            if density.density <= f32::EPSILON {
                continue;
            }
            let alpha = 1.0 - (-density.density).exp();
            let inverse_density = density.density.recip();
            let color = [
                linear_to_srgb((density.red * inverse_density).clamp(0.0, 1.0)),
                linear_to_srgb((density.green * inverse_density).clamp(0.0, 1.0)),
                linear_to_srgb((density.blue * inverse_density).clamp(0.0, 1.0)),
            ];
            let alpha_byte = (alpha * 255.0).round().clamp(0.0, 255.0) as u8;
            output[3] = alpha_byte;
            for (channel, value) in output[..3].iter_mut().zip(color) {
                *channel = (value * alpha * 255.0)
                    .round()
                    .clamp(0.0, f32::from(alpha_byte)) as u8;
            }
        }
        Ok(pixmap)
    }
}

fn linear_rgb(color: Rgba) -> [f32; 3] {
    [
        srgb_to_linear(color.r),
        srgb_to_linear(color.g),
        srgb_to_linear(color.b),
    ]
}

fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

pub(crate) fn optical_density_description() -> String {
    "order-independent optical density with exponential transmittance".to_owned()
}

fn render_software_png_inner(
    atlas: &Atlas,
    output: &Path,
    layout_options: &LayoutOptions,
    render_options: &RenderOptions,
    draw_calls: bool,
    draw_label_layer: bool,
) -> Result<RenderStats> {
    let render_started = Instant::now();
    let palette = Palette::for_theme(render_options.theme);
    let architectural = matches!(render_options.theme, Theme::Architect | Theme::Night);
    let drawing_scale = (layout_options.width.min(layout_options.height) as f32 / 1080.0).max(0.5);
    let mut pixmap = Pixmap::new(layout_options.width, layout_options.height)
        .context("output dimensions are too large for the software renderer")?;
    pixmap.fill(palette.background.color());
    if architectural {
        draw_paper_texture(
            &mut pixmap,
            layout_options.texture_seed,
            palette.paper_fiber,
        )?;
    }

    let root_points = sample_rect(atlas.nodes[0].rect, 3);
    let root_path = polygon_path(&root_points)?;
    fill_path(&mut pixmap, &root_path, palette.land, BlendMode::SourceOver);

    let mut files_drawn = 0;
    for node in atlas
        .nodes
        .iter()
        .filter(|node| node.kind == NodeKind::File)
    {
        let perimeter_samples = if architectural {
            if node.rect.width().min(node.rect.height()) > 30.0 {
                8
            } else {
                3
            }
        } else {
            2
        };
        let points = sample_rect(node.rect, perimeter_samples);
        let path = polygon_path(&points)?;
        fill_path(
            &mut pixmap,
            &path,
            palette.file_color(&node.language),
            BlendMode::SourceOver,
        );
        if architectural {
            pencil_outline(
                &mut pixmap,
                &points,
                palette.file_border,
                PencilStroke {
                    width: render_options.boundary_width * 0.62 * drawing_scale,
                    seed: node.id as u64 ^ layout_options.texture_seed,
                    blend_mode: boundary_blend_mode(render_options.theme),
                },
                0.24 * drawing_scale as f64,
                2,
            )?;
        }
        files_drawn += 1;
    }

    let mut directory_boundaries_drawn = 0;
    let mut directories: Vec<_> = atlas
        .nodes
        .iter()
        .filter(|node| node.kind == NodeKind::Directory && node.id != 0)
        .collect();
    directories.sort_by_key(|node| node.depth);
    for directory in directories {
        let samples = if architectural { 10 } else { 4 };
        let points = sample_rect(directory.rect, samples);
        let path = polygon_path(&points)?;
        let width =
            render_options.boundary_width * drawing_scale * (1.0 + 0.35 / directory.depth as f32);
        if architectural {
            pencil_outline(
                &mut pixmap,
                &points,
                palette.directory_border,
                PencilStroke {
                    width,
                    seed: directory.id as u64 ^ 0xd1ec_7000,
                    blend_mode: boundary_blend_mode(render_options.theme),
                },
                0.34 * drawing_scale as f64,
                3,
            )?;
        } else {
            stroke_path(
                &mut pixmap,
                &path,
                palette.directory_border,
                width,
                BlendMode::SourceOver,
            );
        }
        directory_boundaries_drawn += 1;
    }

    if architectural {
        pencil_outline(
            &mut pixmap,
            &root_points,
            palette.outer_border,
            PencilStroke {
                width: render_options.boundary_width * 1.8 * drawing_scale,
                seed: layout_options.texture_seed ^ 0xc0a5_7000,
                blend_mode: boundary_blend_mode(render_options.theme),
            },
            0.28 * drawing_scale as f64,
            3,
        )?;
    } else {
        stroke_path(
            &mut pixmap,
            &root_path,
            palette.outer_border,
            render_options.boundary_width * 2.2 * drawing_scale,
            BlendMode::SourceOver,
        );
    }
    let effective_call_opacity = effective_call_opacity(atlas.calls.len(), render_options);
    let calls_drawn = if draw_calls {
        // Calls deposit into one floating-point optical-density field. The
        // field resolves exactly once, making shared-route exposure independent
        // of call order and identical for every theme.
        let call_indices: Vec<_> = (0..atlas.calls.len()).collect();
        let (call_layer, calls_drawn) = render_software_call_tile(
            atlas,
            layout_options,
            render_options,
            &call_indices,
            CallTileViewport {
                x: 0,
                y: 0,
                width: layout_options.width,
                height: layout_options.height,
            },
        )?;
        pixmap.draw_pixmap(
            0,
            0,
            call_layer.as_ref(),
            &PixmapPaint::default(),
            Transform::identity(),
            None,
        );
        calls_drawn
    } else {
        0
    };

    // Typography is deliberately composited last. Dense call traffic may run
    // through a parcel, but it must never erase the map's hierarchy or file
    // names.
    let label_stats = if draw_label_layer {
        draw_labels(&mut pixmap, atlas, layout_options, palette)?
    } else {
        LabelStats::default()
    };

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("cannot create output directory {}", parent.display()))?;
    }
    pixmap
        .save_png(output)
        .with_context(|| format!("cannot save {}", output.display()))?;
    let output_bytes = fs::metadata(output)?.len();

    Ok(RenderStats {
        backend: RenderBackend::Software.to_string(),
        gpu_adapter: None,
        width: layout_options.width,
        height: layout_options.height,
        call_layer_width: layout_options.width,
        call_layer_height: layout_options.height,
        call_layer_dpi: None,
        call_layer_tiles: 1,
        call_layer_tile_size: None,
        call_layer_overlap: None,
        files_drawn,
        file_labels_drawn: label_stats.files,
        directory_labels_drawn: label_stats.directories,
        calls_drawn,
        directory_boundaries_drawn,
        output_bytes,
        effective_call_opacity,
        call_compositing: optical_density_description(),
        direction_source_color: palette.direction_source.hex(),
        direction_target_color: palette.direction_target.hex(),
        render_ms: render_started
            .elapsed()
            .as_millis()
            .min(u128::from(u64::MAX)) as u64,
        pdf_tile_cache: None,
    })
}

fn srgb_to_linear(value: u8) -> f32 {
    let value = value as f32 / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

pub fn effective_call_opacity(call_count: usize, options: &RenderOptions) -> u8 {
    let reference_calls = options.density_reference_calls.max(1);
    let full_strength_until = reference_calls.saturating_mul(DENSITY_KNEE_MULTIPLIER);
    if !options.density_aware_exposure
        || call_count <= full_strength_until
        || options.call_opacity == 0
    {
        return options.call_opacity;
    }
    // Keep ordinary maps at full strength. Past the density knee, fall off
    // steeply enough that very large maps still reveal their dominant bundles.
    let scale = (full_strength_until as f64 / call_count as f64).powf(DENSE_EXPOSURE_EXPONENT);
    (f64::from(options.call_opacity) * scale)
        .round()
        .clamp(1.0, f64::from(options.call_opacity)) as u8
}

fn draw_paper_texture(pixmap: &mut Pixmap, seed: u64, color: Rgba) -> Result<()> {
    let width = pixmap.width() as f64;
    let height = pixmap.height() as f64;
    let drawing_scale = (width.min(height) / 1080.0).max(0.5) as f32;
    let fibers = ((width * height / 2600.0) as usize).clamp(280, 1400);
    let mut state = seed ^ 0x5eed_f1be_7000;
    for index in 0..fibers {
        state = splitmix64(state ^ index as u64);
        let x = unit_f64(state) * width;
        state = splitmix64(state);
        let y = unit_f64(state) * height;
        state = splitmix64(state);
        let length = 8.0 + unit_f64(state) * width * 0.075;
        state = splitmix64(state);
        let rise = (unit_f64(state) - 0.5) * 1.8;
        let path = open_path(&[
            Point { x, y },
            Point {
                x: (x + length).min(width),
                y: (y + rise).clamp(0.0, height),
            },
        ])?;
        stroke_path(
            pixmap,
            &path,
            color,
            0.24 * drawing_scale,
            BlendMode::Multiply,
        );
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct PencilStroke {
    width: f32,
    seed: u64,
    blend_mode: BlendMode,
}

fn pencil_outline(
    pixmap: &mut Pixmap,
    points: &[Point],
    color: Rgba,
    stroke: PencilStroke,
    amplitude: f64,
    passes: usize,
) -> Result<()> {
    let passes = passes.max(1);
    for pass in 0..passes {
        let jittered = jitter_points(
            points,
            stroke.seed ^ pass as u64,
            amplitude * (pass + 1) as f64,
        );
        let path = polygon_path(&jittered)?;
        let mut pass_color = color;
        pass_color.a = ((color.a as usize / passes).max(1) as u8).saturating_add(5);
        stroke_path(
            pixmap,
            &path,
            pass_color,
            stroke.width * (0.78 + pass as f32 * 0.13),
            stroke.blend_mode,
        );
    }
    Ok(())
}

fn boundary_blend_mode(theme: Theme) -> BlendMode {
    if theme == Theme::Night {
        BlendMode::Plus
    } else {
        BlendMode::Multiply
    }
}

pub(crate) fn jitter_points(points: &[Point], seed: u64, amplitude: f64) -> Vec<Point> {
    let mut state = seed;
    points
        .iter()
        .map(|point| {
            state = splitmix64(state);
            let dx = (unit_f64(state) - 0.5) * 2.0 * amplitude;
            state = splitmix64(state);
            let dy = (unit_f64(state) - 0.5) * 2.0 * amplitude;
            Point {
                x: point.x + dx,
                y: point.y + dy,
            }
        })
        .collect()
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct LabelStats {
    pub files: usize,
    pub directories: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct LabelCommand {
    pub text: String,
    pub size: f32,
    pub color: Rgba,
    pub placement: LabelPlacement,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum LabelPlacement {
    Horizontal { x: f32, baseline: f32 },
    Vertical { x: f32, y: f32 },
    Signature { x: f32, baseline: f32 },
}

pub(crate) fn draw_labels(
    pixmap: &mut Pixmap,
    atlas: &Atlas,
    options: &LayoutOptions,
    palette: Palette,
) -> Result<LabelStats> {
    let Some(font) = load_label_font() else {
        return Ok(LabelStats::default());
    };
    let (commands, stats) = build_label_commands(&font, atlas, options, palette);
    for command in commands {
        match command.placement {
            LabelPlacement::Horizontal { x, baseline } => draw_text(
                pixmap,
                &font,
                &command.text,
                x,
                baseline,
                command.size,
                command.color,
            ),
            LabelPlacement::Vertical { x, y } => draw_text_rotated_clockwise(
                pixmap,
                &font,
                &command.text,
                x,
                y,
                command.size,
                command.color,
            ),
            LabelPlacement::Signature { x, baseline } => {
                draw_hollow_heart(
                    pixmap,
                    x,
                    baseline - command.size * 0.88,
                    command.size * 0.92,
                    command.color,
                );
                draw_text(
                    pixmap,
                    &font,
                    &command.text,
                    x + command.size * 1.18,
                    baseline,
                    command.size,
                    command.color,
                );
            }
        }
    }
    Ok(stats)
}

pub(crate) fn collect_label_commands(
    atlas: &Atlas,
    options: &LayoutOptions,
    palette: Palette,
) -> (Vec<LabelCommand>, LabelStats) {
    let Some(font) = load_label_font() else {
        return (Vec::new(), LabelStats::default());
    };
    build_label_commands(&font, atlas, options, palette)
}

fn build_label_commands(
    font: &Font,
    atlas: &Atlas,
    options: &LayoutOptions,
    palette: Palette,
) -> (Vec<LabelCommand>, LabelStats) {
    let scale = (options.width.min(options.height) as f32 / 1080.0).max(0.55);
    let root = &atlas.nodes[0];
    let title_size = 17.0 * scale;
    let title = root.name.to_uppercase();
    let title_baseline = (root.rect.y0 as f32 - 13.0 * scale).max(title_size + 2.0);
    let mut commands = vec![LabelCommand {
        text: title,
        size: title_size,
        color: palette.label,
        placement: LabelPlacement::Horizontal {
            x: root.rect.x0 as f32,
            baseline: title_baseline,
        },
    }];

    let meta = format!(
        "{} FILES / {} CALLS",
        atlas.files().count(),
        atlas.calls.len()
    );
    let meta_size = 9.5 * scale;
    let meta_width = measure_text(font, &meta, meta_size);
    commands.push(LabelCommand {
        text: meta,
        size: meta_size,
        color: palette.label,
        placement: LabelPlacement::Horizontal {
            x: (root.rect.x1 as f32 - meta_width).max(root.rect.x0 as f32),
            baseline: title_baseline,
        },
    });

    let footer_size = 8.5 * scale;
    let footer_baseline =
        (root.rect.y1 as f32 + 17.0 * scale).min(options.height as f32 - 5.0 * scale);
    let mut footer_color = palette.label;
    footer_color.a = footer_color.a.saturating_mul(3) / 4;
    let revision: String = atlas.revision.chars().take(12).collect();
    let provenance = format!(
        "GIT {} / {}",
        revision.to_uppercase(),
        current_utc_timestamp()
    );
    commands.push(LabelCommand {
        text: provenance,
        size: footer_size,
        color: footer_color,
        placement: LabelPlacement::Horizontal {
            x: root.rect.x0 as f32,
            baseline: footer_baseline,
        },
    });

    let signature = "tessi";
    let signature_width = footer_size * 1.18 + measure_text(font, signature, footer_size);
    commands.push(LabelCommand {
        text: signature.to_owned(),
        size: footer_size,
        color: footer_color,
        placement: LabelPlacement::Signature {
            x: (root.rect.x1 as f32 - signature_width).max(root.rect.x0 as f32),
            baseline: footer_baseline,
        },
    });

    let mut occupied = Vec::new();
    let mut stats = LabelStats::default();
    let mut directories: Vec<_> = atlas
        .nodes
        .iter()
        .filter(|node| {
            node.kind == NodeKind::Directory
                && node.id != 0
                && node.depth <= 2
                && node.rect.width() > 64.0 * scale as f64
                && node.rect.height() > 25.0 * scale as f64
        })
        .collect();
    directories.sort_by_key(|node| node.depth);

    for node in directories {
        let size = if node.depth == 1 { 11.0 } else { 8.5 } * scale;
        let max_width = (node.rect.width() as f32 - 10.0 * scale).max(1.0);
        let text = node.name.to_uppercase();
        let fitted = fit_font_size(font, &text, size, max_width, 6.0 * scale);
        let x = node.rect.x0 as f32 + 5.0 * scale;
        let preferred_baseline = node.rect.y0 as f32 + fitted + 5.0 * scale;
        let Some((baseline, bounds)) = place_label(
            node.rect,
            x,
            preferred_baseline,
            fitted,
            measure_text(font, &text, fitted),
            2.5 * scale,
            &occupied,
        ) else {
            continue;
        };
        commands.push(LabelCommand {
            text,
            size: fitted,
            color: palette.label,
            placement: LabelPlacement::Horizontal { x, baseline },
        });
        occupied.push(bounds);
        stats.directories += 1;
    }

    // File parcels never overlap each other, so labels only need to avoid the
    // already-placed hierarchy labels. Fit full names whenever possible, then
    // shrink and finally abbreviate at a legible micro-label size. This keeps
    // small parcels identifiable when a high-resolution poster is zoomed.
    for node in atlas
        .nodes
        .iter()
        .filter(|node| node.kind == NodeKind::File)
    {
        let Some(label) = fit_file_label(font, &node.name, node.rect, scale, &occupied) else {
            continue;
        };
        let mut color = palette.label;
        color.a = color.a.saturating_mul(3) / 4;
        commands.push(LabelCommand {
            text: label.text,
            size: label.size,
            color,
            placement: label.placement,
        });
        stats.files += 1;
    }
    (commands, stats)
}

#[derive(Debug, Clone)]
struct FittedFileLabel {
    text: String,
    size: f32,
    placement: LabelPlacement,
}

fn fit_file_label(
    font: &Font,
    text: &str,
    container: Rect,
    scale: f32,
    occupied: &[LabelBounds],
) -> Option<FittedFileLabel> {
    let padding = (1.8 * scale)
        .min(container.width() as f32 * 0.16)
        .min(container.height() as f32 * 0.16)
        .max(0.6);
    let fitted = if container.height() > container.width() * 1.8 {
        fit_vertical_file_label(font, text, container, scale, padding, occupied)
            .or_else(|| fit_horizontal_file_label(font, text, container, scale, padding, occupied))
    } else {
        fit_horizontal_file_label(font, text, container, scale, padding, occupied)
            .or_else(|| fit_vertical_file_label(font, text, container, scale, padding, occupied))
    };
    fitted.or_else(|| fit_compact_file_marker(font, text, container, scale, occupied))
}

fn fit_compact_file_marker(
    font: &Font,
    text: &str,
    container: Rect,
    scale: f32,
    occupied: &[LabelBounds],
) -> Option<FittedFileLabel> {
    let marker = text.chars().next()?.to_string();
    let size = 1.5 * scale;
    let width = measure_text(font, &marker, size);
    let padding = 0.3_f32.min(container.width() as f32 * 0.08);
    let x_positions = [
        container.x0 as f32 + padding,
        (container.x0 + container.x1) as f32 / 2.0 - width / 2.0,
    ];
    let baselines = [
        container.y1 as f32 - padding - size * 0.22,
        container.y0 as f32 + padding + size,
        (container.y0 + container.y1) as f32 / 2.0 + size * 0.39,
    ];
    for x in x_positions {
        for baseline in baselines {
            let bounds = label_bounds(x, baseline, width, size);
            if label_fits(container, bounds)
                && !occupied.iter().any(|other| bounds.overlaps(*other, 0.0))
            {
                return Some(FittedFileLabel {
                    text: marker,
                    size,
                    placement: LabelPlacement::Horizontal { x, baseline },
                });
            }
        }
    }
    None
}

fn fit_horizontal_file_label(
    font: &Font,
    text: &str,
    container: Rect,
    scale: f32,
    padding: f32,
    occupied: &[LabelBounds],
) -> Option<FittedFileLabel> {
    let available_width = container.width() as f32 - 2.0 * padding;
    let available_height = container.height() as f32 - 2.0 * padding;
    let preferred = (7.5 * scale).min(available_height / 1.22);
    let minimum = 1.5 * scale;
    if available_width <= 0.0 || preferred < minimum {
        return None;
    }

    let preferred_width = measure_text(font, text, preferred).max(1.0);
    let required_size = preferred * available_width / preferred_width;
    let size = required_size.clamp(minimum, preferred);
    let fitted_text = truncate_text_to_width(font, text, size, available_width)?;
    let width = measure_text(font, &fitted_text, size);
    let x = container.x0 as f32 + padding;
    let bottom = container.y1 as f32 - padding - size * 0.22;
    let top = container.y0 as f32 + padding + size;
    let middle = (container.y0 + container.y1) as f32 / 2.0 + size * 0.39;

    for baseline in [bottom, top, middle] {
        let bounds = label_bounds(x, baseline, width, size);
        if label_fits(container, bounds)
            && !occupied
                .iter()
                .any(|other| bounds.overlaps(*other, 0.7 * scale))
        {
            return Some(FittedFileLabel {
                text: fitted_text,
                size,
                placement: LabelPlacement::Horizontal { x, baseline },
            });
        }
    }
    None
}

fn fit_vertical_file_label(
    font: &Font,
    text: &str,
    container: Rect,
    scale: f32,
    padding: f32,
    occupied: &[LabelBounds],
) -> Option<FittedFileLabel> {
    let available_cross = container.width() as f32 - 2.0 * padding;
    let available_length = container.height() as f32 - 2.0 * padding;
    let preferred = (7.5 * scale).min(available_cross / 1.22);
    let minimum = 1.5 * scale;
    if available_length <= 0.0 || preferred < minimum {
        return None;
    }

    let preferred_width = measure_text(font, text, preferred).max(1.0);
    let required_size = preferred * available_length / preferred_width;
    let size = required_size.clamp(minimum, preferred);
    let fitted_text = truncate_text_to_width(font, text, size, available_length)?;
    let text_length = measure_text(font, &fitted_text, size);
    let label_width = size * 1.22;
    let left = container.x0 as f32 + padding;
    let right = container.x1 as f32 - padding - label_width;
    let center_x = (container.x0 + container.x1) as f32 / 2.0 - label_width / 2.0;
    let top = container.y0 as f32 + padding;
    let bottom = container.y1 as f32 - padding - text_length;
    let center_y = (container.y0 + container.y1) as f32 / 2.0 - text_length / 2.0;

    for (x, y) in [
        (left, bottom),
        (right, bottom),
        (left, top),
        (right, top),
        (center_x, center_y),
    ] {
        let bounds = LabelBounds {
            x0: x,
            y0: y,
            x1: x + label_width,
            y1: y + text_length,
        };
        if label_fits(container, bounds)
            && !occupied
                .iter()
                .any(|other| bounds.overlaps(*other, 0.7 * scale))
        {
            return Some(FittedFileLabel {
                text: fitted_text,
                size,
                placement: LabelPlacement::Vertical { x, y },
            });
        }
    }
    None
}

fn truncate_text_to_width(font: &Font, text: &str, size: f32, max_width: f32) -> Option<String> {
    if measure_text(font, text, size) <= max_width {
        return Some(text.to_owned());
    }
    let ellipsis = "...";
    let ellipsis_width = measure_text(font, ellipsis, size);
    let mut output = String::new();
    for character in text.chars() {
        let candidate_width = measure_text(font, &output, size)
            + font.metrics(character, size).advance_width
            + ellipsis_width;
        if candidate_width > max_width {
            break;
        }
        output.push(character);
    }
    if output.is_empty() {
        return None;
    }
    output.push_str(ellipsis);
    Some(output)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LabelBounds {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

impl LabelBounds {
    fn overlaps(self, other: Self, gap: f32) -> bool {
        self.x0 < other.x1 + gap
            && self.x1 > other.x0 - gap
            && self.y0 < other.y1 + gap
            && self.y1 > other.y0 - gap
    }
}

fn label_bounds(x: f32, baseline: f32, width: f32, size: f32) -> LabelBounds {
    LabelBounds {
        x0: x,
        y0: baseline - size,
        x1: x + width,
        y1: baseline + size * 0.22,
    }
}

fn label_fits(container: Rect, bounds: LabelBounds) -> bool {
    bounds.x0 >= container.x0 as f32
        && bounds.y0 >= container.y0 as f32
        && bounds.x1 <= container.x1 as f32
        && bounds.y1 <= container.y1 as f32
}

fn place_label(
    container: Rect,
    x: f32,
    preferred_baseline: f32,
    size: f32,
    width: f32,
    gap: f32,
    occupied: &[LabelBounds],
) -> Option<(f32, LabelBounds)> {
    let mut baseline = preferred_baseline;
    for _ in 0..=occupied.len() {
        let bounds = label_bounds(x, baseline, width, size);
        if !label_fits(container, bounds) {
            return None;
        }
        let colliding_bottom = occupied
            .iter()
            .filter(|other| bounds.overlaps(**other, gap))
            .map(|other| other.y1)
            .reduce(f32::max);
        let Some(bottom) = colliding_bottom else {
            return Some((baseline, bounds));
        };
        baseline = bottom + gap + size;
    }
    None
}

fn load_label_font() -> Option<Font> {
    const FONT_PATHS: &[&str] = &[
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    ];
    FONT_PATHS.iter().find_map(|path| {
        let bytes = fs::read(path).ok()?;
        Font::from_bytes(bytes, FontSettings::default()).ok()
    })
}

fn fit_font_size(font: &Font, text: &str, preferred: f32, max_width: f32, minimum: f32) -> f32 {
    let width = measure_text(font, text, preferred).max(1.0);
    (preferred * (max_width / width).min(1.0)).max(minimum.min(preferred))
}

fn measure_text(font: &Font, text: &str, size: f32) -> f32 {
    text.chars()
        .map(|character| font.metrics(character, size).advance_width)
        .sum()
}

fn draw_text(
    pixmap: &mut Pixmap,
    font: &Font,
    text: &str,
    x: f32,
    baseline: f32,
    size: f32,
    color: Rgba,
) {
    let width = pixmap.width() as i32;
    let height = pixmap.height() as i32;
    let mut pen_x = x;
    let data = pixmap.data_mut();
    for character in text.chars() {
        let (metrics, bitmap) = font.rasterize(character, size);
        let glyph_x = pen_x.round() as i32 + metrics.xmin;
        let glyph_y = baseline.round() as i32 - metrics.height as i32 - metrics.ymin;
        for row in 0..metrics.height {
            for column in 0..metrics.width {
                let destination_x = glyph_x + column as i32;
                let destination_y = glyph_y + row as i32;
                if destination_x < 0
                    || destination_y < 0
                    || destination_x >= width
                    || destination_y >= height
                {
                    continue;
                }
                let coverage = bitmap[row * metrics.width + column] as f32 / 255.0;
                let alpha = coverage * (color.a as f32 / 255.0);
                if alpha <= 0.0 {
                    continue;
                }
                let offset = ((destination_y * width + destination_x) * 4) as usize;
                for (channel, source) in [color.r, color.g, color.b].into_iter().enumerate() {
                    let destination = data[offset + channel] as f32;
                    data[offset + channel] =
                        (source as f32 * alpha + destination * (1.0 - alpha)).round() as u8;
                }
            }
        }
        pen_x += metrics.advance_width;
    }
}

fn draw_text_rotated_clockwise(
    pixmap: &mut Pixmap,
    font: &Font,
    text: &str,
    x: f32,
    y: f32,
    size: f32,
    color: Rgba,
) {
    let canvas_width = pixmap.width() as i32;
    let canvas_height = pixmap.height() as i32;
    let normal_height = (size * 1.22).ceil().max(1.0) as i32;
    let origin_x = x.round() as i32;
    let origin_y = y.round() as i32;
    let mut pen_x = 0.0_f32;
    let data = pixmap.data_mut();

    for character in text.chars() {
        let (metrics, bitmap) = font.rasterize(character, size);
        let glyph_x = pen_x.round() as i32 + metrics.xmin;
        let glyph_y = size.round() as i32 - metrics.height as i32 - metrics.ymin;
        for row in 0..metrics.height {
            for column in 0..metrics.width {
                let normal_x = glyph_x + column as i32;
                let normal_y = glyph_y + row as i32;
                if normal_y < 0 || normal_y >= normal_height {
                    continue;
                }
                let destination_x = origin_x + normal_height - 1 - normal_y;
                let destination_y = origin_y + normal_x;
                if destination_x < 0
                    || destination_y < 0
                    || destination_x >= canvas_width
                    || destination_y >= canvas_height
                {
                    continue;
                }
                let coverage = bitmap[row * metrics.width + column] as f32 / 255.0;
                let alpha = coverage * (color.a as f32 / 255.0);
                if alpha <= 0.0 {
                    continue;
                }
                let offset = ((destination_y * canvas_width + destination_x) * 4) as usize;
                for (channel, source) in [color.r, color.g, color.b].into_iter().enumerate() {
                    let destination = data[offset + channel] as f32;
                    data[offset + channel] =
                        (source as f32 * alpha + destination * (1.0 - alpha)).round() as u8;
                }
            }
        }
        pen_x += metrics.advance_width;
    }
}

fn draw_hollow_heart(pixmap: &mut Pixmap, x: f32, y: f32, size: f32, color: Rgba) {
    let mut builder = PathBuilder::new();
    builder.move_to(x + size * 0.5, y + size * 0.92);
    builder.cubic_to(
        x + size * 0.08,
        y + size * 0.64,
        x,
        y + size * 0.34,
        x + size * 0.21,
        y + size * 0.19,
    );
    builder.cubic_to(
        x + size * 0.36,
        y + size * 0.08,
        x + size * 0.5,
        y + size * 0.19,
        x + size * 0.5,
        y + size * 0.32,
    );
    builder.cubic_to(
        x + size * 0.5,
        y + size * 0.19,
        x + size * 0.64,
        y + size * 0.08,
        x + size * 0.79,
        y + size * 0.19,
    );
    builder.cubic_to(
        x + size,
        y + size * 0.34,
        x + size * 0.92,
        y + size * 0.64,
        x + size * 0.5,
        y + size * 0.92,
    );
    if let Some(path) = builder.finish() {
        stroke_path(
            pixmap,
            &path,
            color,
            (size * 0.075).max(0.55),
            BlendMode::SourceOver,
        );
    }
}

fn current_utc_timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .min(i64::MAX as u64) as i64;
    format_utc_timestamp(seconds)
}

fn format_utc_timestamp(seconds: i64) -> String {
    let days = seconds.div_euclid(86_400);
    let seconds_of_day = seconds.rem_euclid(86_400);
    let hour = seconds_of_day / 3_600;
    let minute = seconds_of_day % 3_600 / 60;
    let second = seconds_of_day % 60;

    // Convert days since the Unix epoch to a proleptic Gregorian date.
    let shifted = days + 719_468;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    } / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_prime + 2) / 5 + 1;
    let month = month_prime + if month_prime < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);

    format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02} UTC")
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

fn unit_f64(value: u64) -> f64 {
    value as f64 / u64::MAX as f64
}

fn polygon_path(points: &[Point]) -> Result<SkiaPath> {
    let mut builder = PathBuilder::new();
    let Some(first) = points.first() else {
        bail!("cannot create an empty polygon");
    };
    builder.move_to(first.x as f32, first.y as f32);
    for point in &points[1..] {
        builder.line_to(point.x as f32, point.y as f32);
    }
    builder.close();
    builder.finish().context("invalid polygon geometry")
}

fn open_path(points: &[Point]) -> Result<SkiaPath> {
    let mut builder = PathBuilder::new();
    let Some(first) = points.first() else {
        bail!("cannot create an empty path");
    };
    builder.move_to(first.x as f32, first.y as f32);
    for point in &points[1..] {
        builder.line_to(point.x as f32, point.y as f32);
    }
    builder.finish().context("invalid path geometry")
}

fn fill_path(pixmap: &mut Pixmap, path: &SkiaPath, color: Rgba, blend_mode: BlendMode) {
    let mut paint = Paint {
        anti_alias: true,
        blend_mode,
        ..Paint::default()
    };
    paint.set_color_rgba8(color.r, color.g, color.b, color.a);
    pixmap.fill_path(path, &paint, FillRule::Winding, Transform::identity(), None);
}

fn stroke_path(
    pixmap: &mut Pixmap,
    path: &SkiaPath,
    color: Rgba,
    width: f32,
    blend_mode: BlendMode,
) {
    let mut paint = Paint {
        anti_alias: true,
        blend_mode,
        ..Paint::default()
    };
    paint.set_color_rgba8(color.r, color.g, color.b, color.a);
    let stroke = Stroke {
        width,
        line_cap: LineCap::Round,
        line_join: LineJoin::Round,
        ..Stroke::default()
    };
    pixmap.stroke_path(path, &paint, &stroke, Transform::identity(), None);
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Rgba {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Rgba {
    pub(crate) const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    pub(crate) const fn with_alpha(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    fn color(self) -> Color {
        Color::from_rgba8(self.r, self.g, self.b, self.a)
    }

    pub(crate) fn hex(self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Palette {
    pub background: Rgba,
    pub land: Rgba,
    pub paper_fiber: Rgba,
    pub outer_border: Rgba,
    pub directory_border: Rgba,
    pub file_border: Rgba,
    pub label: Rgba,
    pub direction_source: Rgba,
    pub direction_target: Rgba,
    pub elixir: Rgba,
    pub rust: Rgba,
    pub tests: Rgba,
    pub docs: Rgba,
    pub config: Rgba,
    pub other: Rgba,
}

impl Palette {
    pub(crate) fn for_theme(theme: Theme) -> Self {
        match theme {
            Theme::Architect => Self {
                background: Rgba::rgb(241, 238, 229),
                land: Rgba::rgb(246, 243, 235),
                paper_fiber: Rgba::with_alpha(83, 77, 65, 8),
                outer_border: Rgba::with_alpha(48, 48, 45, 150),
                directory_border: Rgba::with_alpha(54, 55, 53, 105),
                file_border: Rgba::with_alpha(68, 68, 64, 48),
                label: Rgba::with_alpha(48, 48, 45, 150),
                direction_source: Rgba::rgb(172, 52, 68),
                direction_target: Rgba::rgb(34, 92, 150),
                elixir: Rgba::rgb(234, 231, 237),
                rust: Rgba::rgb(238, 231, 222),
                tests: Rgba::rgb(230, 235, 230),
                docs: Rgba::rgb(238, 236, 229),
                config: Rgba::rgb(234, 233, 228),
                other: Rgba::rgb(241, 238, 231),
            },
            Theme::Night => Self {
                background: Rgba::rgb(6, 10, 17),
                land: Rgba::rgb(13, 20, 30),
                paper_fiber: Rgba::with_alpha(155, 175, 202, 6),
                outer_border: Rgba::with_alpha(160, 180, 205, 155),
                directory_border: Rgba::with_alpha(130, 150, 175, 70),
                file_border: Rgba::with_alpha(118, 140, 163, 32),
                label: Rgba::with_alpha(205, 218, 235, 215),
                direction_source: Rgba::rgb(255, 70, 142),
                direction_target: Rgba::rgb(55, 210, 255),
                elixir: Rgba::rgb(22, 30, 42),
                rust: Rgba::rgb(24, 32, 43),
                tests: Rgba::rgb(21, 31, 41),
                docs: Rgba::rgb(20, 28, 39),
                config: Rgba::rgb(25, 32, 42),
                other: Rgba::rgb(18, 26, 37),
            },
            Theme::Ink => Self {
                background: Rgba::rgb(245, 242, 232),
                land: Rgba::rgb(235, 231, 218),
                paper_fiber: Rgba::with_alpha(70, 62, 49, 0),
                outer_border: Rgba::with_alpha(29, 43, 67, 170),
                directory_border: Rgba::with_alpha(40, 52, 73, 70),
                file_border: Rgba::with_alpha(40, 52, 73, 30),
                label: Rgba::with_alpha(29, 43, 67, 150),
                direction_source: Rgba::rgb(190, 35, 70),
                direction_target: Rgba::rgb(30, 70, 175),
                elixir: Rgba::rgb(225, 218, 235),
                rust: Rgba::rgb(235, 219, 211),
                tests: Rgba::rgb(214, 231, 227),
                docs: Rgba::rgb(226, 226, 218),
                config: Rgba::rgb(220, 219, 215),
                other: Rgba::rgb(231, 228, 218),
            },
            Theme::SolarizedDark => Self {
                background: Rgba::rgb(0, 43, 54),
                land: Rgba::rgb(7, 54, 66),
                paper_fiber: Rgba::with_alpha(255, 255, 255, 0),
                outer_border: Rgba::with_alpha(147, 161, 161, 170),
                directory_border: Rgba::with_alpha(88, 110, 117, 90),
                file_border: Rgba::with_alpha(88, 110, 117, 35),
                label: Rgba::with_alpha(147, 161, 161, 170),
                direction_source: Rgba::rgb(220, 50, 47),
                direction_target: Rgba::rgb(38, 139, 210),
                elixir: Rgba::rgb(32, 65, 82),
                rust: Rgba::rgb(73, 55, 44),
                tests: Rgba::rgb(20, 73, 70),
                docs: Rgba::rgb(28, 60, 69),
                config: Rgba::rgb(48, 60, 64),
                other: Rgba::rgb(18, 55, 64),
            },
            Theme::SolarizedLight => Self {
                background: Rgba::rgb(253, 246, 227),
                land: Rgba::rgb(238, 232, 213),
                paper_fiber: Rgba::with_alpha(70, 62, 49, 0),
                outer_border: Rgba::with_alpha(88, 110, 117, 170),
                directory_border: Rgba::with_alpha(101, 123, 131, 80),
                file_border: Rgba::with_alpha(101, 123, 131, 35),
                label: Rgba::with_alpha(88, 110, 117, 160),
                direction_source: Rgba::rgb(220, 50, 47),
                direction_target: Rgba::rgb(38, 139, 210),
                elixir: Rgba::rgb(231, 222, 238),
                rust: Rgba::rgb(240, 219, 201),
                tests: Rgba::rgb(218, 235, 224),
                docs: Rgba::rgb(235, 230, 213),
                config: Rgba::rgb(226, 224, 211),
                other: Rgba::rgb(239, 233, 216),
            },
        }
    }

    pub(crate) fn file_color(self, language: &str) -> Rgba {
        match language {
            "elixir" => self.elixir,
            "erlang" => self.elixir,
            "rust" => self.rust,
            "markdown" => self.docs,
            "config" | "data" => self.config,
            language if language.contains("test") => self.tests,
            _ => self.other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utc_footer_timestamp_uses_stable_iso_style() {
        assert_eq!(format_utc_timestamp(0), "1970-01-01 00:00:00 UTC");
        assert_eq!(format_utc_timestamp(86_400), "1970-01-02 00:00:00 UTC");
    }

    #[test]
    fn directional_palette_has_one_unambiguous_source_to_target_ramp() {
        let palette = Palette::for_theme(Theme::Night);
        let source = linear_rgb(palette.direction_source);
        let target = linear_rgb(palette.direction_target);
        assert_ne!(source, target);
        assert!(source[0] > target[0], "source is the pink end of the ramp");
        assert!(target[2] > source[2], "target is the cyan end of the ramp");
    }

    #[test]
    fn night_treemap_uses_one_subtle_blueprint_family() {
        let palette = Palette::for_theme(Theme::Night);
        let fills = [
            palette.elixir,
            palette.rust,
            palette.tests,
            palette.docs,
            palette.config,
            palette.other,
        ];
        assert!(
            fills
                .iter()
                .all(|color| color.b >= color.g && color.g >= color.r)
        );
        for left in fills {
            for right in fills {
                assert!(left.r.abs_diff(right.r) <= 7);
                assert!(left.g.abs_diff(right.g) <= 7);
                assert!(left.b.abs_diff(right.b) <= 7);
            }
        }
    }

    #[test]
    fn truncated_poster_labels_use_pdf_safe_ascii_ellipsis() {
        let font = load_label_font().expect("test font");
        let text = truncate_text_to_width(&font, "settings_table_model.ex", 12.0, 85.0)
            .expect("truncated label");
        assert!(text.is_ascii());
        assert!(text.ends_with("..."));
    }

    #[test]
    fn unavailable_gpu_capabilities_fall_back_but_render_failures_do_not() {
        assert!(gpu_is_unavailable(&anyhow::anyhow!(
            "no compatible GPU adapter is available"
        )));
        assert!(gpu_is_unavailable(&anyhow::anyhow!(
            "adapter does not support blendable RGBA16Float density textures"
        )));
        assert!(gpu_is_unavailable(&anyhow::anyhow!(
            "adapter does not support 4x multisampled RGBA16Float density textures"
        )));
        assert!(!gpu_is_unavailable(&anyhow::anyhow!("GPU readback failed")));
    }

    #[test]
    fn child_label_moves_below_an_overlapping_parent_label() {
        let container = Rect {
            x0: 0.0,
            y0: 0.0,
            x1: 200.0,
            y1: 100.0,
        };
        let parent = label_bounds(5.0, 16.0, 70.0, 11.0);
        let (baseline, child) =
            place_label(container, 5.0, 14.0, 9.0, 60.0, 3.0, &[parent]).unwrap();

        assert!(baseline > 16.0);
        assert!(!child.overlaps(parent, 3.0));
        assert!(label_fits(container, child));
    }

    #[test]
    fn label_is_omitted_when_no_collision_free_band_fits() {
        let container = Rect {
            x0: 0.0,
            y0: 0.0,
            x1: 100.0,
            y1: 24.0,
        };
        let parent = label_bounds(4.0, 13.0, 70.0, 10.0);

        assert!(place_label(container, 4.0, 12.0, 9.0, 60.0, 3.0, &[parent]).is_none());
    }

    #[test]
    fn small_file_parcels_receive_abbreviated_micro_labels() {
        let Some(font) = load_label_font() else {
            return;
        };
        let parcel = Rect {
            x0: 0.0,
            y0: 0.0,
            x1: 42.0,
            y1: 12.0,
        };
        let label = fit_file_label(&font, "a_very_long_source_filename.ex", parcel, 1.0, &[])
            .expect("small but readable parcel should have a label");

        assert!(label.text.ends_with("..."));
        assert!(label.size >= 1.5);
        assert!(measure_text(&font, &label.text, label.size) <= parcel.width() as f32);
    }

    #[test]
    fn microscopic_parcels_receive_a_compact_zoom_label() {
        let Some(font) = load_label_font() else {
            return;
        };
        let parcel = Rect {
            x0: 0.0,
            y0: 0.0,
            x1: 4.0,
            y1: 4.0,
        };

        let label = fit_file_label(&font, "tiny.ex", parcel, 1.0, &[])
            .expect("even a microscopic parcel should retain a zoom label");
        assert!(!label.text.is_empty());
    }

    #[test]
    fn narrow_tall_parcels_use_vertical_labels() {
        let Some(font) = load_label_font() else {
            return;
        };
        let parcel = Rect {
            x0: 0.0,
            y0: 0.0,
            x1: 12.0,
            y1: 70.0,
        };
        let label = fit_file_label(&font, "formatter.ex", parcel, 1.0, &[])
            .expect("tall parcel should have a rotated label");

        assert!(matches!(label.placement, LabelPlacement::Vertical { .. }));
    }

    #[test]
    fn density_exposure_preserves_small_graphs_and_lowers_large_graphs() {
        let options = RenderOptions::default();
        assert_eq!(effective_call_opacity(2_500, &options), 34);
        assert_eq!(effective_call_opacity(5_345, &options), 34);
        assert_eq!(effective_call_opacity(10_000, &options), 34);
        assert_eq!(effective_call_opacity(21_307, &options), 15);
        assert_eq!(effective_call_opacity(208_147, &options), 1);

        let fixed = RenderOptions {
            density_aware_exposure: false,
            ..options
        };
        assert_eq!(effective_call_opacity(21_307, &fixed), 34);
    }

    #[test]
    fn optical_density_is_order_independent_in_light_and_dark_themes() {
        let first = [Point { x: 4.0, y: 8.0 }, Point { x: 60.0, y: 55.0 }];
        let second = [Point { x: 5.0, y: 56.0 }, Point { x: 59.0, y: 7.0 }];
        for theme in [Theme::Architect, Theme::Night] {
            let palette = Palette::for_theme(theme);
            let mut forward = OpticalDensityLayer::new(64, 64).unwrap();
            forward.deposit_pencil_call(
                &first,
                palette.direction_source,
                palette.direction_target,
                34,
                1.4,
                11,
            );
            forward.deposit_pencil_call(
                &second,
                palette.direction_target,
                palette.direction_source,
                34,
                1.4,
                22,
            );
            let mut reverse = OpticalDensityLayer::new(64, 64).unwrap();
            reverse.deposit_pencil_call(
                &second,
                palette.direction_target,
                palette.direction_source,
                34,
                1.4,
                22,
            );
            reverse.deposit_pencil_call(
                &first,
                palette.direction_source,
                palette.direction_target,
                34,
                1.4,
                11,
            );

            assert_eq!(
                forward.resolve().unwrap().data(),
                reverse.resolve().unwrap().data()
            );
        }
    }

    #[test]
    fn repeated_routes_deposit_more_optical_density_in_every_theme() {
        let points = [Point { x: 4.0, y: 32.0 }, Point { x: 60.0, y: 32.0 }];
        for theme in [Theme::Architect, Theme::Night] {
            let palette = Palette::for_theme(theme);
            let mut single = OpticalDensityLayer::new(64, 64).unwrap();
            single.deposit_pencil_call(
                &points,
                palette.direction_source,
                palette.direction_target,
                34,
                1.4,
                11,
            );
            let single = single.resolve().unwrap();
            let mut double = OpticalDensityLayer::new(64, 64).unwrap();
            for _ in 0..2 {
                double.deposit_pencil_call(
                    &points,
                    palette.direction_source,
                    palette.direction_target,
                    34,
                    1.4,
                    11,
                );
            }
            let double = double.resolve().unwrap();
            let alpha = |pixmap: &Pixmap| {
                pixmap
                    .data()
                    .chunks_exact(4)
                    .map(|pixel| u64::from(pixel[3]))
                    .sum::<u64>()
            };
            assert!(alpha(&double) > alpha(&single));
        }
    }
}
