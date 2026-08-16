use std::{
    fs::{self, File},
    io::{BufWriter, Seek, Write},
    path::{Path, PathBuf},
    sync::{
        Mutex,
        mpsc::{self, Receiver},
    },
    thread,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tiny_skia::Pixmap;
use zlib_rs::{DeflateConfig, ReturnCode, compress_bound, compress_slice};

use crate::{
    layout::{LayoutOptions, sample_rect},
    model::{Atlas, NodeKind, Point, Rect},
    render::{
        CallTile, CallTileViewport, LabelCommand, LabelPlacement, Palette, PdfTileCacheStats,
        PdfTileTimingStats, RenderBackend, RenderOptions, RenderStats, Rgba, Theme,
        collect_label_commands, effective_call_opacity, jitter_points, plan_call_tiles,
        render_call_layer_png, render_software_call_tile,
    },
    render_wgpu::WgpuDensityRenderer,
};

pub(crate) fn render_vector_artifact(
    atlas: &Atlas,
    output: &Path,
    layout_options: &LayoutOptions,
    render_options: &RenderOptions,
) -> Result<RenderStats> {
    let started = Instant::now();
    let format = output
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !matches!(format.as_str(), "svg" | "pdf") {
        bail!("vector output must use a .svg or .pdf extension");
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("cannot create output directory {}", parent.display()))?;
    }

    let palette = Palette::for_theme(render_options.theme);
    let (labels, label_stats) = collect_label_commands(atlas, layout_options, palette);
    if format == "svg" {
        let calls_path = temporary_call_layer_path(output);
        let mut stats = render_call_layer_png(atlas, &calls_path, layout_options, render_options)?;
        let call_png = fs::read(&calls_path)
            .with_context(|| format!("cannot read call layer {}", calls_path.display()))?;
        let result = write_svg(
            atlas,
            output,
            layout_options,
            render_options,
            palette,
            &labels,
            &call_png,
        );
        let _ = fs::remove_file(&calls_path);
        result?;
        stats.backend = format!("{}+svg", stats.backend);
        stats.file_labels_drawn = label_stats.files;
        stats.directory_labels_drawn = label_stats.directories;
        stats.output_bytes = fs::metadata(output)?.len();
        stats.render_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        return Ok(stats);
    }

    let call_layer_dpi = effective_pdf_call_dpi(render_options)?;
    let call_scale = call_layer_dpi / render_options.pdf_dpi;
    let (scaled_atlas, scaled_layout) = scale_call_layer(atlas, layout_options, call_scale)?;
    let tiled = write_tiled_pdf(
        output,
        &PdfScene {
            atlas,
            call_atlas: &scaled_atlas,
            layout: layout_options,
            call_layout: &scaled_layout,
            render: render_options,
            palette,
            labels: &labels,
        },
    )?;
    Ok(RenderStats {
        backend: tiled.backend,
        gpu_adapter: tiled.gpu_adapter,
        width: layout_options.width,
        height: layout_options.height,
        call_layer_width: scaled_layout.width,
        call_layer_height: scaled_layout.height,
        call_layer_dpi: Some(call_layer_dpi),
        call_layer_tiles: tiled.tile_count,
        call_layer_tile_size: Some(render_options.pdf_call_tile_size),
        call_layer_overlap: Some(tiled.overlap),
        files_drawn: atlas.files().count(),
        file_labels_drawn: label_stats.files,
        directory_labels_drawn: label_stats.directories,
        calls_drawn: tiled.calls_drawn,
        directory_boundaries_drawn: atlas
            .nodes
            .iter()
            .filter(|node| node.kind == NodeKind::Directory && node.id != 0)
            .count(),
        output_bytes: fs::metadata(output)?.len(),
        effective_call_opacity: effective_call_opacity(atlas.calls.len(), render_options),
        call_compositing: crate::render::optical_density_description(),
        direction_source_color: palette.direction_source.hex(),
        direction_target_color: palette.direction_target.hex(),
        render_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        pdf_tile_cache: Some(tiled.cache),
    })
}

fn effective_pdf_call_dpi(render: &RenderOptions) -> Result<f64> {
    ensure!(
        render.pdf_dpi.is_finite() && render.pdf_dpi > 0.0,
        "--pdf-dpi must be a positive finite number"
    );
    let call_dpi = render.pdf_call_dpi.unwrap_or(render.pdf_dpi);
    ensure!(
        call_dpi.is_finite() && call_dpi > 0.0,
        "--pdf-call-dpi must be a positive finite number"
    );
    ensure!(
        call_dpi >= render.pdf_dpi,
        "--pdf-call-dpi must be at least --pdf-dpi ({:.2})",
        render.pdf_dpi
    );
    Ok(call_dpi)
}

fn scale_call_layer(
    atlas: &Atlas,
    layout: &LayoutOptions,
    scale: f64,
) -> Result<(Atlas, LayoutOptions)> {
    let width = scaled_dimension(layout.width, scale)?;
    let height = scaled_dimension(layout.height, scale)?;
    let scale_x = width as f64 / layout.width as f64;
    let scale_y = height as f64 / layout.height as f64;
    let mut atlas = atlas.clone();
    for node in &mut atlas.nodes {
        node.rect.x0 *= scale_x;
        node.rect.x1 *= scale_x;
        node.rect.y0 *= scale_y;
        node.rect.y1 *= scale_y;
    }
    let mut layout = layout.clone();
    layout.width = width;
    layout.height = height;
    layout.directory_padding *= scale_x.min(scale_y);
    Ok((atlas, layout))
}

fn scaled_dimension(value: u32, scale: f64) -> Result<u32> {
    let scaled = f64::from(value) * scale;
    ensure!(
        scaled.is_finite() && scaled >= 1.0 && scaled <= f64::from(u32::MAX),
        "requested PDF call-layer dimensions are out of range"
    );
    Ok(scaled.round() as u32)
}

fn temporary_call_layer_path(output: &Path) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let name = output
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("atlas");
    output.with_file_name(format!(".{name}.calls-{}-{stamp}.png", std::process::id()))
}

fn write_svg(
    atlas: &Atlas,
    output: &Path,
    layout: &LayoutOptions,
    render: &RenderOptions,
    palette: Palette,
    labels: &[LabelCommand],
    call_png: &[u8],
) -> Result<()> {
    let width = layout.width;
    let height = layout.height;
    let drawing_scale = (width.min(height) as f32 / 1080.0).max(0.5);
    let mut svg = String::with_capacity(call_png.len() * 4 / 3 + atlas.nodes.len() * 180);
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">\n"
    ));
    svg.push_str("<title>Code Atlas</title>\n<desc>Vector treemap and labels with an embedded raster layer containing every directed callsite.</desc>\n");
    svg_rect(&mut svg, frame_rect(layout), palette.background, None, 0.0);
    svg_rect(&mut svg, atlas.nodes[0].rect, palette.land, None, 0.0);
    for node in atlas
        .nodes
        .iter()
        .filter(|node| node.kind == NodeKind::File)
    {
        svg_rect(
            &mut svg,
            node.rect,
            palette.file_color(&node.language),
            None,
            0.0,
        );
    }

    svg.push_str(&format!(
        "<image id=\"call-density\" x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" href=\"data:image/png;base64,{}\"/>\n",
        BASE64.encode(call_png)
    ));

    let architectural = matches!(render.theme, Theme::Architect | Theme::Night);
    for node in atlas
        .nodes
        .iter()
        .filter(|node| node.kind == NodeKind::File)
    {
        let width = render.boundary_width * 0.62 * drawing_scale;
        if architectural {
            svg_pencil_rect(
                &mut svg,
                node.rect,
                palette.file_border,
                width,
                node.id as u64 ^ layout.texture_seed,
                0.24 * drawing_scale as f64,
                2,
            );
        } else {
            svg_rect(
                &mut svg,
                node.rect,
                Rgba::with_alpha(0, 0, 0, 0),
                Some(palette.file_border),
                width,
            );
        }
    }
    let mut directories: Vec<_> = atlas
        .nodes
        .iter()
        .filter(|node| node.kind == NodeKind::Directory && node.id != 0)
        .collect();
    directories.sort_by_key(|node| node.depth);
    for node in directories {
        let stroke_width =
            render.boundary_width * drawing_scale * (1.0 + 0.35 / node.depth.max(1) as f32);
        if architectural {
            svg_pencil_rect(
                &mut svg,
                node.rect,
                palette.directory_border,
                stroke_width,
                node.id as u64 ^ 0xd1ec_7000,
                0.34 * drawing_scale as f64,
                3,
            );
        } else {
            svg_rect(
                &mut svg,
                node.rect,
                Rgba::with_alpha(0, 0, 0, 0),
                Some(palette.directory_border),
                stroke_width,
            );
        }
    }
    let outer_width = render.boundary_width * 1.8 * drawing_scale;
    if architectural {
        svg_pencil_rect(
            &mut svg,
            atlas.nodes[0].rect,
            palette.outer_border,
            outer_width,
            layout.texture_seed ^ 0xc0a5_7000,
            0.28 * drawing_scale as f64,
            3,
        );
    } else {
        svg_rect(
            &mut svg,
            atlas.nodes[0].rect,
            Rgba::with_alpha(0, 0, 0, 0),
            Some(palette.outer_border),
            outer_width,
        );
    }
    svg.push_str(
        "<g id=\"labels\" font-family=\"SFMono-Regular, Menlo, DejaVu Sans Mono, monospace\">\n",
    );
    for label in labels {
        let text = xml_escape(&label.text);
        let color = svg_color(label.color);
        let opacity = label.color.a as f32 / 255.0;
        match label.placement {
            LabelPlacement::Horizontal { x, baseline } => svg.push_str(&format!(
                "<text x=\"{x:.3}\" y=\"{baseline:.3}\" font-size=\"{:.3}\" fill=\"{color}\" fill-opacity=\"{opacity:.4}\">{text}</text>\n",
                label.size
            )),
            LabelPlacement::Vertical { x, y } => svg.push_str(&format!(
                "<text transform=\"translate({x:.3} {y:.3}) rotate(90)\" y=\"-{:.3}\" font-size=\"{:.3}\" fill=\"{color}\" fill-opacity=\"{opacity:.4}\">{text}</text>\n",
                label.size * 0.18,
                label.size
            )),
            LabelPlacement::Signature { x, baseline } => {
                svg_hollow_heart(
                    &mut svg,
                    x,
                    baseline - label.size * 0.88,
                    label.size * 0.92,
                    label.color,
                );
                svg.push_str(&format!(
                    "<text x=\"{:.3}\" y=\"{baseline:.3}\" font-size=\"{:.3}\" fill=\"{color}\" fill-opacity=\"{opacity:.4}\">{text}</text>\n",
                    x + label.size * 1.18,
                    label.size
                ));
            }
        }
    }
    svg.push_str("</g>\n</svg>\n");
    fs::write(output, svg).with_context(|| format!("cannot save {}", output.display()))
}

fn svg_rect(output: &mut String, rect: Rect, fill: Rgba, stroke: Option<Rgba>, width: f32) {
    let fill_value = if fill.a == 0 {
        "none".to_owned()
    } else {
        svg_color(fill)
    };
    output.push_str(&format!(
        r#"<rect x="{:.3}" y="{:.3}" width="{:.3}" height="{:.3}" fill="{}" fill-opacity="{:.4}""#,
        rect.x0,
        rect.y0,
        rect.width().max(0.0),
        rect.height().max(0.0),
        fill_value,
        fill.a as f32 / 255.0
    ));
    if let Some(stroke) = stroke {
        output.push_str(&format!(
            r#" stroke="{}" stroke-opacity="{:.4}" stroke-width="{width:.3}" vector-effect="non-scaling-stroke""#,
            svg_color(stroke),
            stroke.a as f32 / 255.0
        ));
    }
    output.push_str("/>\n");
}

fn svg_pencil_rect(
    output: &mut String,
    rect: Rect,
    color: Rgba,
    width: f32,
    seed: u64,
    amplitude: f64,
    passes: usize,
) {
    let base = sample_rect(rect, 4);
    for pass in 0..passes.max(1) {
        let points = jitter_points(&base, seed ^ pass as u64, amplitude * (pass + 1) as f64);
        let alpha = ((color.a as usize / passes.max(1)).max(1) as u8).saturating_add(5);
        output.push_str("<path d=\"");
        for (index, point) in points.iter().enumerate() {
            output.push_str(&format!(
                "{} {:.3} {:.3} ",
                if index == 0 { 'M' } else { 'L' },
                point.x,
                point.y
            ));
        }
        output.push_str(&format!(
            "Z\" fill=\"none\" stroke=\"{}\" stroke-opacity=\"{:.4}\" stroke-width=\"{:.3}\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>\n",
            svg_color(color),
            alpha as f32 / 255.0,
            width * (0.78 + pass as f32 * 0.13)
        ));
    }
}

fn svg_hollow_heart(output: &mut String, x: f32, y: f32, size: f32, color: Rgba) {
    output.push_str(&format!(
        "<path d=\"M {:.3} {:.3} C {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} C {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} C {:.3} {:.3} {:.3} {:.3} {:.3} {:.3} C {:.3} {:.3} {:.3} {:.3} {:.3} {:.3}\" fill=\"none\" stroke=\"{}\" stroke-opacity=\"{:.4}\" stroke-width=\"{:.3}\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>\n",
        x + size * 0.5,
        y + size * 0.92,
        x + size * 0.08,
        y + size * 0.64,
        x,
        y + size * 0.34,
        x + size * 0.21,
        y + size * 0.19,
        x + size * 0.36,
        y + size * 0.08,
        x + size * 0.5,
        y + size * 0.19,
        x + size * 0.5,
        y + size * 0.32,
        x + size * 0.5,
        y + size * 0.19,
        x + size * 0.64,
        y + size * 0.08,
        x + size * 0.79,
        y + size * 0.19,
        x + size,
        y + size * 0.34,
        x + size * 0.92,
        y + size * 0.64,
        x + size * 0.5,
        y + size * 0.92,
        svg_color(color),
        color.a as f32 / 255.0,
        (size * 0.075).max(0.55)
    ));
}

fn frame_rect(layout: &LayoutOptions) -> Rect {
    Rect {
        x0: 0.0,
        y0: 0.0,
        x1: layout.width as f64,
        y1: layout.height as f64,
    }
}

fn svg_color(color: Rgba) -> String {
    format!("#{:02x}{:02x}{:02x}", color.r, color.g, color.b)
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[derive(Debug, Clone)]
struct TiledPdfStats {
    calls_drawn: usize,
    tile_count: usize,
    overlap: u32,
    backend: String,
    gpu_adapter: Option<String>,
    cache: PdfTileCacheStats,
}

const TILE_CACHE_VERSION: u32 = 1;
const TILE_ARTIFACT_MAGIC: &[u8; 8] = b"CATILE01";
const PDF_DEFLATE_CODEC: &str = "zlib-rs";
// Level 6 more than halved OLI compression CPU while keeping the final PDF
// within 2.9% of the previous miniz level-6 output.
const PDF_DEFLATE_LEVEL: i32 = 6;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct TilePhaseTimings {
    render_and_readback_ms: u64,
    spline_evaluation_gpu_ms: Option<f64>,
    rasterization_gpu_ms: Option<f64>,
    rgba_split_ms: u64,
    compression_ms: u64,
    cache_write_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TileManifestEntry {
    index: usize,
    core_x: u32,
    core_y: u32,
    core_width: u32,
    core_height: u32,
    render_x: u32,
    render_y: u32,
    render_width: u32,
    render_height: u32,
    candidate_calls: usize,
    artifact: String,
    sha256: String,
    artifact_bytes: u64,
    timings: TilePhaseTimings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TileManifest {
    version: u32,
    cache_key: String,
    renderer_version: String,
    revision: String,
    backend: String,
    configuration: serde_json::Value,
    total_calls: usize,
    tile_count: usize,
    completed_tiles: usize,
    tiles: Vec<Option<TileManifestEntry>>,
}

struct CompressedTile {
    width: u32,
    height: u32,
    rgb: Vec<u8>,
    alpha: Vec<u8>,
}

struct CompletedTileCompression {
    index: usize,
    artifact: String,
    sha256: String,
    artifact_bytes: u64,
    timings: TilePhaseTimings,
}

struct PdfScene<'a> {
    atlas: &'a Atlas,
    call_atlas: &'a Atlas,
    layout: &'a LayoutOptions,
    call_layout: &'a LayoutOptions,
    render: &'a RenderOptions,
    palette: Palette,
    labels: &'a [LabelCommand],
}

fn write_tiled_pdf(output: &Path, scene: &PdfScene<'_>) -> Result<TiledPdfStats> {
    let call_atlas = scene.call_atlas;
    let layout = scene.layout;
    let call_layout = scene.call_layout;
    let render = scene.render;
    let planning_started = Instant::now();
    let (tiles, calls_drawn, overlap) = plan_call_tiles(call_atlas, call_layout, render)?;
    let planning_ms = elapsed_ms(planning_started);
    let mut gpu_renderer = if render.backend == RenderBackend::Wgpu {
        match WgpuDensityRenderer::new() {
            Ok(renderer) => Some(renderer),
            Err(error) if crate::render::gpu_is_unavailable(&error) => {
                eprintln!(
                    "wgpu is unavailable ({error:#}); falling back to software optical-density PDF tiles"
                );
                None
            }
            Err(error) => return Err(error),
        }
    } else {
        None
    };
    if let Some(renderer) = &gpu_renderer {
        let limit = renderer.max_texture_dimension();
        if tiles
            .iter()
            .any(|tile| tile.render_width > limit || tile.render_height > limit)
        {
            eprintln!(
                "PDF call tiles exceed the GPU adapter's {limit}-pixel texture limit; falling back to software optical-density tiles"
            );
            gpu_renderer = None;
        }
    }
    let scene_prepare_started = Instant::now();
    let gpu_scene = if let Some(renderer) = &gpu_renderer {
        match renderer.prepare_scene(call_atlas) {
            Ok(scene) => Some(scene),
            Err(error) if crate::render::gpu_is_unavailable(&error) => {
                eprintln!(
                    "wgpu cannot retain this spline scene ({error:#}); falling back to software optical-density PDF tiles"
                );
                gpu_renderer = None;
                None
            }
            Err(error) => return Err(error),
        }
    } else {
        None
    };
    let scene_prepare_ms = elapsed_ms(scene_prepare_started);
    let backend = if gpu_renderer.is_some() {
        "wgpu spline-evaluated optical-density tiled+pdf".to_owned()
    } else if render.backend == RenderBackend::Wgpu {
        "software optical-density tiled (wgpu unavailable)+pdf".to_owned()
    } else {
        "software-tiled+pdf".to_owned()
    };
    let (cache_key, cache_configuration) = pdf_tile_cache_identity(scene, &backend, overlap)?;
    let cache_dir = pdf_tile_cache_dir(output, render.pdf_resume);
    let manifest_path = cache_dir.join("manifest.json");
    if render.pdf_restart && cache_dir.exists() {
        fs::remove_dir_all(&cache_dir)
            .with_context(|| format!("cannot restart tile cache {}", cache_dir.display()))?;
    }
    fs::create_dir_all(&cache_dir)
        .with_context(|| format!("cannot create tile cache {}", cache_dir.display()))?;
    let mut manifest = load_or_create_manifest(
        &manifest_path,
        &cache_key,
        &backend,
        call_atlas,
        tiles.len(),
        calls_drawn,
        &cache_configuration,
    )?;
    let mut invocation_timings = PdfTileTimingStats {
        planning_ms,
        scene_prepare_ms,
        ..PdfTileTimingStats::default()
    };
    ensure!(
        render.pdf_compression_workers <= 64,
        "--pdf-compression-workers must be between 0 and 64"
    );
    let compression_workers = pdf_compression_worker_count(render.pdf_compression_workers);
    let max_in_flight_tiles = if render.pdf_stop_after_tiles.is_some() {
        1
    } else {
        compression_workers
    };
    let mut tiles_rendered = 0_usize;
    let mut tiles_reused = 0_usize;
    let tile_pipeline_started = Instant::now();
    let (sender, receiver) = mpsc::channel::<Result<CompletedTileCompression>>();
    let receiver = Mutex::new(receiver);
    rayon::scope_fifo(|scope| -> Result<()> {
        let mut in_flight = 0_usize;
        for (index, tile) in tiles.iter().enumerate() {
            if manifest.tiles[index]
                .as_ref()
                .is_some_and(|entry| cached_tile_is_valid(&cache_dir, entry, tile))
            {
                tiles_reused += 1;
                eprintln!(
                    "  reusing PDF call tile {}/{} ({} candidate callsites)",
                    index + 1,
                    tiles.len(),
                    tile.call_indices.len()
                );
                continue;
            }
            while in_flight >= max_in_flight_tiles {
                finish_next_compressed_tile(
                    &receiver,
                    &tiles,
                    &manifest_path,
                    &mut manifest,
                    &mut invocation_timings,
                    &mut tiles_rendered,
                    render.pdf_stop_after_tiles,
                )?;
                in_flight -= 1;
            }
            eprintln!(
                "  rendering PDF call tile {}/{} ({} candidate callsites)",
                index + 1,
                tiles.len(),
                tile.call_indices.len()
            );
            let viewport = CallTileViewport {
                x: tile.render_x,
                y: tile.render_y,
                width: tile.render_width,
                height: tile.render_height,
            };
            let render_started = Instant::now();
            let (pixmap, gpu_timings) =
                if let (Some(renderer), Some(gpu_scene)) = (&gpu_renderer, &gpu_scene) {
                    let (pixmap, _, timings) = renderer.render_call_tile_profiled(
                        gpu_scene,
                        call_layout,
                        render,
                        &tile.call_indices,
                        viewport,
                    )?;
                    (pixmap, timings)
                } else {
                    let (pixmap, _) = render_software_call_tile(
                        call_atlas,
                        call_layout,
                        render,
                        &tile.call_indices,
                        viewport,
                    )?;
                    (pixmap, Default::default())
                };
            let timings = TilePhaseTimings {
                render_and_readback_ms: elapsed_ms(render_started),
                spline_evaluation_gpu_ms: gpu_timings.spline_evaluation_ms,
                rasterization_gpu_ms: gpu_timings.rasterization_ms,
                ..TilePhaseTimings::default()
            };
            let artifact = format!("tile-{index:05}.catile");
            let artifact_path = cache_dir.join(&artifact);
            let result_sender = sender.clone();
            scope.spawn_fifo(move |_| {
                let result =
                    compress_and_cache_tile(index, pixmap, artifact, artifact_path, timings);
                let _ = result_sender.send(result);
            });
            in_flight += 1;
        }
        while in_flight > 0 {
            finish_next_compressed_tile(
                &receiver,
                &tiles,
                &manifest_path,
                &mut manifest,
                &mut invocation_timings,
                &mut tiles_rendered,
                render.pdf_stop_after_tiles,
            )?;
            in_flight -= 1;
        }
        Ok(())
    })?;
    invocation_timings.tile_pipeline_wall_ms = elapsed_ms(tile_pipeline_started);

    manifest.completed_tiles = manifest.tiles.iter().flatten().count();
    write_json_atomic(&manifest_path, &manifest)?;
    ensure!(
        manifest.completed_tiles == tiles.len(),
        "PDF tile cache is incomplete after rendering"
    );
    let scale = 72.0 / render.pdf_dpi.max(1.0);
    let page_width = layout.width as f64 * scale;
    let page_height = layout.height as f64 * scale;
    let content = pdf_content(scene, &tiles, scale);
    let content = deflate(content.as_bytes())?;

    let font_id = 4 + tiles.len() * 2;
    let content_id = font_id + 1;
    let mut xobjects = String::new();
    for index in 0..tiles.len() {
        let rgb_id = 4 + index * 2;
        xobjects.push_str(&format!("/Calls{index} {rgb_id} 0 R "));
    }
    let page = format!(
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width:.4} {page_height:.4}] /Resources << /XObject << {xobjects}>> /Font << /Mono {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
    );
    let assembly_started = Instant::now();
    let temporary_pdf = atomic_temporary_path(output);
    let mut pdf = StreamingPdf::create(&temporary_pdf)?;
    pdf.write_object(b"<< /Type /Catalog /Pages 2 0 R >>")?;
    pdf.write_object(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")?;
    pdf.write_object(page.as_bytes())?;
    for (index, entry) in manifest.tiles.iter().enumerate() {
        let entry = entry.as_ref().context("completed PDF tile is missing")?;
        let artifact_path = cache_dir.join(&entry.artifact);
        let compressed = read_tile_artifact(&artifact_path, &entry.sha256)?;
        let alpha_id = 5 + index * 2;
        pdf.write_stream(
            &format!(
                "/Type /XObject /Subtype /Image /Width {} /Height {} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Interpolate false /Filter /FlateDecode /SMask {alpha_id} 0 R",
                compressed.width,
                compressed.height
            ),
            &compressed.rgb,
        )?;
        pdf.write_stream(
            &format!(
                "/Type /XObject /Subtype /Image /Width {} /Height {} /ColorSpace /DeviceGray /BitsPerComponent 8 /Interpolate false /Filter /FlateDecode",
                compressed.width,
                compressed.height
            ),
            &compressed.alpha,
        )?;
    }
    pdf.write_object(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>",
    )?;
    pdf.write_stream("/Filter /FlateDecode", &content)?;
    pdf.finish()?;
    fs::rename(&temporary_pdf, output)
        .with_context(|| format!("cannot atomically replace {}", output.display()))?;
    invocation_timings.pdf_assembly_ms = elapsed_ms(assembly_started);
    if !render.pdf_resume {
        fs::remove_dir_all(&cache_dir).with_context(|| {
            format!("cannot remove temporary tile cache {}", cache_dir.display())
        })?;
    }
    Ok(TiledPdfStats {
        calls_drawn,
        tile_count: tiles.len(),
        overlap,
        backend,
        gpu_adapter: gpu_renderer
            .as_ref()
            .map(|renderer| renderer.adapter_name().to_owned()),
        cache: PdfTileCacheStats {
            enabled: render.pdf_resume,
            cache_key,
            manifest_path: manifest_path.display().to_string(),
            tiles_rendered,
            tiles_reused,
            compression_workers,
            max_in_flight_tiles,
            compression_codec: PDF_DEFLATE_CODEC.to_owned(),
            compression_level: PDF_DEFLATE_LEVEL,
            timings: invocation_timings,
        },
    })
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}

fn pdf_tile_cache_identity(
    scene: &PdfScene<'_>,
    backend: &str,
    overlap: u32,
) -> Result<(String, serde_json::Value)> {
    let render = scene.render;
    let configuration = serde_json::json!({
        "cache_version": TILE_CACHE_VERSION,
        "renderer_version": env!("CARGO_PKG_VERSION"),
        "revision": &scene.call_atlas.revision,
        "dirty": scene.call_atlas.dirty,
        "excluded_paths": &scene.call_atlas.excluded_paths,
        "files": scene.call_atlas.files().count(),
        "calls": scene.call_atlas.calls.len(),
        "layout": {
            "width": scene.call_layout.width,
            "height": scene.call_layout.height,
            "margin_fraction": scene.call_layout.margin_fraction,
            "directory_padding": scene.call_layout.directory_padding,
            "texture_seed": scene.call_layout.texture_seed,
            "bundle_strength": scene.call_layout.bundle_strength,
        },
        "render": {
            "backend": backend,
            "theme": render.theme.to_string(),
            "call_opacity": render.call_opacity,
            "call_width": render.call_width,
            "density_aware_exposure": render.density_aware_exposure,
            "density_reference_calls": render.density_reference_calls,
            "pdf_dpi": render.pdf_dpi,
            "pdf_call_dpi": render.pdf_call_dpi,
            "pdf_call_tile_size": render.pdf_call_tile_size,
            "overlap": overlap,
        },
        "compression": {
            "codec": PDF_DEFLATE_CODEC,
            "level": PDF_DEFLATE_LEVEL,
        },
    });
    let mut hasher = Sha256::new();
    hasher.update(serde_json::to_vec(&configuration)?);
    hasher.update(serde_json::to_vec(&scene.call_atlas.nodes)?);
    hasher.update(serde_json::to_vec(&scene.call_atlas.calls)?);
    Ok((hex_bytes(&hasher.finalize()), configuration))
}

fn pdf_tile_cache_dir(output: &Path, persistent: bool) -> PathBuf {
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("code-atlas.pdf");
    if persistent {
        output.with_file_name(format!("{name}.tiles"))
    } else {
        output.with_file_name(format!(
            ".{name}.tiles-{}-{}",
            std::process::id(),
            unique_stamp()
        ))
    }
}

fn load_or_create_manifest(
    path: &Path,
    cache_key: &str,
    backend: &str,
    atlas: &Atlas,
    tile_count: usize,
    total_calls: usize,
    configuration: &serde_json::Value,
) -> Result<TileManifest> {
    if path.exists() {
        let bytes = fs::read(path)
            .with_context(|| format!("cannot read tile manifest {}", path.display()))?;
        let manifest: TileManifest = serde_json::from_slice(&bytes)
            .with_context(|| format!("cannot parse tile manifest {}", path.display()))?;
        ensure!(
            manifest.version == TILE_CACHE_VERSION,
            "tile cache format changed; restart with --restart"
        );
        ensure!(
            manifest.cache_key == cache_key,
            "tile cache does not match this repository revision or render configuration; restart with --restart"
        );
        ensure!(
            manifest.tile_count == tile_count && manifest.tiles.len() == tile_count,
            "tile cache geometry changed; restart with --restart"
        );
        return Ok(manifest);
    }
    let manifest = TileManifest {
        version: TILE_CACHE_VERSION,
        cache_key: cache_key.to_owned(),
        renderer_version: env!("CARGO_PKG_VERSION").to_owned(),
        revision: atlas.revision.clone(),
        backend: backend.to_owned(),
        configuration: configuration.clone(),
        total_calls,
        tile_count,
        completed_tiles: 0,
        tiles: vec![None; tile_count],
    };
    write_json_atomic(path, &manifest)?;
    Ok(manifest)
}

fn cached_tile_is_valid(cache_dir: &Path, entry: &TileManifestEntry, tile: &CallTile) -> bool {
    entry.core_x == tile.core_x
        && entry.core_y == tile.core_y
        && entry.core_width == tile.core_width
        && entry.core_height == tile.core_height
        && entry.render_x == tile.render_x
        && entry.render_y == tile.render_y
        && entry.render_width == tile.render_width
        && entry.render_height == tile.render_height
        && entry.candidate_calls == tile.call_indices.len()
        && read_tile_artifact(&cache_dir.join(&entry.artifact), &entry.sha256).is_ok()
}

fn tile_manifest_entry(
    index: usize,
    tile: &CallTile,
    artifact: String,
    sha256: String,
    artifact_bytes: u64,
    timings: TilePhaseTimings,
) -> TileManifestEntry {
    TileManifestEntry {
        index,
        core_x: tile.core_x,
        core_y: tile.core_y,
        core_width: tile.core_width,
        core_height: tile.core_height,
        render_x: tile.render_x,
        render_y: tile.render_y,
        render_width: tile.render_width,
        render_height: tile.render_height,
        candidate_calls: tile.call_indices.len(),
        artifact,
        sha256,
        artifact_bytes,
        timings,
    }
}

fn pdf_compression_worker_count(requested: usize) -> usize {
    if requested > 0 {
        return requested.max(1);
    }
    thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .saturating_sub(1)
        .clamp(1, 4)
}

fn compress_and_cache_tile(
    index: usize,
    pixmap: Pixmap,
    artifact: String,
    artifact_path: PathBuf,
    mut timings: TilePhaseTimings,
) -> Result<CompletedTileCompression> {
    let split_started = Instant::now();
    let (rgb, alpha) = split_unpremultiplied_rgba(pixmap.data());
    timings.rgba_split_ms = elapsed_ms(split_started);
    let compression_started = Instant::now();
    let compressed = CompressedTile {
        width: pixmap.width(),
        height: pixmap.height(),
        rgb: deflate(&rgb)?,
        alpha: deflate(&alpha)?,
    };
    timings.compression_ms = elapsed_ms(compression_started);
    let cache_write_started = Instant::now();
    let (sha256, artifact_bytes) = write_tile_artifact_atomic(&artifact_path, &compressed)?;
    timings.cache_write_ms = elapsed_ms(cache_write_started);
    Ok(CompletedTileCompression {
        index,
        artifact,
        sha256,
        artifact_bytes,
        timings,
    })
}

fn finish_next_compressed_tile(
    receiver: &Mutex<Receiver<Result<CompletedTileCompression>>>,
    tiles: &[CallTile],
    manifest_path: &Path,
    manifest: &mut TileManifest,
    invocation_timings: &mut PdfTileTimingStats,
    tiles_rendered: &mut usize,
    stop_after_tiles: Option<usize>,
) -> Result<()> {
    let completed = receiver
        .lock()
        .map_err(|_| anyhow::anyhow!("PDF tile compression receiver lock was poisoned"))?
        .recv()
        .context("PDF tile compression worker stopped unexpectedly")??;
    let tile = tiles
        .get(completed.index)
        .context("PDF tile compression returned an invalid tile index")?;
    accumulate_tile_timings(invocation_timings, &completed.timings);
    manifest.tiles[completed.index] = Some(tile_manifest_entry(
        completed.index,
        tile,
        completed.artifact,
        completed.sha256,
        completed.artifact_bytes,
        completed.timings,
    ));
    manifest.completed_tiles = manifest.tiles.iter().flatten().count();
    write_json_atomic(manifest_path, manifest)?;
    *tiles_rendered += 1;
    if stop_after_tiles.is_some_and(|limit| *tiles_rendered >= limit) {
        bail!(
            "stopped after {} newly rendered PDF tiles; resume with --resume (manifest: {})",
            *tiles_rendered,
            manifest_path.display()
        );
    }
    Ok(())
}

fn accumulate_tile_timings(total: &mut PdfTileTimingStats, tile: &TilePhaseTimings) {
    total.render_and_readback_ms += tile.render_and_readback_ms;
    total.rgba_split_ms += tile.rgba_split_ms;
    total.compression_ms += tile.compression_ms;
    total.cache_write_ms += tile.cache_write_ms;
    total.spline_evaluation_gpu_ms = sum_optional_f64(
        total.spline_evaluation_gpu_ms,
        tile.spline_evaluation_gpu_ms,
    );
    total.rasterization_gpu_ms =
        sum_optional_f64(total.rasterization_gpu_ms, tile.rasterization_gpu_ms);
}

fn sum_optional_f64(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left + right),
        (None, Some(right)) => Some(right),
        (Some(left), None) => Some(left),
        (None, None) => None,
    }
}

fn write_tile_artifact_atomic(path: &Path, tile: &CompressedTile) -> Result<(String, u64)> {
    let mut bytes = Vec::with_capacity(32 + tile.rgb.len() + tile.alpha.len());
    bytes.extend_from_slice(TILE_ARTIFACT_MAGIC);
    bytes.extend_from_slice(&tile.width.to_le_bytes());
    bytes.extend_from_slice(&tile.height.to_le_bytes());
    bytes.extend_from_slice(&(tile.rgb.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(tile.alpha.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&tile.rgb);
    bytes.extend_from_slice(&tile.alpha);
    let checksum = sha256_hex(&bytes);
    write_bytes_atomic(path, &bytes)?;
    Ok((checksum, bytes.len() as u64))
}

fn read_tile_artifact(path: &Path, expected_sha256: &str) -> Result<CompressedTile> {
    let bytes = fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    ensure!(
        sha256_hex(&bytes) == expected_sha256,
        "tile checksum mismatch for {}",
        path.display()
    );
    ensure!(
        bytes.len() >= 32 && &bytes[..8] == TILE_ARTIFACT_MAGIC,
        "invalid tile artifact {}",
        path.display()
    );
    let width = u32::from_le_bytes(bytes[8..12].try_into()?);
    let height = u32::from_le_bytes(bytes[12..16].try_into()?);
    let rgb_len = u64::from_le_bytes(bytes[16..24].try_into()?) as usize;
    let alpha_len = u64::from_le_bytes(bytes[24..32].try_into()?) as usize;
    ensure!(
        32_usize
            .checked_add(rgb_len)
            .and_then(|length| length.checked_add(alpha_len))
            == Some(bytes.len()),
        "invalid tile lengths in {}",
        path.display()
    );
    Ok(CompressedTile {
        width,
        height,
        rgb: bytes[32..32 + rgb_len].to_vec(),
        alpha: bytes[32 + rgb_len..].to_vec(),
    })
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    write_bytes_atomic(path, &bytes)
}

fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = atomic_temporary_path(path);
    {
        let mut file = File::create(&temporary)
            .with_context(|| format!("cannot create {}", temporary.display()))?;
        file.write_all(bytes)?;
        file.sync_all()?;
    }
    fs::rename(&temporary, path)
        .with_context(|| format!("cannot atomically replace {}", path.display()))
}

fn atomic_temporary_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("artifact");
    path.with_file_name(format!(
        ".{name}.tmp-{}-{}",
        std::process::id(),
        unique_stamp()
    ))
}

fn unique_stamp() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex_bytes(&Sha256::digest(bytes))
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn pdf_content(scene: &PdfScene<'_>, call_tiles: &[CallTile], scale: f64) -> String {
    let atlas = scene.atlas;
    let layout = scene.layout;
    let render = scene.render;
    let palette = scene.palette;
    let labels = scene.labels;
    let call_layer_width = scene.call_layout.width;
    let call_layer_height = scene.call_layout.height;
    let mut content = String::new();
    pdf_fill_rect(
        &mut content,
        frame_rect(layout),
        palette.background,
        layout.height,
        scale,
    );
    pdf_fill_rect(
        &mut content,
        atlas.nodes[0].rect,
        palette.land,
        layout.height,
        scale,
    );
    for node in atlas
        .nodes
        .iter()
        .filter(|node| node.kind == NodeKind::File)
    {
        pdf_fill_rect(
            &mut content,
            node.rect,
            palette.file_color(&node.language),
            layout.height,
            scale,
        );
    }

    let page_width = layout.width as f64 * scale;
    let page_height = layout.height as f64 * scale;
    let pixel_width = page_width / f64::from(call_layer_width);
    let pixel_height = page_height / f64::from(call_layer_height);
    for (index, tile) in call_tiles.iter().enumerate() {
        let core_x = f64::from(tile.core_x) * pixel_width;
        let core_y =
            page_height - f64::from(tile.core_y.saturating_add(tile.core_height)) * pixel_height;
        let core_width = f64::from(tile.core_width) * pixel_width;
        let core_height = f64::from(tile.core_height) * pixel_height;
        let render_x = f64::from(tile.render_x) * pixel_width;
        let render_y = page_height
            - f64::from(tile.render_y.saturating_add(tile.render_height)) * pixel_height;
        let render_width = f64::from(tile.render_width) * pixel_width;
        let render_height = f64::from(tile.render_height) * pixel_height;
        content.push_str(&format!(
            "q {core_x:.6} {core_y:.6} {core_width:.6} {core_height:.6} re W n {render_width:.6} 0 0 {render_height:.6} {render_x:.6} {render_y:.6} cm /Calls{index} Do Q\n"
        ));
    }

    let drawing_scale = (layout.width.min(layout.height) as f32 / 1080.0).max(0.5);
    let architectural = matches!(render.theme, Theme::Architect | Theme::Night);
    for node in atlas
        .nodes
        .iter()
        .filter(|node| node.kind == NodeKind::File)
    {
        let backdrop = palette.file_color(&node.language);
        let width = render.boundary_width * 0.62 * drawing_scale;
        if architectural {
            pdf_pencil_rect(
                &mut content,
                node.rect,
                VectorPencil {
                    color: palette.file_border,
                    backdrop,
                    width,
                    seed: node.id as u64 ^ layout.texture_seed,
                    amplitude: 0.24 * drawing_scale as f64,
                    passes: 2,
                },
                layout.height,
                scale,
            );
        } else {
            pdf_stroke_rect(
                &mut content,
                node.rect,
                flatten_alpha(palette.file_border, backdrop),
                width,
                layout.height,
                scale,
            );
        }
    }
    let mut directories: Vec<_> = atlas
        .nodes
        .iter()
        .filter(|node| node.kind == NodeKind::Directory && node.id != 0)
        .collect();
    directories.sort_by_key(|node| node.depth);
    for node in directories {
        let width = render.boundary_width * drawing_scale * (1.0 + 0.35 / node.depth.max(1) as f32);
        if architectural {
            pdf_pencil_rect(
                &mut content,
                node.rect,
                VectorPencil {
                    color: palette.directory_border,
                    backdrop: palette.land,
                    width,
                    seed: node.id as u64 ^ 0xd1ec_7000,
                    amplitude: 0.34 * drawing_scale as f64,
                    passes: 3,
                },
                layout.height,
                scale,
            );
        } else {
            pdf_stroke_rect(
                &mut content,
                node.rect,
                flatten_alpha(palette.directory_border, palette.land),
                width,
                layout.height,
                scale,
            );
        }
    }
    let outer_width = render.boundary_width * 1.8 * drawing_scale;
    if architectural {
        pdf_pencil_rect(
            &mut content,
            atlas.nodes[0].rect,
            VectorPencil {
                color: palette.outer_border,
                backdrop: palette.background,
                width: outer_width,
                seed: layout.texture_seed ^ 0xc0a5_7000,
                amplitude: 0.28 * drawing_scale as f64,
                passes: 3,
            },
            layout.height,
            scale,
        );
    } else {
        pdf_stroke_rect(
            &mut content,
            atlas.nodes[0].rect,
            flatten_alpha(palette.outer_border, palette.background),
            outer_width,
            layout.height,
            scale,
        );
    }

    for label in labels {
        let color = flatten_alpha(label.color, palette.background);
        let text = pdf_escape(&label.text);
        let size = label.size as f64 * scale;
        let (r, g, b) = pdf_rgb(color);
        match label.placement {
            LabelPlacement::Horizontal { x, baseline } => {
                let x = x as f64 * scale;
                let y = (layout.height as f64 - baseline as f64) * scale;
                content.push_str(&format!(
                    "BT /Mono {size:.4} Tf {r:.5} {g:.5} {b:.5} rg 1 0 0 1 {x:.4} {y:.4} Tm ({text}) Tj ET\n"
                ));
            }
            LabelPlacement::Vertical { x, y } => {
                let x = (x + label.size) as f64 * scale;
                let y = (layout.height as f64 - y as f64) * scale;
                content.push_str(&format!(
                    "BT /Mono {size:.4} Tf {r:.5} {g:.5} {b:.5} rg 0 -1 1 0 {x:.4} {y:.4} Tm ({text}) Tj ET\n"
                ));
            }
            LabelPlacement::Signature { x, baseline } => {
                pdf_hollow_heart(
                    &mut content,
                    x,
                    baseline - label.size * 0.88,
                    label.size * 0.92,
                    color,
                    layout.height,
                    scale,
                );
                let x = (x + label.size * 1.18) as f64 * scale;
                let y = (layout.height as f64 - baseline as f64) * scale;
                content.push_str(&format!(
                    "BT /Mono {size:.4} Tf {r:.5} {g:.5} {b:.5} rg 1 0 0 1 {x:.4} {y:.4} Tm ({text}) Tj ET\n"
                ));
            }
        }
    }
    content
}

fn pdf_hollow_heart(
    output: &mut String,
    x: f32,
    y: f32,
    size: f32,
    color: Rgba,
    height: u32,
    scale: f64,
) {
    let (r, g, b) = pdf_rgb(color);
    let point = |px: f32, py: f32| (px as f64 * scale, (height as f64 - py as f64) * scale);
    let start = point(x + size * 0.5, y + size * 0.92);
    let c11 = point(x + size * 0.08, y + size * 0.64);
    let c12 = point(x, y + size * 0.34);
    let p1 = point(x + size * 0.21, y + size * 0.19);
    let c21 = point(x + size * 0.36, y + size * 0.08);
    let c22 = point(x + size * 0.5, y + size * 0.19);
    let p2 = point(x + size * 0.5, y + size * 0.32);
    let c31 = point(x + size * 0.5, y + size * 0.19);
    let c32 = point(x + size * 0.64, y + size * 0.08);
    let p3 = point(x + size * 0.79, y + size * 0.19);
    let c41 = point(x + size, y + size * 0.34);
    let c42 = point(x + size * 0.92, y + size * 0.64);
    output.push_str(&format!(
        "{r:.5} {g:.5} {b:.5} RG {:.4} w 1 J 1 j {:.4} {:.4} m {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} c {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} c {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} c {:.4} {:.4} {:.4} {:.4} {:.4} {:.4} c S\n",
        (size * 0.075).max(0.55) as f64 * scale,
        start.0,
        start.1,
        c11.0,
        c11.1,
        c12.0,
        c12.1,
        p1.0,
        p1.1,
        c21.0,
        c21.1,
        c22.0,
        c22.1,
        p2.0,
        p2.1,
        c31.0,
        c31.1,
        c32.0,
        c32.1,
        p3.0,
        p3.1,
        c41.0,
        c41.1,
        c42.0,
        c42.1,
        start.0,
        start.1,
    ));
}

fn pdf_fill_rect(output: &mut String, rect: Rect, color: Rgba, height: u32, scale: f64) {
    let (r, g, b) = pdf_rgb(color);
    let x = rect.x0 * scale;
    let y = (height as f64 - rect.y1) * scale;
    output.push_str(&format!(
        "{r:.5} {g:.5} {b:.5} rg {x:.4} {y:.4} {:.4} {:.4} re f\n",
        rect.width() * scale,
        rect.height() * scale
    ));
}

#[derive(Clone, Copy)]
struct VectorPencil {
    color: Rgba,
    backdrop: Rgba,
    width: f32,
    seed: u64,
    amplitude: f64,
    passes: usize,
}

fn pdf_pencil_rect(output: &mut String, rect: Rect, pencil: VectorPencil, height: u32, scale: f64) {
    let base = sample_rect(rect, 4);
    for pass in 0..pencil.passes.max(1) {
        let points = jitter_points(
            &base,
            pencil.seed ^ pass as u64,
            pencil.amplitude * (pass + 1) as f64,
        );
        let mut color = pencil.color;
        color.a = ((color.a as usize / pencil.passes.max(1)).max(1) as u8).saturating_add(5);
        pdf_stroke_path(
            output,
            &points,
            flatten_alpha(color, pencil.backdrop),
            pencil.width * (0.78 + pass as f32 * 0.13),
            height,
            scale,
        );
    }
}

fn pdf_stroke_path(
    output: &mut String,
    points: &[Point],
    color: Rgba,
    width: f32,
    height: u32,
    scale: f64,
) {
    let Some(first) = points.first() else {
        return;
    };
    let (r, g, b) = pdf_rgb(color);
    output.push_str(&format!(
        "{r:.5} {g:.5} {b:.5} RG {:.4} w 1 J 1 j {:.4} {:.4} m ",
        width as f64 * scale,
        first.x * scale,
        (height as f64 - first.y) * scale
    ));
    for point in &points[1..] {
        output.push_str(&format!(
            "{:.4} {:.4} l ",
            point.x * scale,
            (height as f64 - point.y) * scale
        ));
    }
    output.push_str("h S\n");
}

fn pdf_stroke_rect(
    output: &mut String,
    rect: Rect,
    color: Rgba,
    width: f32,
    height: u32,
    scale: f64,
) {
    let (r, g, b) = pdf_rgb(color);
    let x = rect.x0 * scale;
    let y = (height as f64 - rect.y1) * scale;
    let width_points = width as f64 * scale;
    output.push_str(&format!(
        "{r:.5} {g:.5} {b:.5} RG {width_points:.4} w {x:.4} {y:.4} {:.4} {:.4} re S\n",
        rect.width() * scale,
        rect.height() * scale
    ));
}

fn pdf_rgb(color: Rgba) -> (f64, f64, f64) {
    (
        color.r as f64 / 255.0,
        color.g as f64 / 255.0,
        color.b as f64 / 255.0,
    )
}

fn flatten_alpha(foreground: Rgba, background: Rgba) -> Rgba {
    let alpha = foreground.a as f32 / 255.0;
    let channel =
        |front: u8, back: u8| (front as f32 * alpha + back as f32 * (1.0 - alpha)).round() as u8;
    Rgba::with_alpha(
        channel(foreground.r, background.r),
        channel(foreground.g, background.g),
        channel(foreground.b, background.b),
        255,
    )
}

fn pdf_escape(value: &str) -> String {
    value
        .chars()
        .map(|character| if character.is_ascii() { character } else { '?' })
        .collect::<String>()
        .replace('\\', "\\\\")
        .replace('(', "\\(")
        .replace(')', "\\)")
}

fn split_unpremultiplied_rgba(data: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let mut rgb = Vec::with_capacity(data.len() / 4 * 3);
    let mut alpha = Vec::with_capacity(data.len() / 4);
    for pixel in data.chunks_exact(4) {
        let a = pixel[3];
        alpha.push(a);
        for channel in &pixel[..3] {
            rgb.push(if a == 0 {
                0
            } else {
                ((u16::from(*channel) * 255 + u16::from(a) / 2) / u16::from(a)).min(255) as u8
            });
        }
    }
    (rgb, alpha)
}

fn deflate(data: &[u8]) -> Result<Vec<u8>> {
    let mut output = vec![0; compress_bound(data.len())];
    let (compressed, result) =
        compress_slice(&mut output, data, DeflateConfig::new(PDF_DEFLATE_LEVEL));
    ensure!(
        result == ReturnCode::Ok,
        "{PDF_DEFLATE_CODEC} compression failed with {result:?}"
    );
    let compressed_len = compressed.len();
    output.truncate(compressed_len);
    Ok(output)
}

struct StreamingPdf {
    writer: BufWriter<File>,
    offsets: Vec<u64>,
}

impl StreamingPdf {
    fn create(output: &Path) -> Result<Self> {
        let file =
            File::create(output).with_context(|| format!("cannot create {}", output.display()))?;
        let mut writer = BufWriter::new(file);
        writer.write_all(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n")?;
        Ok(Self {
            writer,
            offsets: Vec::new(),
        })
    }

    fn begin_object(&mut self) -> Result<usize> {
        let id = self.offsets.len() + 1;
        self.offsets.push(self.writer.stream_position()?);
        writeln!(self.writer, "{id} 0 obj")?;
        Ok(id)
    }

    fn write_object(&mut self, object: &[u8]) -> Result<usize> {
        let id = self.begin_object()?;
        self.writer.write_all(object)?;
        self.writer.write_all(b"\nendobj\n")?;
        Ok(id)
    }

    fn write_stream(&mut self, dictionary: &str, data: &[u8]) -> Result<usize> {
        let id = self.begin_object()?;
        writeln!(
            self.writer,
            "<< {dictionary} /Length {} >>\nstream",
            data.len()
        )?;
        self.writer.write_all(data)?;
        self.writer.write_all(b"\nendstream\nendobj\n")?;
        Ok(id)
    }

    fn finish(mut self) -> Result<()> {
        let xref = self.writer.stream_position()?;
        writeln!(
            self.writer,
            "xref\n0 {}\n0000000000 65535 f ",
            self.offsets.len() + 1
        )?;
        for offset in &self.offsets {
            writeln!(self.writer, "{offset:010} 00000 n ")?;
        }
        write!(
            self.writer,
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            self.offsets.len() + 1
        )?;
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(())
    }
}

#[cfg(test)]
fn assemble_pdf(objects: &[Vec<u8>]) -> Vec<u8> {
    let mut pdf = b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n".to_vec();
    let mut offsets = Vec::with_capacity(objects.len());
    for (index, object) in objects.iter().enumerate() {
        offsets.push(pdf.len());
        pdf.extend_from_slice(format!("{} 0 obj\n", index + 1).as_bytes());
        pdf.extend_from_slice(object);
        pdf.extend_from_slice(b"\nendobj\n");
    }
    let xref = pdf.len();
    pdf.extend_from_slice(
        format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes(),
    );
    for offset in offsets {
        pdf.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objects.len() + 1
        )
        .as_bytes(),
    );
    pdf
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf};

    use crate::model::{AnalyzerReport, BuildTimings, Callsite, Node};

    use super::*;

    #[test]
    fn pdf_builder_emits_xref_and_catalog() {
        let pdf = assemble_pdf(&[b"<< /Type /Catalog >>".to_vec()]);
        assert!(pdf.starts_with(b"%PDF-1.7"));
        assert!(pdf.windows(4).any(|window| window == b"xref"));
        assert!(pdf.ends_with(b"%%EOF\n"));
    }

    #[test]
    fn xml_and_pdf_text_are_escaped() {
        assert_eq!(xml_escape("a&<b"), "a&amp;&lt;b");
        assert_eq!(pdf_escape("a(b)\\c"), "a\\(b\\)\\\\c");
    }

    #[test]
    fn pdf_call_dpi_must_not_reduce_the_page_resolution() {
        let options = RenderOptions {
            pdf_dpi: 300.0,
            pdf_call_dpi: Some(299.0),
            ..RenderOptions::default()
        };

        assert!(effective_pdf_call_dpi(&options).is_err());
    }

    #[test]
    fn pdf_compression_worker_count_is_bounded_when_automatic() {
        assert!((1..=4).contains(&pdf_compression_worker_count(0)));
        assert_eq!(pdf_compression_worker_count(1), 1);
        assert_eq!(pdf_compression_worker_count(3), 3);
    }

    #[test]
    fn pdf_deflate_is_deterministic_and_lossless() {
        let input = b"architectural graphite route ".repeat(4_096);
        let first = deflate(&input).unwrap();
        let second = deflate(&input).unwrap();
        assert_eq!(first, second);

        let mut decoded = vec![0; input.len()];
        let (decoded, result) =
            zlib_rs::decompress_slice(&mut decoded, &first, zlib_rs::InflateConfig::default());
        assert_eq!(result, ReturnCode::Ok);
        assert_eq!(decoded, input);
    }

    #[test]
    fn interrupted_pdf_resume_reuses_verified_tiles_and_matches_fresh_call_pixels() {
        let directory = tempfile::tempdir().unwrap();
        let interrupted_output = directory.path().join("interrupted.pdf");
        let fresh_output = directory.path().join("fresh.pdf");
        let (atlas, layout) = resume_test_fixture();
        let interrupted = RenderOptions {
            pdf_dpi: 144.0,
            pdf_call_dpi: Some(288.0),
            pdf_call_tile_size: 64,
            pdf_resume: true,
            pdf_restart: true,
            pdf_stop_after_tiles: Some(1),
            pdf_compression_workers: 3,
            ..RenderOptions::default()
        };
        let error =
            render_vector_artifact(&atlas, &interrupted_output, &layout, &interrupted).unwrap_err();
        assert!(error.to_string().contains("stopped after 1"));
        assert!(!interrupted_output.exists());

        let manifest_path = pdf_tile_cache_dir(&interrupted_output, true).join("manifest.json");
        let partial: TileManifest =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        assert_eq!(partial.completed_tiles, 1);
        assert_eq!(partial.configuration["compression"]["codec"], "zlib-rs");
        assert_eq!(partial.configuration["compression"]["level"], 6);

        let resumed = RenderOptions {
            pdf_stop_after_tiles: None,
            pdf_restart: false,
            ..interrupted.clone()
        };
        let resumed_stats =
            render_vector_artifact(&atlas, &interrupted_output, &layout, &resumed).unwrap();
        let resumed_cache = resumed_stats.pdf_tile_cache.unwrap();
        assert_eq!(resumed_cache.tiles_reused, 1);
        assert!(resumed_cache.tiles_rendered > 0);
        assert_eq!(resumed_cache.compression_workers, 3);
        assert_eq!(resumed_cache.max_in_flight_tiles, 3);

        let fresh = RenderOptions {
            pdf_stop_after_tiles: None,
            pdf_compression_workers: 1,
            ..interrupted
        };
        let fresh_stats = render_vector_artifact(&atlas, &fresh_output, &layout, &fresh).unwrap();
        assert_eq!(fresh_stats.calls_drawn, atlas.calls.len());
        let resumed_manifest: TileManifest =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        let fresh_manifest_path = pdf_tile_cache_dir(&fresh_output, true).join("manifest.json");
        let fresh_manifest: TileManifest =
            serde_json::from_slice(&fs::read(fresh_manifest_path).unwrap()).unwrap();
        let checksums = |manifest: &TileManifest| {
            manifest
                .tiles
                .iter()
                .map(|tile| tile.as_ref().unwrap().sha256.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(checksums(&resumed_manifest), checksums(&fresh_manifest));
        assert_eq!(
            resumed_manifest.completed_tiles,
            resumed_manifest.tile_count
        );

        let first = resumed_manifest.tiles[0].as_ref().unwrap();
        fs::write(
            pdf_tile_cache_dir(&interrupted_output, true).join(&first.artifact),
            b"corrupt",
        )
        .unwrap();
        let repaired =
            render_vector_artifact(&atlas, &interrupted_output, &layout, &resumed).unwrap();
        let repaired_cache = repaired.pdf_tile_cache.unwrap();
        assert_eq!(repaired_cache.tiles_rendered, 1);
        assert_eq!(
            repaired_cache.tiles_reused + repaired_cache.tiles_rendered,
            resumed_manifest.tile_count
        );

        let incompatible = RenderOptions {
            call_width: resumed.call_width + 0.25,
            ..resumed
        };
        let error = render_vector_artifact(&atlas, &interrupted_output, &layout, &incompatible)
            .unwrap_err();
        assert!(error.to_string().contains("restart with --restart"));
    }

    fn resume_test_fixture() -> (Atlas, LayoutOptions) {
        let layout = LayoutOptions {
            width: 160,
            height: 120,
            ..LayoutOptions::default()
        };
        let file = |id, path: &str, rect| Node {
            id,
            parent: Some(0),
            children: Vec::new(),
            kind: NodeKind::File,
            name: path.to_owned(),
            path: path.to_owned(),
            depth: 1,
            bytes: 20,
            loc: 20,
            commits: 1,
            weight: 20.0,
            language: "rust".to_owned(),
            rect,
        };
        let atlas = Atlas {
            root_path: PathBuf::from("fixture"),
            revision: "resume-test".to_owned(),
            dirty: false,
            excluded_test_files: 0,
            excluded_hidden_files: 0,
            excluded_custom_files: 0,
            excluded_paths: Vec::new(),
            nodes: vec![
                Node {
                    id: 0,
                    parent: None,
                    children: vec![1, 2],
                    kind: NodeKind::Directory,
                    name: "fixture".to_owned(),
                    path: String::new(),
                    depth: 0,
                    bytes: 40,
                    loc: 40,
                    commits: 2,
                    weight: 40.0,
                    language: "directory".to_owned(),
                    rect: Rect {
                        x0: 0.0,
                        y0: 0.0,
                        x1: 160.0,
                        y1: 120.0,
                    },
                },
                file(
                    1,
                    "a.rs",
                    Rect {
                        x0: 4.0,
                        y0: 4.0,
                        x1: 70.0,
                        y1: 116.0,
                    },
                ),
                file(
                    2,
                    "b.rs",
                    Rect {
                        x0: 90.0,
                        y0: 4.0,
                        x1: 156.0,
                        y1: 116.0,
                    },
                ),
            ],
            calls: vec![
                Callsite {
                    id: 1,
                    source: 1,
                    source_line: 2,
                    target: 2,
                    target_line: Some(19),
                    callee: "b::low".to_owned(),
                    kind: "runtime".to_owned(),
                    analyzer: "fixture".to_owned(),
                    confidence: 1.0,
                },
                Callsite {
                    id: 2,
                    source: 2,
                    source_line: 3,
                    target: 1,
                    target_line: Some(18),
                    callee: "a::high".to_owned(),
                    kind: "runtime".to_owned(),
                    analyzer: "fixture".to_owned(),
                    confidence: 1.0,
                },
            ],
            path_to_id: HashMap::new(),
            report: AnalyzerReport::default(),
            timings: BuildTimings::default(),
        };
        (atlas, layout)
    }

    #[test]
    fn overlapping_call_tiles_match_the_monolithic_pixels_at_every_core_seam() {
        let layout = LayoutOptions {
            width: 160,
            height: 120,
            ..LayoutOptions::default()
        };
        let file = |id, path: &str, rect| Node {
            id,
            parent: Some(0),
            children: Vec::new(),
            kind: NodeKind::File,
            name: path.to_owned(),
            path: path.to_owned(),
            depth: 1,
            bytes: 20,
            loc: 20,
            commits: 1,
            weight: 20.0,
            language: "rust".to_owned(),
            rect,
        };
        let atlas = Atlas {
            root_path: PathBuf::from("fixture"),
            revision: "test".to_owned(),
            dirty: false,
            excluded_test_files: 0,
            excluded_hidden_files: 0,
            excluded_custom_files: 0,
            excluded_paths: Vec::new(),
            nodes: vec![
                Node {
                    id: 0,
                    parent: None,
                    children: vec![1, 2],
                    kind: NodeKind::Directory,
                    name: "fixture".to_owned(),
                    path: String::new(),
                    depth: 0,
                    bytes: 40,
                    loc: 40,
                    commits: 2,
                    weight: 40.0,
                    language: "directory".to_owned(),
                    rect: Rect {
                        x0: 0.0,
                        y0: 0.0,
                        x1: 160.0,
                        y1: 120.0,
                    },
                },
                file(
                    1,
                    "a.rs",
                    Rect {
                        x0: 4.0,
                        y0: 4.0,
                        x1: 70.0,
                        y1: 116.0,
                    },
                ),
                file(
                    2,
                    "b.rs",
                    Rect {
                        x0: 90.0,
                        y0: 4.0,
                        x1: 156.0,
                        y1: 116.0,
                    },
                ),
            ],
            calls: vec![
                Callsite {
                    id: 1,
                    source: 1,
                    source_line: 2,
                    target: 2,
                    target_line: Some(19),
                    callee: "b::low".to_owned(),
                    kind: "runtime".to_owned(),
                    analyzer: "fixture".to_owned(),
                    confidence: 1.0,
                },
                Callsite {
                    id: 2,
                    source: 2,
                    source_line: 3,
                    target: 1,
                    target_line: Some(18),
                    callee: "a::high".to_owned(),
                    kind: "runtime".to_owned(),
                    analyzer: "fixture".to_owned(),
                    confidence: 1.0,
                },
            ],
            path_to_id: HashMap::new(),
            report: AnalyzerReport::default(),
            timings: BuildTimings::default(),
        };
        assert_tiled_call_layer_matches(&atlas, &layout, Theme::Architect);
        assert_tiled_call_layer_matches(&atlas, &layout, Theme::Night);
    }

    fn assert_tiled_call_layer_matches(atlas: &Atlas, layout: &LayoutOptions, theme: Theme) {
        let render = RenderOptions {
            theme,
            pdf_call_tile_size: 64,
            ..RenderOptions::default()
        };
        let indices = [0, 1];
        let (full, full_calls) = render_software_call_tile(
            atlas,
            layout,
            &render,
            &indices,
            CallTileViewport {
                x: 0,
                y: 0,
                width: layout.width,
                height: layout.height,
            },
        )
        .unwrap();
        let (tiles, planned_calls, overlap) = plan_call_tiles(atlas, layout, &render).unwrap();
        assert_eq!(full_calls, 2);
        assert_eq!(planned_calls, 2);
        assert!(overlap >= 4);
        assert!(tiles.len() > 1);

        let mut covered = vec![false; layout.width as usize * layout.height as usize];
        for tile in &tiles {
            let (pixmap, _) = render_software_call_tile(
                atlas,
                layout,
                &render,
                &tile.call_indices,
                CallTileViewport {
                    x: tile.render_x,
                    y: tile.render_y,
                    width: tile.render_width,
                    height: tile.render_height,
                },
            )
            .unwrap();
            for y in tile.core_y..tile.core_y + tile.core_height {
                for x in tile.core_x..tile.core_x + tile.core_width {
                    let full_offset = (y as usize * layout.width as usize + x as usize) * 4;
                    let local_x = x - tile.render_x;
                    let local_y = y - tile.render_y;
                    let tile_offset =
                        (local_y as usize * tile.render_width as usize + local_x as usize) * 4;
                    assert_eq!(
                        &full.data()[full_offset..full_offset + 4],
                        &pixmap.data()[tile_offset..tile_offset + 4],
                        "tile core differs from monolithic render at ({x}, {y})"
                    );
                    covered[y as usize * layout.width as usize + x as usize] = true;
                }
            }
        }
        for (index, pixel) in full.data().chunks_exact(4).enumerate() {
            assert!(covered[index] || pixel[3] == 0);
        }
    }
}
