use std::{fs, path::Path, time::Instant};

use anyhow::{Context, Result, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::Serialize;

use crate::{
    layout::{LayoutOptions, call_curve},
    model::{Atlas, Node, NodeKind, Rect},
    render::{
        LabelPlacement, Palette, RenderOptions, RenderStats, Rgba, collect_label_commands,
        optical_density_description,
    },
};

const VIEWER_TEMPLATE: &str = include_str!("viewer.html");
const VIEWER_WEBGPU: &str = include_str!("viewer_webgpu.js");
const CURVE_QUANTIZATION_MAX: f64 = u16::MAX as f64;
const INTERACTIVE_MAX_CALL_OPACITY: f64 = 160.0;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ViewerData {
    version: u32,
    width: u32,
    height: u32,
    title: String,
    revision: String,
    dirty: bool,
    initial_theme: &'static str,
    source_url_template: Option<String>,
    texture_seed: u64,
    call_opacity: u8,
    effective_call_opacity: u8,
    density_aware_exposure: bool,
    density_reference_calls: usize,
    call_width: f32,
    themes: ViewerThemes,
    files: Vec<ViewerFile>,
    directories: Vec<ViewerDirectory>,
    labels: Vec<ViewerLabel>,
    calls: Vec<ViewerCall>,
    curve_points_base64: String,
}

#[derive(Serialize)]
struct ViewerThemes {
    light: ViewerTheme,
    dark: ViewerTheme,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ViewerTheme {
    palette: ViewerPalette,
    file_fills: Vec<String>,
    label_colors: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ViewerPalette {
    background: String,
    land: String,
    paper_fiber: String,
    outer_border: String,
    directory_border: String,
    file_border: String,
    label: String,
    direction_source: String,
    direction_target: String,
}

#[derive(Serialize)]
struct ViewerFile {
    id: usize,
    path: String,
    name: String,
    language: String,
    depth: usize,
    loc: u64,
    bytes: u64,
    commits: u64,
    rect: [f64; 4],
    incoming: usize,
    outgoing: usize,
}

#[derive(Serialize)]
struct ViewerDirectory {
    id: usize,
    name: String,
    path: String,
    depth: usize,
    rect: [f64; 4],
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ViewerLabel {
    text: String,
    size: f32,
    placement: &'static str,
    x: f32,
    y: f32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ViewerCall {
    id: u64,
    source: usize,
    target: usize,
    source_line: u32,
    target_line: Option<u32>,
    callee: String,
    analyzer: String,
    confidence: f32,
    point_start: usize,
    point_count: usize,
    bounds: [u16; 4],
}

pub(crate) fn render_interactive_html(
    atlas: &Atlas,
    output: &Path,
    layout: &LayoutOptions,
    render: &RenderOptions,
) -> Result<RenderStats> {
    let started = Instant::now();
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("cannot create output directory {}", parent.display()))?;
    }

    let light_palette = Palette::for_theme(crate::render::Theme::Architect);
    let dark_palette = Palette::for_theme(crate::render::Theme::Night);
    let source_url_template = validate_source_url_template(render.source_url_template.as_deref())?;
    let (light_label_commands, label_stats) = collect_label_commands(atlas, layout, light_palette);
    let (dark_label_commands, dark_label_stats) =
        collect_label_commands(atlas, layout, dark_palette);
    ensure!(
        light_label_commands.len() == dark_label_commands.len()
            && label_stats.files == dark_label_stats.files
            && label_stats.directories == dark_label_stats.directories,
        "interactive light and dark label layouts do not match"
    );
    let light_label_colors = light_label_commands
        .iter()
        .map(|command| rgba_css(command.color))
        .collect();
    let dark_label_colors = dark_label_commands
        .iter()
        .map(|command| rgba_css(command.color))
        .collect();
    let labels = light_label_commands
        .iter()
        .map(|command| {
            let (placement, x, y) = match command.placement {
                LabelPlacement::Horizontal { x, baseline } => ("horizontal", x, baseline),
                LabelPlacement::Vertical { x, y } => ("vertical", x, y),
                LabelPlacement::Signature { x, baseline } => ("signature", x, baseline),
            };
            ViewerLabel {
                text: command.text.clone(),
                size: command.size,
                placement,
                x,
                y,
            }
        })
        .collect();

    let mut incoming = vec![0_usize; atlas.nodes.len()];
    let mut outgoing = vec![0_usize; atlas.nodes.len()];
    for call in &atlas.calls {
        outgoing[call.source] += 1;
        incoming[call.target] += 1;
    }

    let files = atlas
        .nodes
        .iter()
        .filter(|node| node.kind == NodeKind::File)
        .map(|node| viewer_file(node, incoming[node.id], outgoing[node.id]))
        .collect();
    let light_file_fills = atlas
        .nodes
        .iter()
        .filter(|node| node.kind == NodeKind::File)
        .map(|node| rgba_css(light_palette.file_color(&node.language)))
        .collect();
    let dark_file_fills = atlas
        .nodes
        .iter()
        .filter(|node| node.kind == NodeKind::File)
        .map(|node| rgba_css(dark_palette.file_color(&node.language)))
        .collect();
    let directories = atlas
        .nodes
        .iter()
        .filter(|node| node.kind == NodeKind::Directory)
        .map(|node| ViewerDirectory {
            id: node.id,
            name: node.name.clone(),
            path: node.path.clone(),
            depth: node.depth,
            rect: rect_array(node.rect),
        })
        .collect();

    let mut point_bytes = Vec::new();
    let mut calls = Vec::with_capacity(atlas.calls.len());
    for call in &atlas.calls {
        let curve = call_curve(atlas, call, layout);
        ensure!(
            curve.len() >= 2,
            "interactive call {} has fewer than two curve points",
            call.id
        );
        let point_start = point_bytes.len() / 4;
        let mut min_x = u16::MAX;
        let mut min_y = u16::MAX;
        let mut max_x = 0_u16;
        let mut max_y = 0_u16;
        for point in &curve {
            let x = quantize(point.x, f64::from(layout.width));
            let y = quantize(point.y, f64::from(layout.height));
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            point_bytes.extend_from_slice(&x.to_le_bytes());
            point_bytes.extend_from_slice(&y.to_le_bytes());
        }
        calls.push(ViewerCall {
            id: call.id,
            source: call.source,
            target: call.target,
            source_line: call.source_line,
            target_line: call.target_line,
            callee: call.callee.clone(),
            analyzer: call.analyzer.clone(),
            confidence: call.confidence,
            point_start,
            point_count: curve.len(),
            bounds: [min_x, min_y, max_x, max_y],
        });
    }

    let data = ViewerData {
        version: 3,
        width: layout.width,
        height: layout.height,
        title: viewer_title(&atlas.nodes[0].name),
        revision: atlas.revision.clone(),
        dirty: atlas.dirty,
        initial_theme: if matches!(
            render.theme,
            crate::render::Theme::Night | crate::render::Theme::SolarizedDark
        ) {
            "dark"
        } else {
            "light"
        },
        source_url_template,
        texture_seed: layout.texture_seed,
        call_opacity: render.call_opacity,
        effective_call_opacity: interactive_call_opacity(atlas.calls.len(), render),
        density_aware_exposure: render.density_aware_exposure,
        density_reference_calls: render.density_reference_calls,
        call_width: render.call_width,
        themes: ViewerThemes {
            light: ViewerTheme {
                palette: viewer_palette(light_palette),
                file_fills: light_file_fills,
                label_colors: light_label_colors,
            },
            dark: ViewerTheme {
                palette: viewer_palette(dark_palette),
                file_fills: dark_file_fills,
                label_colors: dark_label_colors,
            },
        },
        files,
        directories,
        labels,
        calls,
        curve_points_base64: BASE64.encode(point_bytes),
    };
    let json = script_safe_json(&data)?;
    let html = VIEWER_TEMPLATE
        .replace("__CODE_ATLAS_WEBGPU__", VIEWER_WEBGPU)
        .replace("__CODE_ATLAS_DATA__", &json);
    fs::write(output, html.as_bytes())
        .with_context(|| format!("cannot save {}", output.display()))?;
    let output_bytes = fs::metadata(output)?.len();

    Ok(RenderStats {
        backend: "webgpu optical-density interactive html with canvas2d fallback".to_owned(),
        gpu_adapter: None,
        width: layout.width,
        height: layout.height,
        call_layer_width: layout.width,
        call_layer_height: layout.height,
        call_layer_dpi: None,
        call_layer_tiles: 1,
        call_layer_tile_size: None,
        call_layer_overlap: None,
        files_drawn: atlas.files().count(),
        file_labels_drawn: label_stats.files,
        directory_labels_drawn: label_stats.directories,
        calls_drawn: atlas.calls.len(),
        directory_boundaries_drawn: atlas
            .nodes
            .iter()
            .filter(|node| node.kind == NodeKind::Directory && node.id != 0)
            .count(),
        output_bytes,
        effective_call_opacity: interactive_call_opacity(atlas.calls.len(), render),
        call_compositing: format!(
            "{} in WebGPU-capable browsers; the compatibility fallback uses independent Canvas2D source-over strokes",
            optical_density_description()
        ),
        direction_source_color: light_palette.direction_source.hex(),
        direction_target_color: light_palette.direction_target.hex(),
        render_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        pdf_tile_cache: None,
    })
}

fn interactive_call_opacity(call_count: usize, options: &RenderOptions) -> u8 {
    if call_count == 0 || options.call_opacity == 0 {
        return 0;
    }
    if !options.density_aware_exposure {
        return options.call_opacity;
    }
    let reference_calls = options.density_reference_calls.max(1) as f64;
    let exposure_scale = (reference_calls / call_count as f64).sqrt();
    (f64::from(options.call_opacity) * exposure_scale)
        .round()
        .clamp(2.0, INTERACTIVE_MAX_CALL_OPACITY) as u8
}

fn viewer_file(node: &Node, incoming: usize, outgoing: usize) -> ViewerFile {
    ViewerFile {
        id: node.id,
        path: node.path.clone(),
        name: node.name.clone(),
        language: node.language.clone(),
        depth: node.depth,
        loc: node.loc,
        bytes: node.bytes,
        commits: node.commits,
        rect: rect_array(node.rect),
        incoming,
        outgoing,
    }
}

fn viewer_palette(palette: Palette) -> ViewerPalette {
    ViewerPalette {
        background: rgba_css(palette.background),
        land: rgba_css(palette.land),
        paper_fiber: rgba_css(palette.paper_fiber),
        outer_border: rgba_css(palette.outer_border),
        directory_border: rgba_css(palette.directory_border),
        file_border: rgba_css(palette.file_border),
        label: rgba_css(palette.label),
        direction_source: rgba_css(palette.direction_source),
        direction_target: rgba_css(palette.direction_target),
    }
}

fn rect_array(rect: Rect) -> [f64; 4] {
    [rect.x0, rect.y0, rect.x1, rect.y1]
}

fn quantize(value: f64, extent: f64) -> u16 {
    if !value.is_finite() || extent <= 0.0 {
        return 0;
    }
    (value.clamp(0.0, extent) / extent * CURVE_QUANTIZATION_MAX).round() as u16
}

fn rgba_css(color: Rgba) -> String {
    format!(
        "rgba({},{},{},{:.4})",
        color.r,
        color.g,
        color.b,
        f64::from(color.a) / 255.0
    )
}

fn viewer_title(repository_name: &str) -> String {
    let lower = repository_name.to_ascii_lowercase();
    for suffix in ["-code-atlas", "_code_atlas", " code atlas"] {
        if lower.ends_with(suffix) {
            let title = repository_name[..repository_name.len() - suffix.len()].trim();
            if !title.is_empty() {
                return title.to_owned();
            }
        }
    }
    repository_name.to_owned()
}

fn script_safe_json(value: &impl Serialize) -> Result<String> {
    Ok(serde_json::to_string(value)?
        .replace('&', "\\u0026")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e"))
}

fn validate_source_url_template(template: Option<&str>) -> Result<Option<String>> {
    let Some(template) = template else {
        return Ok(None);
    };
    let template = template.trim();
    ensure!(
        template.starts_with("https://") || template.starts_with("http://"),
        "source URL template must begin with http:// or https://"
    );
    for placeholder in ["{revision}", "{path}", "{line}"] {
        ensure!(
            template.contains(placeholder),
            "source URL template must contain {placeholder}"
        );
    }
    let remainder = template
        .replace("{revision}", "")
        .replace("{path}", "")
        .replace("{line}", "");
    ensure!(
        !remainder.contains('{') && !remainder.contains('}'),
        "source URL template contains an unsupported placeholder"
    );
    Ok(Some(template.to_owned()))
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf};

    use crate::model::{AnalyzerReport, BuildTimings, Callsite, Node};

    use super::*;

    #[test]
    fn curve_quantization_clamps_and_preserves_endpoints() {
        assert_eq!(quantize(-1.0, 100.0), 0);
        assert_eq!(quantize(0.0, 100.0), 0);
        assert_eq!(quantize(50.0, 100.0), 32_768);
        assert_eq!(quantize(100.0, 100.0), u16::MAX);
        assert_eq!(quantize(101.0, 100.0), u16::MAX);
    }

    #[test]
    fn embedded_json_cannot_close_its_script_element() {
        #[derive(Serialize)]
        struct Payload<'a> {
            value: &'a str,
        }
        let json = script_safe_json(&Payload {
            value: "</script><script>alert(1)</script>",
        })
        .unwrap();
        assert!(!json.contains('<'));
        assert!(!json.contains('>'));
        assert!(!json.contains("</script>"));
    }

    #[test]
    fn viewer_title_removes_the_local_code_atlas_checkout_suffix() {
        assert_eq!(viewer_title("elixir-code-atlas"), "elixir");
        assert_eq!(viewer_title("Elixir_Code_Atlas"), "Elixir");
        assert_eq!(viewer_title("Elixir Code Atlas"), "Elixir");
        assert_eq!(viewer_title("wasmex"), "wasmex");
        assert_eq!(viewer_title("code-atlas"), "code-atlas");
    }

    #[test]
    fn source_url_templates_are_explicit_complete_and_http_only() {
        let template = "https://example.test/repo/blob/{revision}/{path}#L{line}";
        assert_eq!(
            validate_source_url_template(Some(template)).unwrap(),
            Some(template.to_owned())
        );
        assert!(validate_source_url_template(None).unwrap().is_none());
        assert!(validate_source_url_template(Some("javascript:{path}")).is_err());
        assert!(
            validate_source_url_template(Some("https://example.test/repo/blob/{revision}/{path}"))
                .is_err()
        );
        assert!(
            validate_source_url_template(Some(
                "https://example.test/{revision}/{path}#L{line}/{unknown}"
            ))
            .is_err()
        );
    }

    #[test]
    fn html_export_is_self_contained_and_keeps_every_callsite() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("atlas.html");
        let layout = LayoutOptions {
            width: 320,
            height: 180,
            ..LayoutOptions::default()
        };
        let mut atlas = interactive_fixture();
        let stats = render_interactive_html(
            &atlas,
            &output,
            &layout,
            &RenderOptions {
                theme: crate::render::Theme::Night,
                ..RenderOptions::default()
            },
        )
        .unwrap();
        let html = fs::read_to_string(output).unwrap();
        assert_eq!(stats.calls_drawn, atlas.calls.len());
        assert_eq!(
            stats.backend,
            "webgpu optical-density interactive html with canvas2d fallback"
        );
        assert!(html.starts_with("<!doctype html>"));
        assert!(html.contains("window.__codeAtlas"));
        assert!(html.contains("function hitTestCall(point)"));
        assert!(html.contains("async function createWebGpuCallRenderer"));
        assert!(html.contains("interactive optical-density accumulation"));
        assert!(html.contains("format: 'rgba16float'"));
        assert!(html.contains("1.0 - exp(-deposited.a)"));
        assert!(html.contains("canvas2d-fallback"));
        assert!(html.contains("id=\"atlas-call-canvas\""));
        assert!(!html.contains("__CODE_ATLAS_WEBGPU__"));
        assert!(html.contains("id=\"atlas-selected-call\""));
        assert!(html.contains("id=\"atlas-copy-link\""));
        assert!(html.contains("id=\"atlas-call-query\""));
        assert!(html.contains("id=\"atlas-query-help\""));
        assert!(html.contains("id=\"atlas-layers\""));
        assert!(html.contains("id=\"atlas-theme-mode\""));
        assert!(html.contains("document.title = data.title;"));
        assert!(html.contains("code-atlas.theme.preference.v1"));
        assert!(html.contains("(prefers-color-scheme: dark)"));
        assert!(html.contains("\"version\":3"));
        assert!(html.contains("\"initialTheme\":\"dark\""));
        assert!(html.contains("\"callOpacity\":34"));
        assert!(html.contains("\"densityAwareExposure\":true"));
        assert!(html.contains("\"densityReferenceCalls\":2500"));
        assert!(html.contains("function adaptiveCallOpacity(callCount)"));
        assert!(html.contains("Math.sqrt(referenceCalls / activeCalls)"));
        assert!(html.contains("Math.min(160, baseOpacity * exposureScale)"));
        assert!(html.contains("const alpha = currentCallAlpha(drawnCalls)"));
        assert!(html.contains("currentCallAlpha(visibleCalls.length)"));
        assert!(html.contains("canvas.dataset.visibleCalls"));
        assert!(html.contains("\"themes\":{\"light\""));
        assert!(html.contains("rgba(241,238,229,1.0000)"));
        assert!(html.contains("rgba(6,10,17,1.0000)"));
        assert!(html.contains("id=\"atlas-active-query\""));
        assert!(html.contains("class=\"atlas-map-zoom\""));
        assert!(html.contains("Filter individual calls"));
        assert!(html.contains("Unknown filter “${requestedField}:”"));
        assert!(html.contains("function compileCallQuery(input)"));
        assert!(html.contains("function globMatches(value, pattern)"));
        assert!(html.contains("queryMask: new Uint8Array(data.calls.length).fill(1)"));
        assert!(html.contains("new URLSearchParams(window.location.hash.slice(1))"));
        assert!(html.contains("\"sourceUrlTemplate\":null"));
        assert!(html.contains("dangerous\\u003c/script\\u003e"));
        assert!(!html.contains("https://"));
        assert!(!html.contains("fetch("));
        assert!(!html.contains("XMLHttpRequest"));
        assert!(!html.contains("WebSocket"));

        // The generated page is independent of later atlas mutations.
        atlas.calls.clear();
        assert!(html.contains("curvePointsBase64"));
    }

    #[test]
    fn interactive_exposure_strengthens_sparse_filters_and_restrains_dense_ones() {
        let options = RenderOptions::default();
        assert_eq!(interactive_call_opacity(0, &options), 0);
        assert_eq!(interactive_call_opacity(294, &options), 99);
        assert_eq!(interactive_call_opacity(100, &options), 160);
        assert_eq!(interactive_call_opacity(1_317, &options), 47);
        assert_eq!(interactive_call_opacity(2_500, &options), 34);
        assert_eq!(interactive_call_opacity(20_202, &options), 12);

        let fixed = RenderOptions {
            density_aware_exposure: false,
            ..options
        };
        assert_eq!(interactive_call_opacity(294, &fixed), 34);
        assert_eq!(interactive_call_opacity(20_202, &fixed), 34);
    }

    #[test]
    fn configured_source_link_template_is_embedded() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("linked.html");
        let template = "https://example.test/repo/blob/{revision}/{path}#L{line}";
        render_interactive_html(
            &interactive_fixture(),
            &output,
            &LayoutOptions {
                width: 320,
                height: 180,
                ..LayoutOptions::default()
            },
            &RenderOptions {
                source_url_template: Some(template.to_owned()),
                ..RenderOptions::default()
            },
        )
        .unwrap();
        let html = fs::read_to_string(output).unwrap();
        assert!(html.contains(&format!("\"sourceUrlTemplate\":\"{template}\"")));
    }

    fn interactive_fixture() -> Atlas {
        let root = Node {
            id: 0,
            parent: None,
            children: vec![1, 2],
            kind: NodeKind::Directory,
            name: "fixture".to_owned(),
            path: String::new(),
            depth: 0,
            bytes: 200,
            loc: 20,
            commits: 2,
            weight: 20.0,
            language: "directory".to_owned(),
            rect: Rect {
                x0: 12.0,
                y0: 12.0,
                x1: 308.0,
                y1: 168.0,
            },
        };
        let file = |id, name: &str, rect| Node {
            id,
            parent: Some(0),
            children: Vec::new(),
            kind: NodeKind::File,
            name: name.to_owned(),
            path: name.to_owned(),
            depth: 1,
            bytes: 100,
            loc: 10,
            commits: 1,
            weight: 10.0,
            language: "rust".to_owned(),
            rect,
        };
        Atlas {
            root_path: PathBuf::from("fixture"),
            revision: "interactive-test".to_owned(),
            dirty: false,
            excluded_test_files: 0,
            excluded_hidden_files: 0,
            excluded_custom_files: 0,
            excluded_paths: Vec::new(),
            nodes: vec![
                root,
                file(
                    1,
                    "source.rs",
                    Rect {
                        x0: 14.0,
                        y0: 14.0,
                        x1: 150.0,
                        y1: 166.0,
                    },
                ),
                file(
                    2,
                    "target.rs",
                    Rect {
                        x0: 154.0,
                        y0: 14.0,
                        x1: 306.0,
                        y1: 166.0,
                    },
                ),
            ],
            calls: vec![Callsite {
                id: 7,
                source: 1,
                source_line: 2,
                target: 2,
                target_line: Some(8),
                callee: "dangerous</script>".to_owned(),
                kind: "runtime".to_owned(),
                analyzer: "fixture".to_owned(),
                confidence: 1.0,
            }],
            path_to_id: HashMap::new(),
            report: AnalyzerReport::default(),
            timings: BuildTimings::default(),
        }
    }
}
