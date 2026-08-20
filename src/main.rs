use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use code_atlas::{
    BuildOptions, build_atlas,
    layout::LayoutOptions,
    model::{CallPathFilters, Metric},
    render::{RenderBackend, RenderOptions, Theme},
    render_atlas,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(name = "code-atlas", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Analyze a checkout and render its architectural repository atlas.
    Render(RenderArgs),
    /// Analyze a checkout and print the normalized summary as JSON.
    Inspect(InspectArgs),
}

#[derive(Debug, clap::Args)]
struct RenderArgs {
    /// Local Git checkout to visualize.
    #[arg(long, short = 'r')]
    repo: PathBuf,

    /// Output PNG, SVG, PDF, or self-contained interactive HTML path.
    #[arg(long, short = 'o', default_value = "code-atlas.png")]
    output: PathBuf,

    #[arg(long, default_value_t = 1920)]
    width: u32,

    #[arg(long, default_value_t = 1080)]
    height: u32,

    /// File area metric: loc, bytes, or commits.
    #[arg(long, default_value = "loc")]
    metric: String,

    /// Theme: architect, night, ink, solarized-dark, or solarized-light.
    #[arg(long, default_value = "architect")]
    theme: String,

    /// Rendering backend: software or wgpu. Wgpu accelerates the call layer.
    #[arg(long, default_value = "software")]
    backend: String,

    /// HTTP(S) source-link template for interactive HTML. Must contain
    /// {revision}, {path}, and {line}.
    #[arg(long, value_name = "TEMPLATE")]
    source_url_template: Option<String>,

    /// Empty drafting-paper margin as a fraction of the shorter frame side.
    #[arg(long, default_value_t = 0.045)]
    margin_fraction: f64,

    /// Skip semantic callsite analysis and render only the repository land.
    #[arg(long)]
    no_calls: bool,

    /// Include conventional test directories and test files.
    #[arg(long)]
    include_tests: bool,

    /// Include files and directories whose names begin with a dot.
    #[arg(long)]
    include_hidden: bool,

    /// Exclude a repository-relative file or directory. Repeat as needed.
    #[arg(long = "exclude", value_name = "PATH")]
    excluded_paths: Vec<String>,

    #[command(flatten)]
    call_filters: CallFilterArgs,

    /// Deterministic seed for paper grain and pencil stroke texture.
    #[arg(long, default_value_t = 932)]
    texture_seed: u64,

    /// Hierarchy attraction in the range 0 (straight) to 1 (fully bundled).
    #[arg(long, default_value_t = 0.96)]
    bundle_strength: f64,

    #[arg(long, default_value_t = 34)]
    call_opacity: u8,

    /// Disable automatic exposure adjustment, optionally using this opacity (0-255).
    #[arg(long, value_name = "OPACITY", num_args = 0..=1)]
    fixed_call_opacity: Option<Option<u8>>,

    /// Pencil stroke width in 1080p design pixels.
    #[arg(long, default_value_t = 1.10)]
    call_width: f32,

    /// Pixel density used to determine the physical page size of PDF output.
    #[arg(long, default_value_t = 144.0)]
    pdf_dpi: f64,

    /// Optional higher pixel density for only the PDF call layer.
    #[arg(long, value_name = "DPI")]
    pdf_call_dpi: Option<f64>,

    /// Maximum core size of one bounded-memory PDF call-layer tile.
    #[arg(long, default_value_t = 2_048, value_name = "PIXELS")]
    pdf_call_tile_size: u32,

    /// Concurrent PDF tile compression jobs (0 = automatic, maximum 64).
    #[arg(long, default_value_t = 0, value_name = "COUNT")]
    pdf_compression_workers: usize,

    /// Retain and reuse checksummed PDF call-layer tiles after interruption.
    #[arg(long, conflicts_with = "restart")]
    resume: bool,

    /// Discard the existing PDF tile cache and start a fresh resumable render.
    #[arg(long, conflicts_with = "resume")]
    restart: bool,

    /// Stop after this many PDF tiles, preserving the cache for resume testing.
    #[arg(long, hide = true, value_name = "COUNT")]
    stop_after_pdf_tiles: Option<usize>,
}

#[derive(Debug, clap::Args)]
struct InspectArgs {
    #[arg(long, short = 'r')]
    repo: PathBuf,

    #[arg(long, default_value = "loc")]
    metric: String,

    #[arg(long)]
    no_calls: bool,

    /// Include conventional test directories and test files.
    #[arg(long)]
    include_tests: bool,

    /// Include files and directories whose names begin with a dot.
    #[arg(long)]
    include_hidden: bool,

    /// Exclude a repository-relative file or directory. Repeat as needed.
    #[arg(long = "exclude", value_name = "PATH")]
    excluded_paths: Vec<String>,

    #[command(flatten)]
    call_filters: CallFilterArgs,
}

#[derive(Debug, Default, clap::Args)]
struct CallFilterArgs {
    /// Keep calls whose source is this repository-relative file or directory.
    /// Repeat to match any source prefix.
    #[arg(long = "calls-from", value_name = "PATH")]
    sources: Vec<String>,

    /// Keep calls whose target is this repository-relative file or directory.
    /// Repeat to match any target prefix.
    #[arg(long = "calls-to", value_name = "PATH")]
    targets: Vec<String>,

    /// Keep calls with either endpoint in this repository-relative file or
    /// directory. Repeat to match any endpoint prefix.
    #[arg(long = "calls-in", value_name = "PATH")]
    paths: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Report<'a> {
    repository: String,
    revision: &'a str,
    dirty_worktree: bool,
    files: usize,
    excluded_test_files: usize,
    excluded_hidden_files: usize,
    excluded_custom_files: usize,
    excluded_paths: &'a [String],
    directories: usize,
    total_loc: u64,
    callsites: usize,
    callsites_before_filter: usize,
    call_filters: &'a CallPathFilters,
    same_file_calls_excluded: usize,
    analyzer: &'a code_atlas::model::AnalyzerReport,
    timings: &'a code_atlas::model::BuildTimings,
    render: Option<&'a code_atlas::render::RenderStats>,
    invariant: &'static str,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Render(args) => render(args),
        Command::Inspect(args) => inspect(args),
    }
}

fn render(args: RenderArgs) -> Result<()> {
    let metric: Metric = args.metric.parse()?;
    let theme: Theme = args.theme.parse()?;
    let backend: RenderBackend = args.backend.parse()?;
    let output_extension = args
        .output
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default();
    if (args.resume || args.restart || args.stop_after_pdf_tiles.is_some())
        && !output_extension.eq_ignore_ascii_case("pdf")
    {
        anyhow::bail!("--resume and --restart are available only for PDF output");
    }
    let (call_opacity, density_aware_exposure) = call_exposure(&args);
    eprintln!("Scanning {}…", args.repo.display());
    let mut atlas = build_atlas(
        &args.repo,
        &BuildOptions {
            metric,
            analyze_calls: !args.no_calls,
            include_tests: args.include_tests,
            include_hidden: args.include_hidden,
            excluded_paths: args.excluded_paths,
        },
    )?;
    let calls_before_filter = atlas.calls.len();
    let call_filters = args.call_filters.normalized()?;
    let calls_filtered = atlas.filter_calls(&call_filters);
    eprintln!(
        "Found {} files ({} test, {} hidden, and {} custom-path files excluded) and {} cross-file callsites ({} same-file and {} endpoint-filtered callsites excluded); rendering every retained callsite…",
        atlas.files().count(),
        atlas.excluded_test_files,
        atlas.excluded_hidden_files,
        atlas.excluded_custom_files,
        atlas.calls.len(),
        atlas.report.same_file_calls_excluded,
        calls_filtered,
    );

    let layout_options = LayoutOptions {
        width: args.width,
        height: args.height,
        margin_fraction: args.margin_fraction,
        texture_seed: args.texture_seed,
        bundle_strength: args.bundle_strength,
        ..LayoutOptions::default()
    };
    let render_options = RenderOptions {
        backend,
        theme,
        source_url_template: args.source_url_template.clone(),
        call_opacity,
        call_width: args.call_width,
        density_aware_exposure,
        pdf_dpi: args.pdf_dpi,
        pdf_call_dpi: args.pdf_call_dpi,
        pdf_call_tile_size: args.pdf_call_tile_size,
        pdf_resume: args.resume || args.restart,
        pdf_restart: args.restart,
        pdf_stop_after_tiles: args.stop_after_pdf_tiles,
        pdf_compression_workers: args.pdf_compression_workers,
        ..RenderOptions::default()
    };
    let stats = render_atlas(&mut atlas, &args.output, &layout_options, &render_options)?;
    if stats.calls_drawn != atlas.calls.len() {
        anyhow::bail!(
            "no-aggregation invariant failed: {} callsites but {} splines",
            atlas.calls.len(),
            stats.calls_drawn
        );
    }
    if let Some(cache) = &stats.pdf_tile_cache {
        eprintln!(
            "PDF tiles: {} rendered, {} reused; {}/level {} with {} workers, <= {} in flight; planning {} ms, scene {} ms, pipeline {} ms, render/readback {} ms, split {} ms, compression CPU {} ms, cache writes {} ms, assembly {} ms",
            cache.tiles_rendered,
            cache.tiles_reused,
            cache.compression_codec,
            cache.compression_level,
            cache.compression_workers,
            cache.max_in_flight_tiles,
            cache.timings.planning_ms,
            cache.timings.scene_prepare_ms,
            cache.timings.tile_pipeline_wall_ms,
            cache.timings.render_and_readback_ms,
            cache.timings.rgba_split_ms,
            cache.timings.compression_ms,
            cache.timings.cache_write_ms,
            cache.timings.pdf_assembly_ms,
        );
        if let (Some(spline), Some(raster)) = (
            cache.timings.spline_evaluation_gpu_ms,
            cache.timings.rasterization_gpu_ms,
        ) {
            eprintln!(
                "GPU timestamps: spline evaluation {spline:.3} ms, rasterization {raster:.3} ms"
            );
        }
        if cache.enabled {
            eprintln!(
                "Resume cache: {} ({})",
                cache.manifest_path, cache.cache_key
            );
        }
    }

    let report = report(&atlas, Some(&stats), calls_before_filter, &call_filters);
    let report_path = args.output.with_file_name(format!(
        "{}.report.json",
        args.output
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("code-atlas")
    ));
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("cannot save {}", report_path.display()))?;
    eprintln!(
        "Wrote {} ({} calls) and {}",
        args.output.display(),
        stats.calls_drawn,
        report_path.display()
    );
    Ok(())
}

fn call_exposure(args: &RenderArgs) -> (u8, bool) {
    (
        args.fixed_call_opacity
            .flatten()
            .unwrap_or(args.call_opacity),
        args.fixed_call_opacity.is_none(),
    )
}

fn inspect(args: InspectArgs) -> Result<()> {
    let mut atlas = build_atlas(
        &args.repo,
        &BuildOptions {
            metric: args.metric.parse()?,
            analyze_calls: !args.no_calls,
            include_tests: args.include_tests,
            include_hidden: args.include_hidden,
            excluded_paths: args.excluded_paths,
        },
    )?;
    let calls_before_filter = atlas.calls.len();
    let call_filters = args.call_filters.normalized()?;
    atlas.filter_calls(&call_filters);
    println!(
        "{}",
        serde_json::to_string_pretty(&report(&atlas, None, calls_before_filter, &call_filters,))?
    );
    Ok(())
}

impl CallFilterArgs {
    fn normalized(&self) -> Result<CallPathFilters> {
        CallPathFilters::new(&self.sources, &self.targets, &self.paths)
    }
}

fn report<'a>(
    atlas: &'a code_atlas::model::Atlas,
    render: Option<&'a code_atlas::render::RenderStats>,
    callsites_before_filter: usize,
    call_filters: &'a CallPathFilters,
) -> Report<'a> {
    Report {
        repository: atlas.root_path.display().to_string(),
        revision: &atlas.revision,
        dirty_worktree: atlas.dirty,
        files: atlas.files().count(),
        excluded_test_files: atlas.excluded_test_files,
        excluded_hidden_files: atlas.excluded_hidden_files,
        excluded_custom_files: atlas.excluded_custom_files,
        excluded_paths: &atlas.excluded_paths,
        directories: atlas.nodes.iter().filter(|node| !node.is_file()).count(),
        total_loc: atlas.total_loc(),
        callsites: atlas.calls.len(),
        callsites_before_filter,
        call_filters,
        same_file_calls_excluded: atlas.report.same_file_calls_excluded,
        analyzer: &atlas.report,
        timings: &atlas.timings,
        render,
        invariant: "one retained cross-file callsite equals one line-addressed rendered spline; same-file and CLI-filtered calls are excluded; no aggregation or sampling",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn render_args(extra: &[&str]) -> RenderArgs {
        let mut arguments = vec!["code-atlas", "render", "--repo", "."];
        arguments.extend_from_slice(extra);
        match Cli::try_parse_from(arguments).unwrap().command {
            Command::Render(args) => args,
            Command::Inspect(_) => unreachable!(),
        }
    }

    #[test]
    fn fixed_call_opacity_accepts_an_optional_manual_value() {
        assert_eq!(call_exposure(&render_args(&[])), (34, true));
        assert_eq!(
            call_exposure(&render_args(&["--fixed-call-opacity"])),
            (34, false)
        );
        assert_eq!(
            call_exposure(&render_args(&["--fixed-call-opacity", "72"])),
            (72, false)
        );
    }

    #[test]
    fn bare_fixed_call_opacity_preserves_the_separate_call_opacity_option() {
        assert_eq!(
            call_exposure(&render_args(&[
                "--call-opacity",
                "61",
                "--fixed-call-opacity",
            ])),
            (61, false)
        );
    }
}
