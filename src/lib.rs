//! Repository analysis, treemap layout, and architectural rendering for Code Atlas.
//!
//! Most users should install and use the `code-atlas` command-line application.
//! The library API is public for integrations, but remains experimental during
//! the `0.x` release series.

pub mod analyze;
mod export;
pub mod git;
mod html;
pub mod layout;
pub mod model;
pub mod render;
pub mod render_wgpu;
mod scip;

use std::{path::Path, time::Instant};

use anyhow::Result;

use crate::{
    analyze::analyze_calls,
    git::scan_repository,
    layout::{LayoutOptions, layout_repository},
    model::{Atlas, Metric},
    render::{RenderOptions, RenderStats, render_output},
};

#[derive(Debug, Clone)]
pub struct BuildOptions {
    pub metric: Metric,
    pub analyze_calls: bool,
    pub include_tests: bool,
    pub include_hidden: bool,
    pub excluded_paths: Vec<String>,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            metric: Metric::Loc,
            analyze_calls: true,
            include_tests: false,
            include_hidden: false,
            excluded_paths: Vec::new(),
        }
    }
}

pub fn build_atlas(repo: &Path, options: &BuildOptions) -> Result<Atlas> {
    let scan_started = Instant::now();
    let mut atlas = scan_repository(
        repo,
        options.metric,
        options.include_tests,
        options.include_hidden,
        &options.excluded_paths,
    )?;
    atlas.timings.scan_ms = elapsed_ms(scan_started);
    if options.analyze_calls {
        let analyze_started = Instant::now();
        analyze_calls(&mut atlas)?;
        atlas.timings.analyze_ms = elapsed_ms(analyze_started);
    }
    Ok(atlas)
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}

pub fn render_atlas(
    atlas: &mut Atlas,
    output: &Path,
    layout_options: &LayoutOptions,
    render_options: &RenderOptions,
) -> Result<RenderStats> {
    layout_repository(atlas, layout_options)?;
    render_output(atlas, output, layout_options, render_options)
}
