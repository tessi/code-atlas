use std::{collections::HashMap, path::PathBuf};

use code_atlas::{
    layout::{LayoutOptions, call_control_points, call_curve, line_anchor},
    model::{AnalyzerReport, Atlas, BuildTimings, Callsite, Node, NodeKind, Rect},
    render::{RenderOptions, render_output, render_png},
};

fn node(
    id: usize,
    parent: Option<usize>,
    children: Vec<usize>,
    kind: NodeKind,
    path: &str,
    weight: f64,
) -> Node {
    Node {
        id,
        parent,
        children,
        kind,
        name: path.rsplit('/').next().unwrap_or("root").to_owned(),
        path: path.to_owned(),
        depth: usize::from(parent.is_some()),
        bytes: weight as u64,
        loc: 20,
        commits: 1,
        weight,
        language: "rust".to_owned(),
        rect: Rect::default(),
    }
}

#[test]
fn line_addressed_routes_remain_individual_splines() {
    let nodes = vec![
        node(0, None, vec![1, 2], NodeKind::Directory, "", 20.0),
        node(1, Some(0), vec![], NodeKind::File, "src/a.rs", 10.0),
        node(2, Some(0), vec![], NodeKind::File, "src/b.rs", 10.0),
    ];
    let calls = vec![
        Callsite {
            id: 1,
            source: 1,
            source_line: 4,
            target: 2,
            target_line: Some(8),
            callee: "b::run".to_owned(),
            kind: "runtime".to_owned(),
            analyzer: "fixture".to_owned(),
            confidence: 1.0,
        },
        Callsite {
            id: 2,
            source: 1,
            source_line: 17,
            target: 2,
            target_line: Some(8),
            callee: "b::run".to_owned(),
            kind: "runtime".to_owned(),
            analyzer: "fixture".to_owned(),
            confidence: 1.0,
        },
    ];
    let mut atlas = Atlas {
        root_path: PathBuf::from("fixture"),
        revision: "test".to_owned(),
        dirty: false,
        excluded_test_files: 0,
        excluded_hidden_files: 0,
        excluded_custom_files: 0,
        excluded_paths: Vec::new(),
        nodes,
        calls,
        path_to_id: HashMap::new(),
        report: AnalyzerReport::default(),
        timings: BuildTimings::default(),
    };
    let options = LayoutOptions {
        width: 320,
        height: 200,
        ..LayoutOptions::default()
    };
    code_atlas::layout::layout_repository(&mut atlas, &options).unwrap();

    let first_curve = call_curve(&atlas, &atlas.calls[0], &options);
    let second_curve = call_curve(&atlas, &atlas.calls[1], &options);
    assert_ne!(
        first_curve, second_curve,
        "calls from different source lines must have distinct geometry"
    );
    assert_eq!(
        first_curve.first(),
        Some(&line_anchor(atlas.nodes[1].rect, atlas.nodes[1].loc, 4))
    );
    assert_eq!(
        second_curve.first(),
        Some(&line_anchor(atlas.nodes[1].rect, atlas.nodes[1].loc, 17))
    );
    assert_eq!(first_curve.last(), second_curve.last());

    let controls = call_control_points(&atlas, &atlas.calls[0]);
    assert_eq!(controls[0], *first_curve.first().unwrap());
    assert_eq!(controls[1], atlas.nodes[1].rect.center());
    assert_eq!(controls[controls.len() - 2], atlas.nodes[2].rect.center());
    assert_eq!(controls.last(), first_curve.last());

    let directory = tempfile::tempdir().unwrap();
    let single_output = directory.path().join("single.png");
    let double_output = directory.path().join("double.png");
    let mut exact_overlap_atlas = atlas.clone();
    exact_overlap_atlas.calls[1].source_line = exact_overlap_atlas.calls[0].source_line;
    exact_overlap_atlas.calls[1].target_line = exact_overlap_atlas.calls[0].target_line;
    let mut single_call_atlas = exact_overlap_atlas.clone();
    single_call_atlas.calls.truncate(1);
    render_png(
        &single_call_atlas,
        &single_output,
        &options,
        &RenderOptions::default(),
    )
    .unwrap();
    let stats = render_png(
        &exact_overlap_atlas,
        &double_output,
        &options,
        &RenderOptions::default(),
    )
    .unwrap();

    let single = tiny_skia::Pixmap::load_png(&single_output).unwrap();
    let double = tiny_skia::Pixmap::load_png(&double_output).unwrap();
    assert!(
        rgb_darkness(&double, 9, 9, 311, 191) > rgb_darkness(&single, 9, 9, 311, 191),
        "drawing a second callsite over the same route must darken the pencil stroke"
    );

    assert_eq!(atlas.calls.len(), 2);
    assert_eq!(stats.calls_drawn, 2);
    assert_eq!(stats.files_drawn, 2);
    assert!(double_output.exists());

    let svg_output = directory.path().join("atlas.svg");
    let pdf_output = directory.path().join("atlas.pdf");
    let svg_stats = render_output(
        &exact_overlap_atlas,
        &svg_output,
        &options,
        &RenderOptions::default(),
    )
    .unwrap();
    let pdf_options = RenderOptions {
        pdf_dpi: 144.0,
        pdf_call_dpi: Some(288.0),
        pdf_call_tile_size: 128,
        ..RenderOptions::default()
    };
    let pdf_stats =
        render_output(&exact_overlap_atlas, &pdf_output, &options, &pdf_options).unwrap();
    let svg = std::fs::read_to_string(svg_output).unwrap();
    let pdf = std::fs::read(pdf_output).unwrap();
    assert_eq!(svg.matches("<image id=\"call-density\"").count(), 1);
    assert!(svg.contains("<g id=\"labels\""));
    assert!(svg.contains("<g id=\"direction-legend\""));
    assert!(svg.find("call-density").unwrap() < svg.find("id=\"labels\"").unwrap());
    assert!(pdf.starts_with(b"%PDF-1.7"));
    assert!(
        pdf.windows(b"/MediaBox [0 0 160.0000 100.0000]".len())
            .any(|window| window == b"/MediaBox [0 0 160.0000 100.0000]")
    );
    let courier = b"/BaseFont /Courier";
    assert!(pdf.windows(courier.len()).any(|window| window == courier));
    assert_eq!(svg_stats.calls_drawn, 2);
    assert_eq!(pdf_stats.calls_drawn, 2);
    assert_eq!(pdf_stats.width, 320);
    assert_eq!(pdf_stats.height, 200);
    assert_eq!(pdf_stats.call_layer_width, 640);
    assert_eq!(pdf_stats.call_layer_height, 400);
    assert_eq!(pdf_stats.call_layer_dpi, Some(288.0));
    assert!(pdf_stats.call_layer_tiles > 1);
    assert_eq!(pdf_stats.call_layer_tile_size, Some(128));
    assert!(
        pdf_stats
            .call_layer_overlap
            .is_some_and(|overlap| overlap >= 4)
    );
    assert_eq!(
        pdf.windows(b"/Subtype /Image".len())
            .filter(|window| *window == b"/Subtype /Image")
            .count(),
        pdf_stats.call_layer_tiles * 2
    );
}

fn rgb_darkness(pixmap: &tiny_skia::Pixmap, x0: u32, y0: u32, x1: u32, y1: u32) -> u64 {
    let width = pixmap.width() as usize;
    let data = pixmap.data();
    let mut darkness = 0;
    for y in y0 as usize..y1 as usize {
        for x in x0 as usize..x1 as usize {
            let offset = (y * width + x) * 4;
            darkness += u64::from(255 - data[offset]);
            darkness += u64::from(255 - data[offset + 1]);
            darkness += u64::from(255 - data[offset + 2]);
        }
    }
    darkness
}

#[test]
fn hierarchy_path_passes_through_lowest_common_ancestor_once() {
    let atlas = Atlas {
        root_path: PathBuf::from("fixture"),
        revision: "test".to_owned(),
        dirty: false,
        excluded_test_files: 0,
        excluded_hidden_files: 0,
        excluded_custom_files: 0,
        excluded_paths: Vec::new(),
        nodes: vec![
            node(0, None, vec![1, 2], NodeKind::Directory, "", 2.0),
            node(1, Some(0), vec![3], NodeKind::Directory, "lib", 1.0),
            node(2, Some(0), vec![4], NodeKind::Directory, "native", 1.0),
            node(3, Some(1), vec![], NodeKind::File, "lib/a.ex", 1.0),
            node(4, Some(2), vec![], NodeKind::File, "native/a.rs", 1.0),
        ],
        calls: Vec::new(),
        path_to_id: HashMap::new(),
        report: AnalyzerReport::default(),
        timings: BuildTimings::default(),
    };

    assert_eq!(atlas.hierarchy_path(3, 4), vec![3, 1, 0, 2, 4]);
}
