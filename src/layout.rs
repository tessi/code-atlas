use std::cmp::Ordering;

use anyhow::{Result, ensure};

use crate::model::{Atlas, Callsite, NodeId, Point, Rect};

#[derive(Debug, Clone)]
pub struct LayoutOptions {
    pub width: u32,
    pub height: u32,
    pub margin_fraction: f64,
    pub directory_padding: f64,
    pub texture_seed: u64,
    pub bundle_strength: f64,
}

impl Default for LayoutOptions {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            margin_fraction: 0.045,
            directory_padding: 1.5,
            texture_seed: 932,
            bundle_strength: 0.96,
        }
    }
}

pub fn layout_repository(atlas: &mut Atlas, options: &LayoutOptions) -> Result<()> {
    ensure!(
        options.width > 0 && options.height > 0,
        "output dimensions must be positive"
    );
    ensure!(!atlas.nodes.is_empty(), "repository has no hierarchy nodes");
    let margin = (options.width.min(options.height) as f64
        * options.margin_fraction.clamp(0.0, 0.20))
    .round();
    atlas.nodes[0].rect = Rect {
        x0: margin,
        y0: margin,
        x1: options.width as f64 - margin,
        y1: options.height as f64 - margin,
    };
    layout_children(atlas, 0, options);
    Ok(())
}

fn layout_children(atlas: &mut Atlas, parent: NodeId, options: &LayoutOptions) {
    let mut children = atlas.nodes[parent].children.clone();
    if children.is_empty() {
        return;
    }
    children.sort_by(|left, right| {
        atlas.nodes[*right]
            .weight
            .partial_cmp(&atlas.nodes[*left].weight)
            .unwrap_or(Ordering::Equal)
            .then_with(|| atlas.nodes[*left].path.cmp(&atlas.nodes[*right].path))
    });

    let depth = atlas.nodes[parent].depth;
    let padding = if parent == 0 {
        0.0
    } else {
        options.directory_padding / (depth as f64).sqrt().max(1.0)
    };
    let area = atlas.nodes[parent].rect.inset(padding);
    let rectangles = squarify(
        &children
            .iter()
            .map(|id| atlas.nodes[*id].weight.max(0.0001))
            .collect::<Vec<_>>(),
        area,
    );

    for (child, rect) in children.into_iter().zip(rectangles) {
        atlas.nodes[child].rect = rect;
        layout_children(atlas, child, options);
    }
}

pub fn squarify(weights: &[f64], rect: Rect) -> Vec<Rect> {
    if weights.is_empty() {
        return Vec::new();
    }
    let total = weights.iter().sum::<f64>().max(f64::EPSILON);
    let scale = rect.area() / total;
    let areas: Vec<f64> = weights.iter().map(|weight| weight * scale).collect();
    let mut output = vec![Rect::default(); weights.len()];
    let mut remaining = rect;
    let mut row: Vec<usize> = Vec::new();

    for index in 0..areas.len() {
        let short_side = remaining.width().min(remaining.height()).max(f64::EPSILON);
        if row.is_empty()
            || worst_ratio(&row_with(&row, index), &areas, short_side)
                <= worst_ratio(&row, &areas, short_side)
        {
            row.push(index);
        } else {
            layout_row(&row, &areas, &mut remaining, &mut output);
            row.clear();
            row.push(index);
        }
    }
    if !row.is_empty() {
        layout_row(&row, &areas, &mut remaining, &mut output);
    }
    output
}

fn row_with(row: &[usize], index: usize) -> Vec<usize> {
    let mut candidate = row.to_vec();
    candidate.push(index);
    candidate
}

fn worst_ratio(row: &[usize], areas: &[f64], side: f64) -> f64 {
    if row.is_empty() {
        return f64::INFINITY;
    }
    let sum = row.iter().map(|index| areas[*index]).sum::<f64>();
    let min = row
        .iter()
        .map(|index| areas[*index])
        .fold(f64::INFINITY, f64::min)
        .max(f64::EPSILON);
    let max = row
        .iter()
        .map(|index| areas[*index])
        .fold(0.0, f64::max)
        .max(f64::EPSILON);
    ((side * side * max) / (sum * sum)).max((sum * sum) / (side * side * min))
}

fn layout_row(row: &[usize], areas: &[f64], remaining: &mut Rect, output: &mut [Rect]) {
    let row_area = row.iter().map(|index| areas[*index]).sum::<f64>();
    if remaining.width() <= remaining.height() {
        let height = (row_area / remaining.width().max(f64::EPSILON)).min(remaining.height());
        let mut x = remaining.x0;
        for (position, index) in row.iter().enumerate() {
            let width = if position + 1 == row.len() {
                remaining.x1 - x
            } else {
                areas[*index] / height.max(f64::EPSILON)
            };
            output[*index] = Rect {
                x0: x,
                y0: remaining.y0,
                x1: x + width,
                y1: remaining.y0 + height,
            };
            x += width;
        }
        remaining.y0 += height;
    } else {
        let width = (row_area / remaining.height().max(f64::EPSILON)).min(remaining.width());
        let mut y = remaining.y0;
        for (position, index) in row.iter().enumerate() {
            let height = if position + 1 == row.len() {
                remaining.y1 - y
            } else {
                areas[*index] / width.max(f64::EPSILON)
            };
            output[*index] = Rect {
                x0: remaining.x0,
                y0: y,
                x1: remaining.x0 + width,
                y1: y + height,
            };
            y += height;
        }
        remaining.x0 += width;
    }
}

pub fn call_curve(atlas: &Atlas, call: &Callsite, options: &LayoutOptions) -> Vec<Point> {
    let mut controls = call_control_points(atlas, call);

    let first = controls[0];
    let last = *controls.last().expect("target control point");
    let denominator = controls.len().saturating_sub(1).max(1) as f64;
    let bundle_strength = options.bundle_strength.clamp(0.0, 1.0);
    for (index, point) in controls.iter_mut().enumerate() {
        let t = index as f64 / denominator;
        let direct = Point {
            x: lerp(first.x, last.x, t),
            y: lerp(first.y, last.y, t),
        };
        point.x = lerp(direct.x, point.x, bundle_strength);
        point.y = lerp(direct.y, point.y, bundle_strength);
    }
    clamped_bspline(&controls)
}

/// Build the unbundled source-line -> file -> directory hierarchy -> file ->
/// target-line control polygon. Line anchors are the leaves; file centers are
/// the first shared hierarchy level, so calls from different lines diverge
/// locally before joining the same bundled route.
pub fn call_control_points(atlas: &Atlas, call: &Callsite) -> Vec<Point> {
    let source_file = &atlas.nodes[call.source];
    let target_file = &atlas.nodes[call.target];
    let source = line_anchor(source_file.rect, source_file.loc, call.source_line);
    let target = call
        .target_line
        .map(|line| line_anchor(target_file.rect, target_file.loc, line))
        .unwrap_or_else(|| target_file.rect.center());
    let hierarchy = atlas.hierarchy_path(call.source, call.target);
    let mut controls = Vec::with_capacity(hierarchy.len() + 2);
    controls.push(source);
    controls.extend(hierarchy.iter().map(|id| atlas.nodes[*id].rect.center()));
    controls.push(target);
    controls
}

/// Map a one-based source line to the center of its deterministic raster cell
/// inside a file parcel. Cells follow a serpentine scanline, keeping adjacent
/// line ranges spatially adjacent at row boundaries. If multiple pixels map to
/// one line, the center of that pixel range is used; if multiple lines share a
/// pixel, they intentionally share that pixel center.
pub fn line_anchor(rect: Rect, line_count: u64, line: u32) -> Point {
    let min_x = rect.x0.ceil() as i64;
    let min_y = rect.y0.ceil() as i64;
    let max_x = rect.x1.floor() as i64;
    let max_y = rect.y1.floor() as i64;
    let width = max_x.saturating_sub(min_x) as u64;
    let height = max_y.saturating_sub(min_y) as u64;
    if width == 0 || height == 0 {
        return rect.center();
    }

    let pixel_count = width.saturating_mul(height).max(1);
    let line_count = line_count.max(1);
    let line_index = u64::from(line.saturating_sub(1)).min(line_count - 1);
    let region_start = scale_ratio_floor(line_index, pixel_count, line_count);
    let region_end = scale_ratio_ceil(line_index + 1, pixel_count, line_count)
        .max(region_start + 1)
        .min(pixel_count);
    let pixel_index = region_start + (region_end - region_start - 1) / 2;
    let row = pixel_index / width;
    let scan_column = pixel_index % width;
    let column = if row.is_multiple_of(2) {
        scan_column
    } else {
        width - 1 - scan_column
    };
    Point {
        x: min_x as f64 + column as f64 + 0.5,
        y: min_y as f64 + row as f64 + 0.5,
    }
}

fn scale_ratio_floor(value: u64, scale: u64, denominator: u64) -> u64 {
    ((u128::from(value) * u128::from(scale)) / u128::from(denominator)) as u64
}

fn scale_ratio_ceil(value: u64, scale: u64, denominator: u64) -> u64 {
    let numerator = u128::from(value) * u128::from(scale);
    numerator.div_ceil(u128::from(denominator)) as u64
}

pub fn sample_rect(rect: Rect, samples_per_side: usize) -> Vec<Point> {
    let samples = samples_per_side.max(1);
    let mut points = Vec::with_capacity(samples * 4);
    for index in 0..samples {
        let t = index as f64 / samples as f64;
        points.push(Point {
            x: lerp(rect.x0, rect.x1, t),
            y: rect.y0,
        });
    }
    for index in 0..samples {
        let t = index as f64 / samples as f64;
        points.push(Point {
            x: rect.x1,
            y: lerp(rect.y0, rect.y1, t),
        });
    }
    for index in 0..samples {
        let t = index as f64 / samples as f64;
        points.push(Point {
            x: lerp(rect.x1, rect.x0, t),
            y: rect.y1,
        });
    }
    for index in 0..samples {
        let t = index as f64 / samples as f64;
        points.push(Point {
            x: rect.x0,
            y: lerp(rect.y1, rect.y0, t),
        });
    }
    points
}

fn clamped_bspline(controls: &[Point]) -> Vec<Point> {
    if controls.len() < 2 {
        return controls.to_vec();
    }
    let degree = controls.len().saturating_sub(1).min(3);
    let knots = clamped_uniform_knots(controls.len(), degree);
    let knot_spans = controls.len().saturating_sub(degree).max(1);
    let line_segments = (knot_spans * 16).max(64);
    let mut output = Vec::with_capacity(line_segments + 1);
    for sample in 0..=line_segments {
        let t = sample as f64 / line_segments as f64;
        output.push(de_boor(controls, degree, &knots, t));
    }
    output
}

fn clamped_uniform_knots(control_count: usize, degree: usize) -> Vec<f64> {
    let interior_count = control_count.saturating_sub(degree + 1);
    let mut knots = Vec::with_capacity(control_count + degree + 1);
    knots.extend(std::iter::repeat_n(0.0, degree + 1));
    let denominator = (interior_count + 1) as f64;
    for index in 1..=interior_count {
        knots.push(index as f64 / denominator);
    }
    knots.extend(std::iter::repeat_n(1.0, degree + 1));
    knots
}

fn de_boor(controls: &[Point], degree: usize, knots: &[f64], t: f64) -> Point {
    let last_control = controls.len() - 1;
    let span = if t >= 1.0 {
        last_control
    } else {
        (degree..=last_control)
            .find(|index| t >= knots[*index] && t < knots[*index + 1])
            .unwrap_or(last_control)
    };
    let mut points: Vec<Point> = (0..=degree)
        .map(|index| controls[span - degree + index])
        .collect();

    for level in 1..=degree {
        for index in (level..=degree).rev() {
            let knot_index = span - degree + index;
            let denominator = knots[knot_index + degree + 1 - level] - knots[knot_index];
            let alpha = if denominator.abs() <= f64::EPSILON {
                0.0
            } else {
                (t - knots[knot_index]) / denominator
            };
            points[index] = Point {
                x: lerp(points[index - 1].x, points[index].x, alpha),
                y: lerp(points[index - 1].y, points[index].y, alpha),
            };
        }
    }
    points[degree]
}

fn lerp(start: f64, end: f64, amount: f64) -> f64 {
    start + (end - start) * amount
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squarify_preserves_total_area() {
        let root = Rect {
            x0: 0.0,
            y0: 0.0,
            x1: 1600.0,
            y1: 900.0,
        };
        let rectangles = squarify(&[8.0, 5.0, 3.0, 2.0, 1.0], root);
        let total = rectangles.iter().map(|rect| rect.area()).sum::<f64>();
        assert!((total - root.area()).abs() < 0.01);
        assert!(
            rectangles
                .iter()
                .all(|rect| rect.width() > 0.0 && rect.height() > 0.0)
        );
    }

    #[test]
    fn default_bundle_strength_is_tight_but_not_fully_confluent() {
        assert_eq!(LayoutOptions::default().bundle_strength, 0.96);
    }

    #[test]
    fn clamped_spline_interpolates_endpoints_and_reduces_degree() {
        let linear = clamped_bspline(&[Point { x: 0.0, y: 0.0 }, Point { x: 2.0, y: 2.0 }]);
        assert_eq!(linear.first(), Some(&Point { x: 0.0, y: 0.0 }));
        assert_eq!(linear.last(), Some(&Point { x: 2.0, y: 2.0 }));
        assert_point_near(linear[32], Point { x: 1.0, y: 1.0 });

        let quadratic = clamped_bspline(&[
            Point { x: 0.0, y: 0.0 },
            Point { x: 1.0, y: 1.0 },
            Point { x: 2.0, y: 0.0 },
        ]);
        assert_point_near(quadratic[32], Point { x: 1.0, y: 0.5 });

        let cubic = clamped_bspline(&[
            Point { x: 0.0, y: 0.0 },
            Point { x: 0.0, y: 1.0 },
            Point { x: 1.0, y: 1.0 },
            Point { x: 1.0, y: 0.0 },
        ]);
        assert_point_near(cubic[32], Point { x: 0.5, y: 0.75 });
    }

    #[test]
    fn line_anchors_follow_a_serpentine_pixel_scan() {
        let rect = Rect {
            x0: 0.0,
            y0: 0.0,
            x1: 4.0,
            y1: 2.0,
        };
        assert_eq!(line_anchor(rect, 8, 1), Point { x: 0.5, y: 0.5 });
        assert_eq!(line_anchor(rect, 8, 4), Point { x: 3.5, y: 0.5 });
        assert_eq!(line_anchor(rect, 8, 5), Point { x: 3.5, y: 1.5 });
        assert_eq!(line_anchor(rect, 8, 8), Point { x: 0.5, y: 1.5 });
        assert_eq!(
            line_anchor(rect, 8, 999),
            Point { x: 0.5, y: 1.5 },
            "out-of-range analyzer lines clamp to the final line cell"
        );
    }

    #[test]
    fn every_pixel_maps_to_a_line_range_when_pixels_outnumber_lines() {
        let rect = Rect {
            x0: 10.0,
            y0: 20.0,
            x1: 18.0,
            y1: 21.0,
        };
        let anchors: Vec<_> = (1..=4).map(|line| line_anchor(rect, 4, line)).collect();
        assert_eq!(anchors[0], Point { x: 10.5, y: 20.5 });
        assert_eq!(anchors[1], Point { x: 12.5, y: 20.5 });
        assert_eq!(anchors[2], Point { x: 14.5, y: 20.5 });
        assert_eq!(anchors[3], Point { x: 16.5, y: 20.5 });
    }

    fn assert_point_near(actual: Point, expected: Point) {
        assert!((actual.x - expected.x).abs() < 1e-9, "x: {actual:?}");
        assert!((actual.y - expected.y).abs() < 1e-9, "y: {actual:?}");
    }
}
