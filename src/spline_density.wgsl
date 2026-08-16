struct ControlPoint {
    position: vec2<f32>,
};

struct CallRecord {
    control_offset: u32,
    control_count: u32,
    segment_count: u32,
    seed: u32,
};

struct TileInstance {
    call_index: u32,
    pass_index: u32,
};

struct SampledPoint {
    position: vec2<f32>,
};

struct Params {
    viewport: vec4<f32>,
    source: vec4<f32>,
    destination: vec4<f32>,
    pass0: vec4<f32>,
    pass1: vec4<f32>,
    settings: vec4<f32>,
    counts: vec4<u32>,
};

@group(0) @binding(0)
var<storage, read> controls: array<ControlPoint>;
@group(0) @binding(1)
var<storage, read> calls: array<CallRecord>;
@group(0) @binding(2)
var<storage, read> instances: array<TileInstance>;
@group(0) @binding(3)
var<storage, read_write> sampled_points: array<SampledPoint>;
@group(0) @binding(4)
var<uniform> params: Params;

fn knot(index: u32, control_count: u32, degree: u32) -> f32 {
    if index <= degree {
        return 0.0;
    }
    if index >= control_count {
        return 1.0;
    }
    return f32(index - degree) / f32(control_count - degree);
}

fn bundled_control(call: CallRecord, index: u32) -> vec2<f32> {
    let point = controls[call.control_offset + index].position;
    let first = controls[call.control_offset].position;
    let last = controls[call.control_offset + call.control_count - 1u].position;
    let amount = f32(index) / f32(max(call.control_count - 1u, 1u));
    let direct = mix(first, last, amount);
    return mix(direct, point, params.settings.x);
}

fn spline_point(call: CallRecord, amount: f32) -> vec2<f32> {
    let degree = min(call.control_count - 1u, 3u);
    var span = call.control_count - 1u;
    if amount < 1.0 {
        span = min(
            degree + u32(floor(amount * f32(call.control_count - degree))),
            call.control_count - 1u,
        );
    }

    var points: array<vec2<f32>, 4>;
    for (var index = 0u; index <= degree; index += 1u) {
        points[index] = bundled_control(call, span - degree + index);
    }
    for (var level = 1u; level <= degree; level += 1u) {
        var index = degree;
        loop {
            if index < level {
                break;
            }
            let knot_index = span - degree + index;
            let denominator =
                knot(knot_index + degree + 1u - level, call.control_count, degree)
                - knot(knot_index, call.control_count, degree);
            var alpha = 0.0;
            if abs(denominator) > 0.000001 {
                alpha = (amount - knot(knot_index, call.control_count, degree)) / denominator;
            }
            points[index] = mix(points[index - 1u], points[index], alpha);
            if index == 0u {
                break;
            }
            index -= 1u;
        }
    }
    return points[degree];
}

fn hash(value: u32) -> u32 {
    var result = value;
    result ^= result >> 16u;
    result *= 0x7feb352du;
    result ^= result >> 15u;
    result *= 0x846ca68bu;
    result ^= result >> 16u;
    return result;
}

fn jitter(seed: u32, sample: u32, pass_index: u32, amplitude: f32) -> vec2<f32> {
    let base = seed ^ (sample * 0x9e3779b9u) ^ (pass_index * 0x85ebca6bu);
    let x = f32(hash(base) & 0x00ffffffu) / 16777215.0;
    let y = f32(hash(base ^ 0xc2b2ae35u) & 0x00ffffffu) / 16777215.0;
    return (vec2<f32>(x, y) * 2.0 - 1.0) * amplitude;
}

@compute @workgroup_size(8, 8, 1)
fn compute_points(@builtin(global_invocation_id) id: vec3<u32>) {
    let sample = id.x;
    let instance_index = id.y;
    if instance_index >= params.counts.y || sample >= params.counts.x {
        return;
    }
    let instance = instances[instance_index];
    let call = calls[instance.call_index];
    if sample > call.segment_count {
        return;
    }
    let amount = f32(sample) / f32(max(call.segment_count, 1u));
    let pass_settings = select(params.pass0, params.pass1, instance.pass_index == 1u);
    let point = spline_point(call, amount)
        + jitter(call.seed, sample, instance.pass_index, pass_settings.z)
        - params.viewport.xy;
    sampled_points[instance_index * params.counts.x + sample].position = point;
}
