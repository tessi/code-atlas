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

@group(0) @binding(1)
var<storage, read> calls: array<CallRecord>;
@group(0) @binding(2)
var<storage, read> instances: array<TileInstance>;
@group(0) @binding(3)
var<storage, read> sampled_points: array<SampledPoint>;
@group(0) @binding(4)
var<uniform> params: Params;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) density: vec4<f32>,
};

fn safe_normal(start: vec2<f32>, end: vec2<f32>) -> vec2<f32> {
    let delta = end - start;
    let magnitude = max(length(delta), 0.000001);
    return vec2<f32>(-delta.y, delta.x) / magnitude;
}

fn joined_offset(
    previous: vec2<f32>,
    current: vec2<f32>,
    next: vec2<f32>,
    half_width: f32,
) -> vec2<f32> {
    let incoming = safe_normal(previous, current);
    let outgoing = safe_normal(current, next);
    let sum = incoming + outgoing;
    let sum_length = length(sum);
    if sum_length <= 0.000001 {
        return outgoing * half_width;
    }
    let miter = sum / sum_length;
    let denominator = max(abs(dot(miter, outgoing)), 0.25);
    let miter_length = min(half_width / denominator, half_width * 2.5);
    return miter * miter_length;
}

fn point_for(instance_index: u32, sample: u32) -> vec2<f32> {
    return sampled_points[instance_index * params.counts.x + sample].position;
}

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    let instance = instances[instance_index];
    let call = calls[instance.call_index];
    let segment = vertex_index / 6u;
    var output: VertexOutput;
    if segment >= call.segment_count {
        output.position = vec4<f32>(2.0, 2.0, 0.0, 1.0);
        output.density = vec4<f32>(0.0);
        return output;
    }

    let start = point_for(instance_index, segment);
    let end = point_for(instance_index, segment + 1u);
    let pass_settings = select(params.pass0, params.pass1, instance.pass_index == 1u);
    var start_offset = safe_normal(start, end) * pass_settings.x;
    if segment > 0u {
        start_offset = joined_offset(
            point_for(instance_index, segment - 1u),
            start,
            end,
            pass_settings.x,
        );
    }
    var end_offset = safe_normal(start, end) * pass_settings.x;
    if segment + 1u < call.segment_count {
        end_offset = joined_offset(
            start,
            end,
            point_for(instance_index, segment + 2u),
            pass_settings.x,
        );
    }

    let corner = vertex_index % 6u;
    var point = start + start_offset;
    var amount = f32(segment) / f32(call.segment_count);
    if corner == 1u || corner == 4u {
        point = start - start_offset;
    } else if corner == 2u || corner == 3u {
        point = end + end_offset;
        amount = f32(segment + 1u) / f32(call.segment_count);
    } else if corner == 5u {
        point = end - end_offset;
        amount = f32(segment + 1u) / f32(call.segment_count);
    }

    let linear_color = mix(params.source.rgb, params.destination.rgb, amount);
    output.position = vec4<f32>(
        point.x / params.viewport.z * 2.0 - 1.0,
        1.0 - point.y / params.viewport.w * 2.0,
        0.0,
        1.0,
    );
    output.density = vec4<f32>(linear_color * pass_settings.y, pass_settings.y);
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return input.density;
}
