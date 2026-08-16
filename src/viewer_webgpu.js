async function createWebGpuCallRenderer(canvas, data, packedPoints, onFailure) {
  if (!data.calls.length || !packedPoints.byteLength || !navigator.gpu || typeof GPUBufferUsage === 'undefined') return null;
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;
  const callWords = new Uint32Array(data.calls.length * 4);
  let maximumSegments = 1;
  for (let index = 0; index < data.calls.length; index++) {
    const call = data.calls[index];
    callWords[index * 4] = call.pointStart;
    callWords[index * 4 + 1] = call.pointCount;
    maximumSegments = Math.max(maximumSegments, call.pointCount - 1);
  }
  const pointBytes = Math.max(4, Math.ceil(packedPoints.byteLength / 4) * 4);
  const callBytes = Math.max(16, callWords.byteLength);
  const visibilityBytes = Math.max(4, data.calls.length * 4);
  const requiredStorageBytes = Math.max(pointBytes, callBytes, visibilityBytes);
  if (requiredStorageBytes > adapter.limits.maxStorageBufferBindingSize || requiredStorageBytes > adapter.limits.maxBufferSize) return null;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: requiredStorageBytes,
      maxBufferSize: requiredStorageBytes
    }
  });
  const context = canvas.getContext('webgpu');
  if (!context) {
    device.destroy();
    return null;
  }

  const pointBuffer = device.createBuffer({
    label: 'Code Atlas packed curve points',
    size: pointBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(pointBuffer, 0, packedPoints);
  const callBuffer = device.createBuffer({
    label: 'Code Atlas call records',
    size: callBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(callBuffer, 0, callWords);
  const visibilityBuffer = device.createBuffer({
    label: 'Code Atlas call visibility',
    size: visibilityBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  const uniformBuffer = device.createBuffer({
    label: 'Code Atlas interactive density parameters',
    size: 80,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  const accumulationShader = device.createShaderModule({
    label: 'Code Atlas interactive optical-density accumulation shader',
    code: `
struct Params {
  viewport: vec4<f32>,
  transform: vec4<f32>,
  source: vec4<f32>,
  destination: vec4<f32>,
  settings: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> points: array<u32>;
@group(0) @binding(1) var<storage, read> calls: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> visibility: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) density: vec4<f32>,
};

fn world_point(index: u32) -> vec2<f32> {
  let packed = points[index];
  return vec2<f32>(
    f32(packed & 0xffffu) / 65535.0 * params.transform.z,
    f32(packed >> 16u) / 65535.0 * params.transform.w,
  );
}

fn screen_point(index: u32) -> vec2<f32> {
  return world_point(index) * params.viewport.z + params.transform.xy;
}

fn safe_normal(start: vec2<f32>, end: vec2<f32>) -> vec2<f32> {
  let delta = end - start;
  return vec2<f32>(-delta.y, delta.x) / max(length(delta), 0.000001);
}

fn joined_offset(previous: vec2<f32>, current: vec2<f32>, next: vec2<f32>, half_width: f32) -> vec2<f32> {
  let incoming = safe_normal(previous, current);
  let outgoing = safe_normal(current, next);
  let sum = incoming + outgoing;
  if length(sum) <= 0.000001 { return outgoing * half_width; }
  let miter = normalize(sum);
  let denominator = max(abs(dot(miter, outgoing)), 0.25);
  return miter * min(half_width / denominator, half_width * 2.5);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) call_index: u32) -> VertexOutput {
  let call = calls[call_index];
  let mode = visibility[call_index];
  let segment = vertex_index / 6u;
  var output: VertexOutput;
  if mode == 0u || segment + 1u >= call.y {
    output.position = vec4<f32>(2.0, 2.0, 0.0, 1.0);
    output.density = vec4<f32>(0.0);
    return output;
  }

  let point_start = call.x;
  let start = screen_point(point_start + segment);
  let end = screen_point(point_start + segment + 1u);
  var width_scale = 1.0;
  var deposited = params.settings.y;
  if mode == 2u {
    width_scale = 1.35;
    deposited = max(deposited * 4.0, params.settings.z);
  } else if mode == 3u {
    width_scale = 2.4;
    deposited = params.settings.w;
  }
  let half_width = params.settings.x * width_scale;
  var start_offset = safe_normal(start, end) * half_width;
  if segment > 0u {
    start_offset = joined_offset(screen_point(point_start + segment - 1u), start, end, half_width);
  }
  var end_offset = safe_normal(start, end) * half_width;
  if segment + 2u < call.y {
    end_offset = joined_offset(start, end, screen_point(point_start + segment + 2u), half_width);
  }

  let corner = vertex_index % 6u;
  var point = start + start_offset;
  var amount = f32(segment) / f32(max(call.y - 1u, 1u));
  if corner == 1u || corner == 4u {
    point = start - start_offset;
  } else if corner == 2u || corner == 3u {
    point = end + end_offset;
    amount = f32(segment + 1u) / f32(max(call.y - 1u, 1u));
  } else if corner == 5u {
    point = end - end_offset;
    amount = f32(segment + 1u) / f32(max(call.y - 1u, 1u));
  }

  let linear_color = mix(params.source.rgb, params.destination.rgb, amount);
  output.position = vec4<f32>(
    point.x / params.viewport.x * 2.0 - 1.0,
    1.0 - point.y / params.viewport.y * 2.0,
    0.0,
    1.0,
  );
  output.density = vec4<f32>(linear_color * deposited, deposited);
  return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  return input.density;
}`
  });
  const resolveShader = device.createShaderModule({
    label: 'Code Atlas interactive optical-density resolve shader',
    code: `
@group(0) @binding(0) var density_texture: texture_2d<f32>;

struct VertexOutput { @builtin(position) position: vec4<f32> };

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
  );
  var output: VertexOutput;
  output.position = vec4<f32>(positions[index], 0.0, 1.0);
  return output;
}

fn linear_to_srgb(value: f32) -> f32 {
  if value <= 0.0031308 { return value * 12.92; }
  return 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  let deposited = textureLoad(density_texture, vec2<i32>(input.position.xy), 0);
  if deposited.a <= 0.000001 { return vec4<f32>(0.0); }
  let alpha = 1.0 - exp(-deposited.a);
  let linear = clamp(deposited.rgb / deposited.a, vec3<f32>(0.0), vec3<f32>(1.0));
  let srgb = vec3<f32>(linear_to_srgb(linear.r), linear_to_srgb(linear.g), linear_to_srgb(linear.b));
  return vec4<f32>(srgb * alpha, alpha);
}`
  });

  async function assertShaderCompiles(module, label) {
    const information = await module.getCompilationInfo();
    const errors = information.messages.filter(message => message.type === 'error');
    if (!errors.length) return;
    const details = errors
      .map(message => `${message.lineNum}:${message.linePos} ${message.message}`)
      .join('; ');
    throw new Error(`${label} shader failed: ${details}`);
  }
  await assertShaderCompiles(accumulationShader, 'WebGPU accumulation');
  await assertShaderCompiles(resolveShader, 'WebGPU resolve');

  const accumulationPipeline = await device.createRenderPipelineAsync({
    label: 'Code Atlas interactive optical-density accumulation pipeline',
    layout: 'auto',
    vertex: { module: accumulationShader, entryPoint: 'vs_main' },
    fragment: {
      module: accumulationShader,
      entryPoint: 'fs_main',
      targets: [{
        format: 'rgba16float',
        blend: {
          color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }
        }
      }]
    },
    primitive: { topology: 'triangle-list' },
    multisample: { count: 4 }
  });
  const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  const resolvePipeline = await device.createRenderPipelineAsync({
    label: 'Code Atlas interactive optical-density resolve pipeline',
    layout: 'auto',
    vertex: { module: resolveShader, entryPoint: 'vs_main' },
    fragment: { module: resolveShader, entryPoint: 'fs_main', targets: [{ format: canvasFormat }] },
    primitive: { topology: 'triangle-list' }
  });
  const accumulationBindGroup = device.createBindGroup({
    label: 'Code Atlas interactive call data',
    layout: accumulationPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: pointBuffer } },
      { binding: 1, resource: { buffer: callBuffer } },
      { binding: 2, resource: { buffer: visibilityBuffer } },
      { binding: 3, resource: { buffer: uniformBuffer } }
    ]
  });

  let densityTexture = null;
  let densityMsaaTexture = null;
  let resolveBindGroup = null;
  let configuredWidth = 0;
  let configuredHeight = 0;
  let destroyed = false;
  device.lost.then(info => {
    if (!destroyed) onFailure(new Error(`WebGPU device lost: ${info.message || info.reason}`));
  });

  function srgbToLinear(value) {
    value /= 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  }

  function ensureTargets(width, height) {
    if (width === configuredWidth && height === configuredHeight && densityTexture) return;
    if (densityTexture) densityTexture.destroy();
    if (densityMsaaTexture) densityMsaaTexture.destroy();
    configuredWidth = width;
    configuredHeight = height;
    context.configure({ device, format: canvasFormat, alphaMode: 'premultiplied' });
    densityTexture = device.createTexture({
      label: 'Code Atlas interactive optical-density texture',
      size: [width, height],
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
    });
    densityMsaaTexture = device.createTexture({
      label: 'Code Atlas multisampled interactive optical-density texture',
      size: [width, height],
      sampleCount: 4,
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT
    });
    resolveBindGroup = device.createBindGroup({
      label: 'Code Atlas interactive density resolve input',
      layout: resolvePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: densityTexture.createView() }]
    });
  }

  return {
    render(options) {
      if (destroyed) return;
      ensureTargets(options.width, options.height);
      device.queue.writeBuffer(visibilityBuffer, 0, options.modes);
      const source = options.source.map(srgbToLinear);
      const target = options.target.map(srgbToLinear);
      const baseDensity = -Math.log(Math.max(1e-6, 1 - options.alpha));
      const parameters = new Float32Array([
        options.width, options.height, options.scale * options.dpr, options.dpr,
        options.x * options.dpr, options.y * options.dpr, data.width, data.height,
        source[0], source[1], source[2], 1,
        target[0], target[1], target[2], 1,
        Math.max(0.35, options.callWidth * options.scale * options.dpr) * 0.5,
        baseDensity,
        -Math.log(1 - 0.26),
        -Math.log(1 - 0.96)
      ]);
      device.queue.writeBuffer(uniformBuffer, 0, parameters);
      const encoder = device.createCommandEncoder({ label: 'Code Atlas interactive optical-density frame' });
      const densityPass = encoder.beginRenderPass({
        label: 'Code Atlas interactive density accumulation',
        colorAttachments: [{
          view: densityMsaaTexture.createView(),
          resolveTarget: densityTexture.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store'
        }]
      });
      densityPass.setPipeline(accumulationPipeline);
      densityPass.setBindGroup(0, accumulationBindGroup);
      densityPass.draw(maximumSegments * 6, data.calls.length);
      densityPass.end();
      const resolvePass = encoder.beginRenderPass({
        label: 'Code Atlas interactive density resolve',
        colorAttachments: [{
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store'
        }]
      });
      resolvePass.setPipeline(resolvePipeline);
      resolvePass.setBindGroup(0, resolveBindGroup);
      resolvePass.draw(3);
      resolvePass.end();
      device.queue.submit([encoder.finish()]);
    },
    destroy() {
      destroyed = true;
      if (densityTexture) densityTexture.destroy();
      if (densityMsaaTexture) densityMsaaTexture.destroy();
      pointBuffer.destroy();
      callBuffer.destroy();
      visibilityBuffer.destroy();
      uniformBuffer.destroy();
      device.destroy();
    }
  };
}
