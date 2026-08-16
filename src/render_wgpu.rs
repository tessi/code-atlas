use std::{fs, path::Path, sync::mpsc, time::Instant};

use anyhow::{Context, Result, anyhow, ensure};
use bytemuck::{Pod, Zeroable};
use tiny_skia::{IntSize, Pixmap};
use wgpu::util::DeviceExt;

use crate::{
    layout::{LayoutOptions, call_control_points},
    model::Atlas,
    render::{
        CallTile, CallTileViewport, Palette, RenderOptions, RenderStats, draw_labels,
        effective_call_opacity, optical_density_description, plan_call_tiles,
        render_software_base_png,
    },
};

const DENSITY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const SAMPLE_COUNT: u32 = 4;
const SPLINE_COMPUTE_SHADER: &str = include_str!("spline_density.wgsl");
const SPLINE_RENDER_SHADER: &str = include_str!("spline_render.wgsl");

const RESOLVE_SHADER: &str = r#"
@group(0) @binding(0)
var density_texture: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

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
    if value <= 0.0031308 {
        return value * 12.92;
    }
    return 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(input.position.xy);
    let deposited = textureLoad(density_texture, pixel, 0);
    if deposited.a <= 0.000001 {
        return vec4<f32>(0.0);
    }
    let alpha = 1.0 - exp(-deposited.a);
    let linear = clamp(deposited.rgb / deposited.a, vec3<f32>(0.0), vec3<f32>(1.0));
    let srgb = vec3<f32>(
        linear_to_srgb(linear.r),
        linear_to_srgb(linear.g),
        linear_to_srgb(linear.b),
    );
    return vec4<f32>(srgb * alpha, alpha);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuControlPoint {
    position: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuCallRecord {
    control_offset: u32,
    control_count: u32,
    segment_count: u32,
    seed: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTileInstance {
    call_index: u32,
    pass_index: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSplineParams {
    viewport: [f32; 4],
    source: [f32; 4],
    target: [f32; 4],
    pass0: [f32; 4],
    pass1: [f32; 4],
    settings: [f32; 4],
    counts: [u32; 4],
}

pub(crate) struct GpuSplineScene {
    controls: wgpu::Buffer,
    calls: wgpu::Buffer,
    call_ids: Vec<u64>,
    segment_counts: Vec<u32>,
    max_segments: u32,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct GpuPassTimings {
    pub spline_evaluation_ms: Option<f64>,
    pub rasterization_ms: Option<f64>,
}

pub(crate) struct WgpuDensityRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    spline_compute_pipeline: wgpu::ComputePipeline,
    accumulate_pipeline: wgpu::RenderPipeline,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    accumulate_bind_group_layout: wgpu::BindGroupLayout,
    resolve_pipeline: wgpu::RenderPipeline,
    resolve_bind_group_layout: wgpu::BindGroupLayout,
    timestamp_queries: bool,
}

fn create_spline_bind_group_layout(
    device: &wgpu::Device,
    sampled_points_read_only: bool,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayout {
    let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(if sampled_points_read_only {
            "code-atlas GPU spline render bind group layout"
        } else {
            "code-atlas GPU spline compute bind group layout"
        }),
        entries: &[
            storage(0, true),
            storage(1, true),
            storage(2, true),
            storage(3, sampled_points_read_only),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

impl WgpuDensityRenderer {
    pub(crate) fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .context("no compatible GPU adapter is available for the wgpu backend")?;
        let density_features = adapter.get_texture_format_features(DENSITY_FORMAT);
        ensure!(
            density_features
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::BLENDABLE),
            "the GPU adapter does not support blendable RGBA16Float density textures"
        );
        ensure!(
            density_features
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_X4)
                && density_features
                    .flags
                    .contains(wgpu::TextureFormatFeatureFlags::MULTISAMPLE_RESOLVE),
            "the GPU adapter does not support 4x multisampled RGBA16Float density textures"
        );
        let adapter_info = adapter.get_info();
        let timestamp_queries = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let required_features = if timestamp_queries {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("code-atlas optical-density device"),
            required_features,
            ..Default::default()
        }))
        .context("cannot create the wgpu optical-density device")?;

        let spline_compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("code-atlas GPU spline compute shader"),
            source: wgpu::ShaderSource::Wgsl(SPLINE_COMPUTE_SHADER.into()),
        });
        let spline_render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("code-atlas GPU spline render shader"),
            source: wgpu::ShaderSource::Wgsl(SPLINE_RENDER_SHADER.into()),
        });
        let compute_bind_group_layout =
            create_spline_bind_group_layout(&device, false, wgpu::ShaderStages::COMPUTE);
        let accumulate_bind_group_layout =
            create_spline_bind_group_layout(&device, true, wgpu::ShaderStages::VERTEX);
        let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("code-atlas GPU spline compute layout"),
            bind_group_layouts: &[Some(&compute_bind_group_layout)],
            immediate_size: 0,
        });
        let spline_compute_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("code-atlas GPU spline evaluation pipeline"),
                layout: Some(&compute_layout),
                module: &spline_compute_shader,
                entry_point: Some("compute_points"),
                compilation_options: Default::default(),
                cache: None,
            });
        let accumulate_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("code-atlas optical-density accumulation layout"),
            bind_group_layouts: &[Some(&accumulate_bind_group_layout)],
            immediate_size: 0,
        });
        let accumulate_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("code-atlas optical-density accumulation pipeline"),
            layout: Some(&accumulate_layout),
            vertex: wgpu::VertexState {
                module: &spline_render_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: SAMPLE_COUNT,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &spline_render_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: DENSITY_FORMAT,
                    blend: Some(wgpu::BlendState::ADDITIVE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let resolve_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("code-atlas optical-density resolve bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
        let resolve_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("code-atlas optical-density resolve shader"),
            source: wgpu::ShaderSource::Wgsl(RESOLVE_SHADER.into()),
        });
        let resolve_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("code-atlas optical-density resolve layout"),
            bind_group_layouts: &[Some(&resolve_bind_group_layout)],
            immediate_size: 0,
        });
        let resolve_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("code-atlas optical-density resolve pipeline"),
            layout: Some(&resolve_layout),
            vertex: wgpu::VertexState {
                module: &resolve_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &resolve_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: OUTPUT_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            device,
            queue,
            adapter_name: format!("{} ({:?})", adapter_info.name, adapter_info.backend),
            spline_compute_pipeline,
            accumulate_pipeline,
            compute_bind_group_layout,
            accumulate_bind_group_layout,
            resolve_pipeline,
            resolve_bind_group_layout,
            timestamp_queries,
        })
    }

    pub(crate) fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    pub(crate) fn max_texture_dimension(&self) -> u32 {
        self.device.limits().max_texture_dimension_2d
    }

    pub(crate) fn prepare_scene(&self, atlas: &Atlas) -> Result<GpuSplineScene> {
        let mut controls = Vec::new();
        let mut records = Vec::with_capacity(atlas.calls.len());
        let mut call_ids = Vec::with_capacity(atlas.calls.len());
        let mut segment_counts = Vec::with_capacity(atlas.calls.len());
        let mut max_segments = 64_u32;
        for call in &atlas.calls {
            let control_points = call_control_points(atlas, call);
            ensure!(
                control_points.len() >= 2,
                "call {} has fewer than two hierarchy control points",
                call.id
            );
            let control_offset = u32::try_from(controls.len())
                .context("GPU spline control-point offset exceeds u32")?;
            let control_count = u32::try_from(control_points.len())
                .context("GPU spline control-point count exceeds u32")?;
            controls.extend(control_points.into_iter().map(|point| GpuControlPoint {
                position: [point.x as f32, point.y as f32],
            }));
            let degree = control_count.saturating_sub(1).min(3);
            let segment_count = control_count
                .saturating_sub(degree)
                .saturating_mul(16)
                .max(64);
            max_segments = max_segments.max(segment_count);
            let seed = (call.id as u32) ^ ((call.id >> 32) as u32).rotate_left(13) ^ 0xca11_517e;
            records.push(GpuCallRecord {
                control_offset,
                control_count,
                segment_count,
                seed,
            });
            call_ids.push(call.id);
            segment_counts.push(segment_count);
        }
        if controls.is_empty() {
            controls.push(GpuControlPoint {
                position: [0.0, 0.0],
            });
        }
        if records.is_empty() {
            records.push(GpuCallRecord {
                control_offset: 0,
                control_count: 1,
                segment_count: 0,
                seed: 0,
            });
        }
        let limits = self.device.limits();
        let control_bytes = std::mem::size_of_val(controls.as_slice()) as u64;
        let call_bytes = std::mem::size_of_val(records.as_slice()) as u64;
        ensure!(
            control_bytes <= limits.max_storage_buffer_binding_size
                && call_bytes <= limits.max_storage_buffer_binding_size,
            "GPU spline scene exceeds the adapter's storage-buffer binding limit"
        );
        let controls = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("code-atlas persistent hierarchy control points"),
                contents: bytemuck::cast_slice(&controls),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let calls = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("code-atlas persistent compact call records"),
                contents: bytemuck::cast_slice(&records),
                usage: wgpu::BufferUsages::STORAGE,
            });
        Ok(GpuSplineScene {
            controls,
            calls,
            call_ids,
            segment_counts,
            max_segments,
        })
    }

    pub(crate) fn render_call_tile(
        &self,
        scene: &GpuSplineScene,
        layout: &LayoutOptions,
        render: &RenderOptions,
        call_indices: &[usize],
        viewport: CallTileViewport,
    ) -> Result<(Pixmap, usize)> {
        let (pixmap, calls, _) =
            self.render_call_tile_profiled(scene, layout, render, call_indices, viewport)?;
        Ok((pixmap, calls))
    }

    pub(crate) fn render_call_tile_profiled(
        &self,
        scene: &GpuSplineScene,
        layout: &LayoutOptions,
        render: &RenderOptions,
        call_indices: &[usize],
        viewport: CallTileViewport,
    ) -> Result<(Pixmap, usize, GpuPassTimings)> {
        ensure!(
            viewport.width > 0 && viewport.height > 0,
            "GPU call tile dimensions must be non-zero"
        );
        let max_dimension = self.device.limits().max_texture_dimension_2d;
        ensure!(
            viewport.width <= max_dimension && viewport.height <= max_dimension,
            "GPU call tile {}x{} exceeds the adapter's {}-pixel texture limit",
            viewport.width,
            viewport.height,
            max_dimension
        );
        let extent = wgpu::Extent3d {
            width: viewport.width,
            height: viewport.height,
            depth_or_array_layers: 1,
        };
        let density_msaa = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("code-atlas multisampled optical-density tile"),
            size: extent,
            mip_level_count: 1,
            sample_count: SAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format: DENSITY_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let density_resolved = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("code-atlas resolved optical-density tile"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DENSITY_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let output = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("code-atlas resolved call tile"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OUTPUT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let density_msaa_view = density_msaa.create_view(&wgpu::TextureViewDescriptor::default());
        let density_resolved_view =
            density_resolved.create_view(&wgpu::TextureViewDescriptor::default());
        let output_view = output.create_view(&wgpu::TextureViewDescriptor::default());

        let opacity = effective_call_opacity(scene.call_ids.len(), render);
        let mut ordered_indices = call_indices.to_vec();
        ordered_indices.sort_unstable_by_key(|index| {
            (
                scene.call_ids.get(*index).copied().unwrap_or(u64::MAX),
                *index,
            )
        });
        let mut instances = Vec::with_capacity(ordered_indices.len().saturating_mul(2));
        let mut calls_drawn = 0;
        for index in ordered_indices {
            let Some(&segment_count) = scene.segment_counts.get(index) else {
                continue;
            };
            if segment_count == 0 {
                continue;
            }
            let call_index = u32::try_from(index).context("GPU call index exceeds u32")?;
            instances.push(GpuTileInstance {
                call_index,
                pass_index: 0,
            });
            instances.push(GpuTileInstance {
                call_index,
                pass_index: 1,
            });
            calls_drawn += 1;
        }
        self.clear_density_target(&density_msaa_view, &density_resolved_view);
        let batch_capacity = self.instance_batch_capacity(scene)?;
        let mut pass_timings = GpuPassTimings::default();
        for batch in instances.chunks(batch_capacity) {
            let batch_timings = self.submit_spline_instances(
                scene,
                layout,
                render,
                opacity,
                viewport,
                batch,
                &density_msaa_view,
                &density_resolved_view,
            )?;
            pass_timings.spline_evaluation_ms = sum_optional(
                pass_timings.spline_evaluation_ms,
                batch_timings.spline_evaluation_ms,
            );
            pass_timings.rasterization_ms = sum_optional(
                pass_timings.rasterization_ms,
                batch_timings.rasterization_ms,
            );
        }

        let resolve_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("code-atlas optical-density resolve bind group"),
            layout: &self.resolve_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&density_resolved_view),
            }],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("code-atlas optical-density resolve encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("code-atlas optical-density resolve pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.resolve_pipeline);
            pass.set_bind_group(0, &resolve_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));

        let pixels = read_texture(&self.device, &self.queue, &output, extent)?;
        let size = IntSize::from_wh(viewport.width, viewport.height)
            .context("invalid GPU call tile dimensions")?;
        let pixmap = Pixmap::from_vec(pixels, size).context("invalid GPU call tile pixels")?;
        Ok((pixmap, calls_drawn, pass_timings))
    }

    fn instance_batch_capacity(&self, scene: &GpuSplineScene) -> Result<usize> {
        let point_stride = u64::from(scene.max_segments + 1);
        let bytes_per_instance = point_stride
            .checked_mul(std::mem::size_of::<GpuControlPoint>() as u64)
            .context("GPU sampled-point stride overflows")?;
        let limits = self.device.limits();
        let storage_capacity = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size)
            / bytes_per_instance.max(1);
        let dispatch_capacity = u64::from(limits.max_compute_workgroups_per_dimension) * 8;
        let capacity = storage_capacity
            .min(dispatch_capacity)
            .min(usize::MAX as u64) as usize;
        let capacity = capacity & !1;
        ensure!(
            capacity >= 2,
            "GPU storage limits cannot hold one two-pass spline instance"
        );
        Ok(capacity)
    }

    fn clear_density_target(
        &self,
        multisampled_view: &wgpu::TextureView,
        resolved_view: &wgpu::TextureView,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("code-atlas optical-density clear encoder"),
            });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("code-atlas optical-density clear pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: multisampled_view,
                    depth_slice: None,
                    resolve_target: Some(resolved_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        self.queue.submit(Some(encoder.finish()));
    }

    #[allow(clippy::too_many_arguments)]
    fn submit_spline_instances(
        &self,
        scene: &GpuSplineScene,
        layout: &LayoutOptions,
        render: &RenderOptions,
        opacity: u8,
        viewport: CallTileViewport,
        instances: &[GpuTileInstance],
        multisampled_view: &wgpu::TextureView,
        resolved_view: &wgpu::TextureView,
    ) -> Result<GpuPassTimings> {
        if instances.is_empty() {
            return Ok(GpuPassTimings::default());
        }
        let point_stride = scene.max_segments + 1;
        let point_count = u64::try_from(instances.len())
            .context("GPU tile instance count exceeds u64")?
            .checked_mul(u64::from(point_stride))
            .context("GPU sampled-point count overflows")?;
        let point_bytes = point_count
            .checked_mul(std::mem::size_of::<GpuControlPoint>() as u64)
            .context("GPU sampled-point buffer size overflows")?;
        let instance_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("code-atlas tile-local call/pass IDs"),
                contents: bytemuck::cast_slice(instances),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let sampled_points = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("code-atlas GPU-evaluated spline points"),
            size: point_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let palette = Palette::for_theme(render.theme);
        let drawing_scale = (layout.width.min(layout.height) as f32 / 1080.0).max(0.5);
        let pass = |alpha_scale: f32, width_scale: f32, amplitude: f32| {
            let alpha = (f32::from(opacity) * alpha_scale).round().max(1.0) / 255.0;
            [
                render.call_width * drawing_scale * width_scale * 0.5,
                -(1.0 - alpha.min(0.999_999)).ln(),
                amplitude,
                0.0,
            ]
        };
        let params = GpuSplineParams {
            viewport: [
                viewport.x as f32,
                viewport.y as f32,
                viewport.width as f32,
                viewport.height as f32,
            ],
            source: linear_rgba(palette.direction_source),
            target: linear_rgba(palette.direction_target),
            pass0: pass(0.62, 0.82, 0.14),
            pass1: pass(0.38, 1.28, 0.34),
            settings: [layout.bundle_strength.clamp(0.0, 1.0) as f32, 0.0, 0.0, 0.0],
            counts: [
                point_stride,
                u32::try_from(instances.len()).context("GPU tile instance count exceeds u32")?,
                scene.max_segments,
                0,
            ],
        };
        let params_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("code-atlas GPU spline tile parameters"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let compute_bind_group = self.create_spline_bind_group(
            &self.compute_bind_group_layout,
            scene,
            &instance_buffer,
            &sampled_points,
            &params_buffer,
        );
        let render_bind_group = self.create_spline_bind_group(
            &self.accumulate_bind_group_layout,
            scene,
            &instance_buffer,
            &sampled_points,
            &params_buffer,
        );
        let timestamp_resources = self.timestamp_queries.then(|| {
            let query_set = self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("code-atlas GPU spline phase timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count: 4,
            });
            let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("code-atlas GPU timestamp resolve"),
                size: 32,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("code-atlas GPU timestamp readback"),
                size: 32,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            (query_set, resolve, readback)
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("code-atlas GPU spline evaluation and density encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("code-atlas GPU spline evaluation pass"),
                timestamp_writes: timestamp_resources.as_ref().map(|(query_set, _, _)| {
                    wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }
                }),
            });
            pass.set_pipeline(&self.spline_compute_pipeline);
            pass.set_bind_group(0, &compute_bind_group, &[]);
            pass.dispatch_workgroups(
                point_stride.div_ceil(8),
                u32::try_from(instances.len())
                    .context("GPU tile instance count exceeds u32")?
                    .div_ceil(8),
                1,
            );
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("code-atlas GPU spline optical-density pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: multisampled_view,
                    depth_slice: None,
                    resolve_target: Some(resolved_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: timestamp_resources.as_ref().map(|(query_set, _, _)| {
                    wgpu::RenderPassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(2),
                        end_of_pass_write_index: Some(3),
                    }
                }),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.accumulate_pipeline);
            pass.set_bind_group(0, &render_bind_group, &[]);
            pass.draw(
                0..scene.max_segments.saturating_mul(6),
                0..u32::try_from(instances.len()).context("GPU instance count exceeds u32")?,
            );
        }
        if let Some((query_set, resolve, readback)) = &timestamp_resources {
            encoder.resolve_query_set(query_set, 0..4, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, 32);
        }
        self.queue.submit(Some(encoder.finish()));
        if let Some((_, _, readback)) = timestamp_resources {
            let values = read_u64_buffer(&self.device, &readback)?;
            let period = f64::from(self.queue.get_timestamp_period()) / 1_000_000.0;
            return Ok(GpuPassTimings {
                spline_evaluation_ms: Some(values[1].saturating_sub(values[0]) as f64 * period),
                rasterization_ms: Some(values[3].saturating_sub(values[2]) as f64 * period),
            });
        }
        Ok(GpuPassTimings::default())
    }

    fn create_spline_bind_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        scene: &GpuSplineScene,
        instances: &wgpu::Buffer,
        sampled_points: &wgpu::Buffer,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("code-atlas GPU spline bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene.controls.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scene.calls.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sampled_points.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }
}

pub(crate) fn render_wgpu_png(
    atlas: &Atlas,
    output: &Path,
    layout: &LayoutOptions,
    render: &RenderOptions,
) -> Result<RenderStats> {
    let started = Instant::now();
    let mut stats = render_software_base_png(atlas, output, layout, render)?;
    let mut pixmap = Pixmap::load_png(output)
        .with_context(|| format!("cannot reload software base {}", output.display()))?;
    let renderer = WgpuDensityRenderer::new()?;
    let scene = renderer.prepare_scene(atlas)?;
    let (tiles, calls_drawn, overlap) = plan_call_tiles(atlas, layout, render)?;
    for tile in &tiles {
        let (call_tile, _) = render_tile(&renderer, &scene, layout, render, tile)?;
        composite_tile_core(&mut pixmap, &call_tile, tile);
    }
    let labels = draw_labels(&mut pixmap, atlas, layout, Palette::for_theme(render.theme))?;
    pixmap
        .save_png(output)
        .with_context(|| format!("cannot save {}", output.display()))?;
    stats.backend = "wgpu spline-evaluated optical-density tiled".to_owned();
    stats.gpu_adapter = Some(renderer.adapter_name().to_owned());
    stats.call_layer_tiles = tiles.len();
    stats.call_layer_tile_size = Some(render.pdf_call_tile_size);
    stats.call_layer_overlap = Some(overlap);
    stats.calls_drawn = calls_drawn;
    stats.file_labels_drawn = labels.files;
    stats.directory_labels_drawn = labels.directories;
    stats.output_bytes = fs::metadata(output)?.len();
    stats.effective_call_opacity = effective_call_opacity(atlas.calls.len(), render);
    stats.call_compositing = optical_density_description();
    stats.render_ms = started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
    Ok(stats)
}

pub(crate) fn render_wgpu_call_layer_png(
    atlas: &Atlas,
    output: &Path,
    layout: &LayoutOptions,
    render: &RenderOptions,
) -> Result<RenderStats> {
    let started = Instant::now();
    let renderer = WgpuDensityRenderer::new()?;
    let scene = renderer.prepare_scene(atlas)?;
    let (tiles, calls_drawn, overlap) = plan_call_tiles(atlas, layout, render)?;
    let mut pixmap = Pixmap::new(layout.width, layout.height)
        .context("output dimensions are too large for the GPU call renderer")?;
    for tile in &tiles {
        let (call_tile, _) = render_tile(&renderer, &scene, layout, render, tile)?;
        composite_tile_core(&mut pixmap, &call_tile, tile);
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("cannot create output directory {}", parent.display()))?;
    }
    pixmap
        .save_png(output)
        .with_context(|| format!("cannot save {}", output.display()))?;
    let palette = Palette::for_theme(render.theme);
    Ok(RenderStats {
        backend: "wgpu spline-evaluated optical-density tiled".to_owned(),
        gpu_adapter: Some(renderer.adapter_name().to_owned()),
        width: layout.width,
        height: layout.height,
        call_layer_width: layout.width,
        call_layer_height: layout.height,
        call_layer_dpi: None,
        call_layer_tiles: tiles.len(),
        call_layer_tile_size: Some(render.pdf_call_tile_size),
        call_layer_overlap: Some(overlap),
        files_drawn: atlas.files().count(),
        file_labels_drawn: 0,
        directory_labels_drawn: 0,
        calls_drawn,
        directory_boundaries_drawn: atlas
            .nodes
            .iter()
            .filter(|node| node.kind == crate::model::NodeKind::Directory && node.id != 0)
            .count(),
        output_bytes: fs::metadata(output)?.len(),
        effective_call_opacity: effective_call_opacity(atlas.calls.len(), render),
        call_compositing: optical_density_description(),
        direction_source_color: palette.direction_source.hex(),
        direction_target_color: palette.direction_target.hex(),
        render_ms: started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64,
        pdf_tile_cache: None,
    })
}

fn render_tile(
    renderer: &WgpuDensityRenderer,
    scene: &GpuSplineScene,
    layout: &LayoutOptions,
    render: &RenderOptions,
    tile: &CallTile,
) -> Result<(Pixmap, usize)> {
    renderer.render_call_tile(
        scene,
        layout,
        render,
        &tile.call_indices,
        CallTileViewport {
            x: tile.render_x,
            y: tile.render_y,
            width: tile.render_width,
            height: tile.render_height,
        },
    )
}

fn linear_rgba(color: crate::render::Rgba) -> [f32; 4] {
    let convert = |value: u8| {
        let value = f32::from(value) / 255.0;
        if value <= 0.04045 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    };
    [
        convert(color.r),
        convert(color.g),
        convert(color.b),
        f32::from(color.a) / 255.0,
    ]
}

fn sum_optional(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left + right),
        (None, Some(right)) => Some(right),
        (Some(left), None) => Some(left),
        (None, None) => None,
    }
}

fn composite_tile_core(destination: &mut Pixmap, source: &Pixmap, tile: &CallTile) {
    for y in tile.core_y..tile.core_y + tile.core_height {
        for x in tile.core_x..tile.core_x + tile.core_width {
            let source_x = x - tile.render_x;
            let source_y = y - tile.render_y;
            let source_offset =
                (source_y as usize * source.width() as usize + source_x as usize) * 4;
            let destination_offset = (y as usize * destination.width() as usize + x as usize) * 4;
            let source_pixel = &source.data()[source_offset..source_offset + 4];
            let destination_pixel =
                &mut destination.data_mut()[destination_offset..destination_offset + 4];
            let inverse_alpha = 255_u32 - u32::from(source_pixel[3]);
            for channel in 0..3 {
                destination_pixel[channel] = (u32::from(source_pixel[channel])
                    + (u32::from(destination_pixel[channel]) * inverse_alpha + 127) / 255)
                    .min(255) as u8;
            }
            destination_pixel[3] = (u32::from(source_pixel[3])
                + (u32::from(destination_pixel[3]) * inverse_alpha + 127) / 255)
                .min(255) as u8;
        }
    }
}

fn read_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    extent: wgpu::Extent3d,
) -> Result<Vec<u8>> {
    let unpadded_bytes_per_row = extent.width * 4;
    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(alignment) * alignment;
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("code-atlas optical-density readback buffer"),
        size: u64::from(padded_bytes_per_row) * u64::from(extent.height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("code-atlas optical-density readback encoder"),
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(extent.height),
            },
        },
        extent,
    );
    queue.submit(Some(encoder.finish()));

    let slice = output_buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| anyhow!("GPU optical-density readback poll failed: {error}"))?;
    receiver
        .recv()
        .context("GPU optical-density readback callback was dropped")?
        .context("cannot map the GPU optical-density readback buffer")?;

    let mapped = slice
        .get_mapped_range()
        .context("cannot access the GPU optical-density readback buffer")?;
    let mut pixels = Vec::with_capacity((unpadded_bytes_per_row * extent.height) as usize);
    for row in mapped.chunks_exact(padded_bytes_per_row as usize) {
        pixels.extend_from_slice(&row[..unpadded_bytes_per_row as usize]);
    }
    drop(mapped);
    output_buffer.unmap();
    Ok(pixels)
}

fn read_u64_buffer(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<[u64; 4]> {
    let slice = buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| anyhow!("GPU timestamp readback poll failed: {error}"))?;
    receiver
        .recv()
        .context("GPU timestamp readback callback was dropped")?
        .context("cannot map GPU timestamps")?;
    let mapped = slice
        .get_mapped_range()
        .context("cannot access GPU timestamps")?;
    let values = bytemuck::try_from_bytes::<[u64; 4]>(&mapped)
        .map_err(|error| anyhow!("GPU timestamp buffer has an invalid size: {error:?}"))?
        .to_owned();
    drop(mapped);
    buffer.unmap();
    Ok(values)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf};

    use crate::{
        model::{AnalyzerReport, Atlas, BuildTimings, Callsite, Node, NodeKind, Rect},
        render::{Theme, plan_call_tiles, render_software_call_tile},
    };

    use super::*;

    #[test]
    fn gpu_density_preserves_order_overlap_reference_and_tile_invariants() {
        let Ok(renderer) = WgpuDensityRenderer::new() else {
            eprintln!("skipping GPU density invariants because no compatible adapter is available");
            return;
        };
        let (atlas, layout) = fixture();
        let scene = renderer.prepare_scene(&atlas).unwrap();
        let indices = [0, 1, 2];
        let viewport = CallTileViewport {
            x: 0,
            y: 0,
            width: layout.width,
            height: layout.height,
        };
        for theme in [Theme::Architect, Theme::Night] {
            let render = RenderOptions {
                theme,
                pdf_call_tile_size: 64,
                ..RenderOptions::default()
            };
            let (forward, _) = renderer
                .render_call_tile(&scene, &layout, &render, &indices, viewport)
                .unwrap();
            let (reverse, _) = renderer
                .render_call_tile(&scene, &layout, &render, &[2, 1, 0], viewport)
                .unwrap();
            assert_eq!(
                forward.data(),
                reverse.data(),
                "GPU density resolve depends on input order for {theme}"
            );

            let (single, _) = renderer
                .render_call_tile(&scene, &layout, &render, &[0], viewport)
                .unwrap();
            let (double, _) = renderer
                .render_call_tile(&scene, &layout, &render, &[0, 1], viewport)
                .unwrap();
            assert!(
                alpha_sum(&double) > alpha_sum(&single),
                "repeated GPU routes do not increase optical density for {theme}"
            );

            let (cpu, _) =
                render_software_call_tile(&atlas, &layout, &render, &indices, viewport).unwrap();
            assert_reference_similarity(&cpu, &forward, theme);
            assert_gpu_tiles_match(&renderer, &scene, &atlas, &layout, &render, &forward);
        }
    }

    fn assert_reference_similarity(cpu: &Pixmap, gpu: &Pixmap, theme: Theme) {
        let mut alpha_difference = 0_u64;
        let mut cpu_alpha = 0_u64;
        let mut intersection = 0_u64;
        let mut union = 0_u64;
        for (cpu, gpu) in cpu.data().chunks_exact(4).zip(gpu.data().chunks_exact(4)) {
            alpha_difference += u64::from(cpu[3].abs_diff(gpu[3]));
            cpu_alpha += u64::from(cpu[3]);
            let cpu_covered = cpu[3] >= 4;
            let gpu_covered = gpu[3] >= 4;
            intersection += u64::from(cpu_covered && gpu_covered);
            union += u64::from(cpu_covered || gpu_covered);
        }
        let relative_alpha_error = alpha_difference as f64 / cpu_alpha.max(1) as f64;
        let coverage_iou = intersection as f64 / union.max(1) as f64;
        assert!(
            relative_alpha_error < 0.45,
            "GPU/CPU relative alpha error {relative_alpha_error:.3} is too high for {theme}"
        );
        assert!(
            coverage_iou > 0.72,
            "GPU/CPU coverage IoU {coverage_iou:.3} is too low for {theme}"
        );
    }

    fn assert_gpu_tiles_match(
        renderer: &WgpuDensityRenderer,
        scene: &GpuSplineScene,
        atlas: &Atlas,
        layout: &LayoutOptions,
        render: &RenderOptions,
        full: &Pixmap,
    ) {
        let (tiles, _, _) = plan_call_tiles(atlas, layout, render).unwrap();
        assert!(tiles.len() > 1);
        for tile in &tiles {
            let (pixmap, _) = render_tile(renderer, scene, layout, render, tile).unwrap();
            for y in tile.core_y..tile.core_y + tile.core_height {
                for x in tile.core_x..tile.core_x + tile.core_width {
                    let full_offset = (y as usize * layout.width as usize + x as usize) * 4;
                    let local_x = x - tile.render_x;
                    let local_y = y - tile.render_y;
                    let tile_offset =
                        (local_y as usize * tile.render_width as usize + local_x as usize) * 4;
                    for channel in 0..4 {
                        assert!(
                            full.data()[full_offset + channel]
                                .abs_diff(pixmap.data()[tile_offset + channel])
                                <= 2,
                            "GPU tile differs at ({x}, {y}) channel {channel}"
                        );
                    }
                }
            }
        }
    }

    fn alpha_sum(pixmap: &Pixmap) -> u64 {
        pixmap
            .data()
            .chunks_exact(4)
            .map(|pixel| u64::from(pixel[3]))
            .sum()
    }

    fn fixture() -> (Atlas, LayoutOptions) {
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
        let call = |id, source, source_line, target, target_line, callee: &str| Callsite {
            id,
            source,
            source_line,
            target,
            target_line: Some(target_line),
            callee: callee.to_owned(),
            kind: "runtime".to_owned(),
            analyzer: "fixture".to_owned(),
            confidence: 1.0,
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
                call(11, 1, 2, 2, 19, "b::low"),
                call(11, 1, 2, 2, 19, "b::low"),
                call(7, 2, 3, 1, 18, "a::high"),
            ],
            path_to_id: HashMap::new(),
            report: AnalyzerReport::default(),
            timings: BuildTimings::default(),
        };
        (atlas, layout)
    }
}
