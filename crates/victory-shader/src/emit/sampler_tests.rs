//! End-to-end tests of immediate samplers chosen by control flow: a
//! synthetic GFX10 program lowered to DXIL, validated by `dxil.dll` when
//! present, and run on the host GPU and WARP.

use windows::Win32::Graphics::Direct3D12::*;

use super::host::{Dispatch, Host, Root, View};
use super::sampler::StaticSampler;
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// The console's words for two trilinear samplers that differ only in
/// their address modes, as the title's programs pick between them: wrap,
/// then clamp to the last texel (x, y and z mode 2).
const WRAP: [u32; 4] = [0x0000_0000, 0x00C0_0000, 0x1850_0000, 0];
const CLAMP: [u32; 4] = [0x0000_0092, 0x00C0_0000, 0x1850_0000, 0];

/// A wave32 compute program of 32 threads, the table of views at s0 (a 2D
/// texture at entry 0, a raw UAV at entry 2) and the group id in s1. Group
/// 0 takes the branch and builds [`WRAP`] in s[8:11], other groups fall
/// through and build [`CLAMP`]; at the join each lane samples the texture
/// at (1.25, 0.5) and stores red at dword group * 32 + lane.
const CHOSEN_BY_GROUP: &[u32] = &[
    0xBE84_0300, // s_mov_b32 s4, s0
    0xBE85_0380, // s_mov_b32 s5, 0
    0xF40C_0302,
    0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
    0xF408_0A02,
    0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
    0xBE89_03FF,
    0x00C0_0000, // s_mov_b32 s9, 0xc00000
    0xBE8A_03FF,
    0x1850_0000, // s_mov_b32 s10, 0x18500000
    0xBE8B_0380, // s_mov_b32 s11, 0
    0xBF06_8001, // s_cmp_eq_u32 s1, 0
    0xBF85_0003, // s_cbranch_scc1 -> 0x40
    0xBE88_03FF,
    0x0000_0092, // s_mov_b32 s8, 0x92
    0xBF82_0001, // s_branch -> 0x44
    0xBE88_0380, // 0x40: s_mov_b32 s8, 0
    0x7E04_02FF,
    0x3FA0_0000, // 0x44: v_mov_b32 v2, 1.25
    0x7E06_02F0, // v_mov_b32 v3, 0.5
    0x3420_0082, // v_lshlrev_b32 v16, 2, v0
    0x8F02_8701, // s_lshl_b32 s2, s1, 7
    0x4A20_2002, // v_add_nc_u32 v16, s2, v16
    0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
    0xF09C_0F08,
    0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
    0xE070_1000,
    0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
    0xBF81_0000, // s_endpgm
];

/// Lowers `words` (see [`CHOSEN_BY_GROUP`] for the bindings) as the
/// translator does, its immediate samplers made static samplers.
fn build(words: &[u32]) -> HostShader {
    let range = |kind, table_offset_dwords, dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride: 0,
        dwords,
        user_data: 0,
        slot: 0,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 32,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![
            range(ResourceType::Srv, 0, 8),
            range(ResourceType::Uav, 16, 4),
        ],
        io: StageIo::Cs(ComputeIo {
            // One user SGPR, then the group id x.
            rsrc2: 1 << 7 | 1 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        static_samplers: lowered
            .samplers
            .iter()
            .map(|&w| StaticSampler::from_words(w))
            .collect(),
        samplers: lowered.samplers,
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        table_views: Vec::new(),
        stride_table: false,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

#[test]
fn a_sampler_chosen_by_control_flow_is_a_static_sampler_per_choice() {
    let shader = build(CHOSEN_BY_GROUP);
    let mut samplers = shader.samplers.clone();
    samplers.sort_unstable();
    assert_eq!(samplers, [WRAP, CLAMP]);
}

#[test]
fn groups_sample_with_the_sampler_their_branch_chose() {
    let shader = build(CHOSEN_BY_GROUP);
    // Red 1.0 then 2.0: past the right edge, wrap reads the first texel and
    // clamp the last.
    let texels = [[1.0f32, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]];
    let unused = [[0.0f32; 4]];
    let views = [
        View::Texture(2, 1, &texels),
        View::Texture(1, 1, &unused),
        View::RawUav(64),
    ];
    // The heap's samplers are not read: the program's are static.
    let heap_samplers = [D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    }];
    let mut ran = 0;
    // One GPU test at a time, with the VALU and waterfall tests.
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let uavs = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &views,
                    samplers: &heap_samplers,
                    groups: [2, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        let red: Vec<f32> = uavs[0].iter().map(|&w| f32::from_bits(w)).collect();
        let want: Vec<f32> = (0..64).map(|i| if i < 32 { 1.0 } else { 2.0 }).collect();
        assert_eq!(red, want, "{label}");
        ran += 1;
    }
    if ran == 0 {
        eprintln!("no device ran the program");
    }
}
