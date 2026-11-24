//! End-to-end tests of the image atomics DXIL has no operation for
//! (`image_atomic_inc`, `_dec`, `_fcmpswap`, `_fmin`, `_fmax`): synthetic
//! GFX10 compute shaders lowered to DXIL, validated by `dxil.dll` when
//! present, and run on the host GPU and WARP against RDNA2's definitions.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SMEM `s_load_dwordx{4,8} s[dst:..], s[base:base+1], byte_off` (soffset = null).
const fn s_load(op: u8, dst: u8, base: u8, byte_off: u32) -> [u32; 2] {
    [
        0x3D << 26 | (op as u32) << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25 | byte_off,
    ]
}

/// VOP1 `v_mov_b32 vdst, src0` (`src0`: 0..105 = SGPR, 128 = 0, 256.. = VGPR).
const fn v_mov_b32(vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | 0x001 << 9 | (src0 as u32 & 0x1FF)
}

/// MIMG `op vdata, vaddr[2], s[srsrc:srsrc+7] dmask dim:2D [glc]`.
const fn mimg(op: u8, vdata: u8, vaddr: u8, srsrc: u8, dmask: u8, glc: bool) -> [u32; 2] {
    [
        0x3C << 26 | ((op as u32) & 0x7F) << 18 | (glc as u32) << 13 | (dmask as u32) << 8 | 1 << 3,
        (srsrc as u32 >> 2) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0`.
const fn buffer_store_dword(vdata: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

const IMAGE_ATOMIC_SWAP: u8 = 0x0F;

/// A one-thread compute shader: root constants `s0..s3` are (start, data,
/// compare, 0). It swaps `start` into texel (0, 0) of the R32_UINT texture
/// at heap index 1, runs `op` there with glc and the data (and compare, for
/// the two-dword forms), and stores the returned old value to dword 0 of
/// the raw UAV at heap index 0.
fn build(op: u8, pair: bool) -> HostShader {
    let mut words = vec![s_mov_b32(5, 0x80)]; // s[4:5]: the table at 0
    words.extend(s_load(0x02, 8, 4, 0)); // the raw UAV
    words.extend(s_load(0x03, 16, 4, 32)); // the texture
    for i in 0..4u8 {
        words.push(v_mov_b32(i + 1, u16::from(i))); // v1..v4 = s0..s3
    }
    words.push(v_mov_b32(5, 128)); // x = 0
    words.push(v_mov_b32(6, 128)); // y = 0
    words.push(v_mov_b32(7, 256 + 1)); // the start value
    words.extend(mimg(IMAGE_ATOMIC_SWAP, 7, 5, 16, 1, false));
    words.push(v_mov_b32(8, 256 + 2)); // data
    words.push(v_mov_b32(9, 256 + 3)); // compare
    words.extend(mimg(op, 8, 5, 16, if pair { 3 } else { 1 }, true));
    words.extend(buffer_store_dword(8, 8));
    words.push(S_ENDPGM);

    let used = |kind, table_offset_dwords, dwords| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: table_offset_dwords / 8,
        stride: 0,
        dwords,
        user_data: 4,
        slot: 1,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 12,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![used(ResourceType::Uav, 0, 4), used(ResourceType::Uav, 8, 8)],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
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
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        table_views: lowered.table_views,
        stride_table: lowered.stride_table,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }
    shader
}

/// Runs `op` for each (start, data, compare) and checks the old value it
/// returns and the texel it leaves against `want(start, data, compare)`.
fn check(name: &str, op: u8, pair: bool, cases: &[[u32; 3]], want: impl Fn(u32, u32, u32) -> u32) {
    let shader = build(op, pair);
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for &[start, data, compare] in cases {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => [start, data, compare, 0][u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(1), View::UintTextureUav(1, 1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            let (old, texel) = (out[0][0], out[1][0]);
            let expect = want(start, data, compare);
            assert_eq!(
                (old, texel),
                (start, expect),
                "{label}: {name}({start:#010x}, data {data:#010x}, compare {compare:#010x}) \
                 returned {old:#010x} and left {texel:#010x}, want {start:#010x} and {expect:#010x}",
            );
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn image_atomic_inc_wraps_at_the_data() {
    let cases = [
        [0, 5, 0],
        [4, 5, 0],
        [5, 5, 0],
        [9, 5, 0],
        [7, 0, 0],
        [u32::MAX - 1, u32::MAX, 0],
    ];
    check("image_atomic_inc", 0x1B, false, &cases, |old, data, _| {
        if old >= data { 0 } else { old + 1 }
    });
}

#[test]
fn image_atomic_dec_reloads_at_zero_or_above_the_data() {
    let cases = [
        [5, 9, 0],
        [1, 9, 0],
        [0, 9, 0],
        [12, 9, 0],
        [9, 9, 0],
        [0, 0, 0],
    ];
    check("image_atomic_dec", 0x1C, false, &cases, |old, data, _| {
        if old == 0 || old > data {
            data
        } else {
            old - 1
        }
    });
}

#[test]
fn image_atomic_fcmpswap_compares_as_floats() {
    let f = f32::to_bits;
    let nan = 0x7FC0_0000;
    let cases = [
        [f(1.5), f(7.0), f(1.5)], // equal: swapped
        [f(1.5), f(7.0), f(2.5)], // unequal: kept
        [0x8000_0000, f(3.0), 0], // -0 equals +0
        [nan, f(3.0), nan],       // NaN equals nothing
    ];
    check(
        "image_atomic_fcmpswap",
        0x1D,
        true,
        &cases,
        |old, data, compare| {
            if f32::from_bits(old) == f32::from_bits(compare) {
                data
            } else {
                old
            }
        },
    );
}

#[test]
fn image_atomic_fmin_and_fmax_keep_the_smaller_and_larger() {
    let f = f32::to_bits;
    let cases = [
        [f(2.0), f(-3.5), 0],
        [f(-3.5), f(2.0), 0],
        [f(1.0), f(1.0), 0],
        [f(-0.5), f(-0.25), 0],
    ];
    let pick = |min: bool| {
        move |old: u32, data: u32, _| {
            let (o, d) = (f32::from_bits(old), f32::from_bits(data));
            f32::to_bits(if min { d.min(o) } else { d.max(o) })
        }
    };
    check("image_atomic_fmin", 0x1E, false, &cases, pick(true));
    check("image_atomic_fmax", 0x1F, false, &cases, pick(false));
}
