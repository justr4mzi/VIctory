//! End-to-end tests of synthetic GFX10 VALU shaders lowered to DXIL, validated
//! by `dxil.dll` when present, and executed on the host GPU and WARP.
//!
//! Covers RDNA2 hardware semantics that differ from naive DXIL lowering:
//! `v_med3_f32`/`v_med3_f16` with NaN operands, `v_fract_f32`/`v_fract_f16`
//! near `1.0`, turn-range reduction for `v_sin_*`/`v_cos_*`, and the LDS
//! float atomics `ds_min_f32`/`ds_max_f32`.

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

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SOPC `op src0, src1`.
const fn sopc(op: u8, src0: u8, src1: u8) -> u32 {
    0xBF00_0000 | (op as u32 & 0x7F) << 16 | (src1 as u32) << 8 | src0 as u32
}

/// SOPP `op simm16`.
const fn sopp(op: u8, simm16: i16) -> u32 {
    0xBF80_0000 | (op as u32 & 0x7F) << 16 | simm16 as u16 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// VOP3B `op vdst, sdst, src0, src1, src2`.
const fn vop3b(op: u16, vdst: u8, sdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | (sdst as u32 & 0x7F) << 8 | vdst as u32,
        (src0 as u32 & 0x1FF) | (src1 as u32 & 0x1FF) << 9 | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// GLOBAL `global_load_dword vdst, vaddr, saddr` (`saddr = 125` for null).
const fn global_load_dword(vdst: u8, vaddr: u8, saddr: u8, offset: i16) -> [u32; 2] {
    [
        0x37 << 26 | 0x0C << 18 | 2 << 14 | (offset as u32 & 0xFFF),
        (vdst as u32) << 24 | (saddr as u32 & 0x7F) << 16 | vaddr as u32,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// Builds and lowers a 1-thread synthetic RDNA2 compute shader that receives
/// 4 dwords of root constants in `s0..s3`, runs `body` (leaving its result in
/// `v0`), and stores `v0` to dword 0 of the raw UAV at heap index 0.
fn build_valu_cs(body: &[u32]) -> HostShader {
    build_cs(body, 1, 0)
}

/// [`build_valu_cs`] with `threads` threads in x (thread index in `v0`
/// on entry, before `body`) and `lds_bytes` of LDS (a multiple of 512).
fn build_cs(body: &[u32], threads: u32, lds_bytes: u32) -> HostShader {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 so VALU instructions operate on
    // VGPRs just as they do in real shaders.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
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
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: (5 << 1) | ((lds_bytes / 512) << 15), // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code,
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
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

/// One GPU test at a time. With these tests on the hardware adapter and WARP
/// from several test threads at once, now and then none of their dispatches
/// ever completes and the test binary hangs; and two GPU tests at once on
/// WARP can end the test process with an access violation. Other GPU test
/// modules take this lock too.
pub(super) static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Runs `shader` on every available SM 6.6 host (hardware GPU and WARP) for
/// each 4-dword input set and invokes `check(backend, case_index, inputs, result_dword)`.
fn run_cases(
    shader: &HostShader,
    cases: &[[u32; 4]],
    check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    run_cases_on(shader, cases, &[("hardware", false), ("warp", true)], check);
}

fn run_cases_on(
    shader: &HostShader,
    cases: &[[u32; 4]],
    devices: &[(&str, bool)],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for &(label, warp) in devices {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
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
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, out[0][0]);
        }
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available ({devices:?})");
}

fn is_f16_nan(bits: u16) -> bool {
    (bits & 0x7FFF) > 0x7C00
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let f32_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut m = u32::from(mant);
                let mut e = 127 - 14;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x3FF) << 13)
            }
        }
        0x1F => sign | (0xFF << 23) | (u32::from(mant) << 13),
        _ => sign | ((u32::from(exp) + 112) << 23) | (u32::from(mant) << 13),
    };
    f32::from_bits(f32_bits)
}

#[test]
fn packed_add_selects_halves_of_inline_constants() {
    // Add the inline constant 1/(2*pi) to v1 (zero) to read back its two halves.
    // The same GFX10 source field (248) has a different bit pattern for a
    // packed floating-point and a packed integer instruction.
    let cases = [
        // v_pk_add_f16 v0, 0x3118, v1
        ([0xcc0f4000, 0x1a0202f8], 0x0000_3118),
        // v_pk_add_f16 v0, 0x3118, v1 op_sel_hi:[0,1] (use low twice)
        ([0xcc0f4000, 0x120202f8], 0x3118_3118),
        // v_pk_add_u16 v0, 0x3e22f983, v1
        ([0xcc0a4000, 0x1a0202f8], 0x3e22_f983),
    ];
    for (words, expected) in cases {
        let shader = build_valu_cs(&words);
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, expected, "{backend}: {words:08x?}");
        });
    }
}

#[test]
fn v_med3_f32_matches_rdna2_with_nan() {
    // v_med3_f32 v0, v1, v2, v3 (VOP3 opcode 0x157)
    let shader = build_valu_cs(&vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)));
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;

    let cases: &[([u32; 4], Option<f32>)] = &[
        // Normal medians (all orderings).
        ([f(1.0), f(2.0), f(3.0), 0], Some(2.0)),
        ([f(3.0), f(1.0), f(2.0), 0], Some(2.0)),
        ([f(-5.0), f(0.0), f(5.0), 0], Some(0.0)),
        ([f(1.0), f(1.0), f(2.0), 0], Some(1.0)),
        // One NaN: RDNA2 returns min3 of the non-NaN operands.
        ([f(1.0), f(2.0), nan, 0], Some(1.0)),
        ([nan, f(1.0), f(0.0), 0], Some(0.0)),
        ([nan, f(0.0), f(1.0), 0], Some(0.0)),
        ([f(2.0), nan, f(1.0), 0], Some(1.0)),
        ([f(-3.0), nan, f(4.0), 0], Some(-3.0)),
        // Two NaNs: returns the sole non-NaN operand.
        ([nan, f(5.0), nan, 0], Some(5.0)),
        ([nan, nan, f(-2.0), 0], Some(-2.0)),
        ([f(7.0), nan, nan, 0], Some(7.0)),
        // Three NaNs: returns NaN.
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert_eq!(
                got_f,
                want,
                "{backend}: v_med3_f32({:?}, {:?}, {:?}) = {got_f} ({got:#010x}), want {want}",
                f32::from_bits(inp[0]),
                f32::from_bits(inp[1]),
                f32::from_bits(inp[2]),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_med3_f32(NaN, NaN, NaN) = {got_f} ({got:#010x}), want NaN",
            ),
        }
    });
}

/// RDNA2 orders -0 below +0 in its maximums and minimums; DXIL's FMax and
/// FMin compare and select, which can give either zero.
#[test]
fn max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_0000_u32);
    // Each case has a -0 and a +0 among its first two operands.
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    let ops = [
        ("v_max_f32", 0x110, pz),
        ("v_min_f32", 0x10F, nz),
        ("v_max3_f32", 0x154, pz),
        ("v_min3_f32", 0x151, nz),
    ];
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            assert_eq!(got, want, "{backend}: {name} {inp:x?}");
        });
    }
}

#[test]
fn v_med3_f16_matches_rdna2_with_nan() {
    // v_med3_f16 v0, v1, v2, v3 (VOP3 opcode 0x357)
    let shader = build_valu_cs(&vop3(0x357, 0, vgpr(1), vgpr(2), vgpr(3)));
    let h0 = 0x0000u32; // 0.0
    let h1 = 0x3C00u32; // 1.0
    let h2 = 0x4000u32; // 2.0
    let h3 = 0x4200u32; // 3.0
    let h5 = 0x4500u32; // 5.0
    let hn2 = 0xC000u32; // -2.0
    let nan = 0x7E00u32;

    let cases: &[([u32; 4], Option<u16>)] = &[
        ([h1, h2, h3, 0], Some(h2 as u16)),
        ([hn2, h0, h3, 0], Some(h0 as u16)),
        ([h1, h2, nan, 0], Some(h1 as u16)),
        ([nan, h1, h0, 0], Some(h0 as u16)),
        ([nan, h0, h1, 0], Some(h0 as u16)),
        ([h2, nan, h1, 0], Some(h1 as u16)),
        ([nan, h5, nan, 0], Some(h5 as u16)),
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_med3_f16({:#06x}, {:#06x}, {:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0], inp[1], inp[2],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_med3_f16(NaN, NaN, NaN) = {got_h:#06x}, want NaN",
            ),
        }
    });
}

/// `v_cmp_lg_*` is "less or greater": ordered and not equal, so false when
/// either operand is a NaN (RDNA2 ISA; the same on gfx1036 hardware).
#[test]
fn v_cmp_lg_is_false_for_nan() {
    // v_cmp_lg_{f32,f16}_e64 s20, v1, v2 (VOPC 0x005 / 0x0cd in VOP3);
    // v_cndmask_b32_e64 v0, 0, 1, s20 (VOP3 0x101)
    let nan = f32::NAN.to_bits();
    let (inf, ninf) = (f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
    let f = f32::to_bits;
    let f32_cases: &[([u32; 4], u32)] = &[
        ([f(1.0), f(2.0), 0, 0], 1),
        ([f(2.0), f(1.0), 0, 0], 1),
        ([f(1.0), f(1.0), 0, 0], 0),
        ([f(0.0), f(-0.0), 0, 0], 0),
        ([inf, ninf, 0, 0], 1),
        ([inf, inf, 0, 0], 0),
        ([nan, f(1.0), 0, 0], 0),
        ([f(1.0), nan, 0, 0], 0),
        ([nan, nan, 0, 0], 0),
        ([0x7F80_0001, f(0.0), 0, 0], 0), // a signalling NaN
    ];
    let f16_cases: &[([u32; 4], u32)] = &[
        ([0x3C00, 0x4000, 0, 0], 1),
        ([0x4000, 0x3C00, 0, 0], 1),
        ([0x3C00, 0x3C00, 0, 0], 0),
        ([0x0000, 0x8000, 0, 0], 0),
        ([0x7C00, 0xFC00, 0, 0], 1),
        ([0x7E00, 0x3C00, 0, 0], 0),
        ([0x3C00, 0x7E00, 0, 0], 0),
        ([0x7E00, 0x7E00, 0, 0], 0),
        ([0x7C01, 0x0000, 0, 0], 0),           // a signalling NaN
        ([0xABCD_3C00, 0x1234_7E00, 0, 0], 0), // only the low halves are read
    ];
    for (op, what, cases) in [
        (0x005u16, "v_cmp_lg_f32", f32_cases),
        (0x0cd, "v_cmp_lg_f16", f16_cases),
    ] {
        let mut body = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }
}

#[test]
fn v_fract_f32_never_returns_one() {
    // v_fract_f32 v0, v1 (VOP1 opcode 0x020)
    let shader = build_valu_cs(&[vop1(0x020, 0, vgpr(1))]);
    let f = f32::to_bits;
    let below_one = 0x3F7F_FFFFu32;

    let cases: &[(u32, Option<u32>)] = &[
        // Tiny negatives: must clamp strictly below 1.0 (not 0x3f800000).
        (f(-1e-10), Some(below_one)),
        (f(-1e-8), Some(below_one)),
        (f(-2.0f32.powi(-25)), Some(below_one)),
        (f(-2.0f32.powi(-24)), Some(below_one)),
        // Normal values.
        (f(0.0), Some(f(0.0))),
        (f(-0.0), Some(f(0.0))),
        (f(0.25), Some(f(0.25))),
        (f(3.75), Some(f(0.75))),
        (f(-0.25), Some(f(0.75))),
        (f(-3.25), Some(f(0.75))),
        (f(1.0), Some(f(0.0))),
        (f(-1.0), Some(f(0.0))),
        // Non-finite inputs yield NaN.
        (f(f32::NAN), None),
        (f(f32::INFINITY), None),
        (f(f32::NEG_INFINITY), None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        match expect {
            Some(want) => assert_eq!(
                got,
                want,
                "{backend}: v_fract_f32({:?} / {:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
                f32::from_bits(inp[0]),
                inp[0],
                f32::from_bits(got),
            ),
            None => assert!(
                f32::from_bits(got).is_nan(),
                "{backend}: v_fract_f32({:?}) = {got:#010x}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });
}

#[test]
fn v_fract_f16_never_returns_one() {
    // v_fract_f16 v0, v1 (VOP1 opcode 0x05f)
    let shader = build_valu_cs(&[vop1(0x05f, 0, vgpr(1))]);
    let below_one = 0x3BFFu16; // 1.0 - 2^-11

    assert_eq!(f16_to_f32(0x9400), -2.0f32.powi(-10));
    assert_eq!(f16_to_f32(0x9000), -2.0f32.powi(-11));
    assert_eq!(f16_to_f32(0x8C00), -2.0f32.powi(-12));

    let cases: &[(u16, Option<u16>)] = &[
        // Negatives in (-2^-11, 0): must clamp to 0x3BFF, never 0x3C00 (1.0).
        (0x8C00, Some(below_one)), // -2^-12
        (0x8400, Some(below_one)), // -2^-14
        (0x8001, Some(below_one)), // -2^-24 (smallest negative f16 subnormal)
        (0x9000, Some(below_one)), // -2^-11 (exact 1 - 2^-11 = 0x3BFF)
        (0x9400, Some(0x3BFE)),    // -2^-10 (exact 1 - 2^-10 = 0x3BFE)
        // Normal values.
        (0x0000, Some(0x0000)), // 0.0 -> 0.0
        (0x3400, Some(0x3400)), // 0.25 -> 0.25
        (0xB400, Some(0x3A00)), // -0.25 -> 0.75
        (0x4300, Some(0x3800)), // 3.5 -> 0.5
        (0x3C00, Some(0x0000)), // 1.0 -> 0.0
        // Non-finite inputs yield NaN.
        (0x7E00, None), // NaN
        (0x7C00, None), // +inf
        (0xFC00, None), // -inf
    ];

    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

/// RDNA2's flt16_to_snorm16 / flt16_to_unorm16: clamp, scale by 32767 or
/// 65535, round the exact product to nearest even; NaN gives 0.
fn cvt_norm16_reference(bits: u16, signed: bool) -> u16 {
    let x = f16_to_f32(bits);
    if x.is_nan() {
        return 0;
    }
    let (lo, scale) = if signed {
        (-1.0, 32767.0)
    } else {
        (0.0, 65535.0)
    };
    // f64 holds the product of an f16 and a 16-bit scale exactly.
    let p = f64::from(x.clamp(lo, 1.0)) * scale;
    p.round_ties_even() as i32 as u16
}

#[test]
fn v_cvt_norm_i16_and_u16_f16_round_the_exact_product() {
    // Every 251st f16 pattern, plus the values where an f32 product would
    // round before the rounding to an integer, ties, and the edges.
    let mut bits: Vec<u16> = (0..=u16::MAX).step_by(251).collect();
    bits.extend([
        0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3800, 0xB800, 0x3801, 0xB801, 0x37FF,
        0x3BFF, 0xBBFF, 0x3C00, 0xBC00, 0x3C01, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01,
        0x3A9B, 0xBA9B, 0x2E66, 0x1400,
    ]);
    for (op, signed, what) in [
        (0x063u16, true, "v_cvt_norm_i16_f16"),
        (0x064, false, "v_cvt_norm_u16_f16"),
    ] {
        // v_cvt_norm_*16_f16 v0, v1 (VOP1); v0's high half starts at 0.
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = bits.iter().map(|&b| [u32::from(b), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = cvt_norm16_reference(bits[i], signed);
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                bits[i],
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f32_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x035, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x036, 0, vgpr(1))]);
    let f = f32::to_bits;
    let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    let sin_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(0.0)),
        (0.25, Some(1.0)),
        (255.0, Some(0.0)),
        (255.25, Some(1.0)),
        (255.5, Some(0.0)),
        (-255.25, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_sin_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_sin_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    let cos_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(1.0)),
        (0.25, Some(0.0)),
        (255.0, Some(1.0)),
        (255.25, Some(0.0)),
        (255.5, Some(-1.0)),
        (-255.5, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_cos_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_cos_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    // Main's `trig_turns` preserves sin(-0.0) = -0.0 and produces exact +0.0 at
    // half-turns (sin) and quarter-turns (cos).
    let exact_sin: &[(u32, u32)] = &[
        (0x8000_0000, 0x8000_0000), // sin(-0.0) = -0.0
        (f(255.0), 0),              // sin(255.0) = +0.0
        (f(255.5), 0),              // sin(255.5) = +0.0
    ];
    let exact_sin_in: Vec<[u32; 4]> = exact_sin.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&sin_shader, &exact_sin_in, |backend, i, inp, got| {
        let want = exact_sin[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_sin_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    let exact_cos: &[(u32, u32)] = &[
        (f(0.25), 0),   // cos(0.25) = +0.0
        (f(255.25), 0), // cos(255.25) = +0.0
    ];
    let exact_cos_in: Vec<[u32; 4]> = exact_cos.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&cos_shader, &exact_cos_in, |backend, i, inp, got| {
        let want = exact_cos[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cos_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
}

#[test]
fn v_cvt_u16_and_i16_f16_saturate() {
    // v_mov_b32 v0, 0; v_cvt_u16_f16 / v_cvt_i16_f16 v0, v1 (VOP1 0x052 / 0x053).
    let cases: &[(u16, u16, u16)] = &[
        // (f16 input, u16 result, i16 result)
        (0x0000, 0, 0),          // 0
        (0x8000, 0, 0),          // -0
        (0x3E00, 1, 1),          // 1.5 -> 1 (toward zero)
        (0xBE00, 0, 0xFFFF),     // -1.5 -> 0 / -1
        (0x57D0, 125, 125),      // 125.0
        (0x7800, 32768, 32767),  // 32768 saturates i16
        (0x77FF, 32752, 32752),  // 32752, the largest f16 below 32768
        (0xF800, 0, 0x8000),     // -32768 fits i16
        (0xF801, 0, 0x8000),     // -32800 saturates i16
        (0x7BFF, 65504, 32767),  // max finite f16
        (0xFBFF, 0, 0x8000),     // -65504
        (0x7C00, 0xFFFF, 32767), // +inf
        (0xFC00, 0, 0x8000),     // -inf
        (0x7E00, 0, 0),          // NaN
        (0xFE00, 0, 0),          // -NaN
        (0x0001, 0, 0),          // smallest subnormal
    ];
    for (op, what, pick) in [
        (0x052u16, "v_cvt_u16_f16", 0usize),
        (0x053, "v_cvt_i16_f16", 1),
    ] {
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [u32::from(c.0), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = if pick == 0 { cases[i].1 } else { cases[i].2 };
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                cases[i].0,
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f16_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x060, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x061, 0, vgpr(1))]);

    // Sanity-check the f16 bit encodings of large turns.
    assert_eq!(f16_to_f32(0x5BF8), 255.0);
    assert_eq!(f16_to_f32(0x5BFA), 255.25);
    assert_eq!(f16_to_f32(0x5BFC), 255.5);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);

    let sin_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(0.0)),  // 255.0
        (0x5BFA, Some(1.0)),  // 255.25
        (0x5BFC, Some(0.0)),  // 255.5
        (0xDBFA, Some(-1.0)), // -255.25
        (0x7BFF, Some(0.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_sin_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_sin_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });

    let cos_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(1.0)),  // 255.0
        (0x5BFA, Some(0.0)),  // 255.25
        (0x5BFC, Some(-1.0)), // 255.5
        (0x7BFF, Some(1.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_cos_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_cos_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

#[test]
fn bfe_signed_and_unsigned_match_rdna2() {
    // v_bfe_i32 v0, v1, v2, v3 (VOP3 opcode 0x149)
    let v_signed = build_valu_cs(&vop3(0x149, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_bfe_u32 v0, v1, v2, v3 (VOP3 opcode 0x148)
    let v_unsigned = build_valu_cs(&vop3(0x148, 0, vgpr(1), vgpr(2), vgpr(3)));
    // s_bfe_i32 s6, s0, s1 (SOP2 opcode 0x28: s1 = off[4:0] | width[22:16]); v_mov_b32 v0, s6
    let s_signed = build_valu_cs(&[sop2(0x28, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);

    let signed_cases: &[([u32; 4], u32)] = &[
        // Field crosses bit 31: extends from bit 31.
        ([0xd2f9_0159, 28, 16, 0], 0xffff_fffd),
        ([0x8000_0000, 31, 4, 0], 0xffff_ffff),
        ([0x7000_0000, 28, 8, 0], 0x0000_0007),
        // Normal fields inside 32 bits.
        ([0x0000_0080, 4, 4, 0], 0xffff_fff8),
        ([0x0000_0070, 4, 4, 0], 0x0000_0007),
        // Edge widths: 0 yields 0; v_bfe masks width to 5 bits (32 -> 0),
        // while 31 extracts all remaining bits from offset 4.
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0xf800_0000),
    ];
    let v_in: Vec<[u32; 4]> = signed_cases.iter().map(|c| c.0).collect();
    run_cases(&v_signed, &v_in, |backend, i, inp, got| {
        let want = signed_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_i32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
    // v_bfe_i32 width is 5-bit (src2[4:0]), so width=32 wraps to 0.
    run_cases(
        &v_signed,
        &[[0x8000_0000, 4, 32, 0]],
        |backend, _, _, got| {
            assert_eq!(got, 0, "{backend}: v_bfe_i32 width 32 should mask to 0");
        },
    );

    // s_bfe_i32 width is 7-bit (src1[22:16]), so width=32 keeps all 32 bits.
    let mut s_cases = signed_cases.to_vec();
    s_cases.push(([0x8000_0000, 4, 32, 0], 0xf800_0000));
    let s_in: Vec<[u32; 4]> = s_cases
        .iter()
        .map(|&(inp, _)| [inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0])
        .collect();
    run_cases(&s_signed, &s_in, |backend, i, inp, got| {
        let want = s_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_i32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    let unsigned_cases: &[([u32; 4], u32)] = &[
        ([0xd2f9_0159, 28, 16, 0], 0x0000_000d),
        ([0x8000_0000, 31, 4, 0], 0x0000_0001),
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0x0800_0000),
    ];
    let u_in: Vec<[u32; 4]> = unsigned_cases.iter().map(|c| c.0).collect();
    run_cases(&v_unsigned, &u_in, |backend, i, inp, got| {
        let want = unsigned_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_u32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn v_cmp_class_f16_matches_rdna2() {
    // v_cmp_class_f16_e64 s6, v1, v2 (VOP3 opcode 0x08f); v_cndmask_b32_e64 v0, 0, 1, s6
    let cmp_lo = build_valu_cs(&[
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[0],
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[1],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);
    // Same with op_sel:[1,0] (high half of v1) + neg on src0 (bit 29 of w1).
    let [mut w0, mut w1] = vop3(0x08f, 6, vgpr(1), vgpr(2), 0);
    w0 |= 1 << 11; // opsel[0] = 1 (high half)
    w1 |= 1 << 29; // neg[0] = 1
    let cmp_hi_neg = build_valu_cs(&[
        w0,
        w1,
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);

    // All 10 IEEE-754 binary16 classes: bit k corresponds to class k.
    let classes: &[(u32, u32)] = &[
        (0x7C01, 0), // sNaN (+sNaN, quiet bit 0x0200 clear)
        (0xFDFF, 0), // -sNaN (sign ignored for NaNs)
        (0x7E00, 1), // +qNaN
        (0xFE01, 1), // -qNaN
        (0xFC00, 2), // -inf
        (0xBC00, 3), // -normal (-1.0)
        (0x8001, 4), // -denormal
        (0x8000, 5), // -0.0
        (0x0000, 6), // +0.0
        (0x0001, 7), // +denormal
        (0x3C00, 8), // +normal (+1.0)
        (0x7C00, 9), // +inf
    ];
    let mut cases = Vec::new();
    for &(bits, cls) in classes {
        // Exact class bit -> true; complement of class bit across all 10 bits -> false.
        cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_lo, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16({:#06x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });

    // High half with neg: high half 0x3C00 (+1.0) negated -> -normal (class 3);
    // high half 0x7C01 (+sNaN) negated -> still sNaN (class 0, not quieted).
    let hi_cases: &[([u32; 4], u32)] = &[
        ([0x3C00_0000, 1 << 3, 0, 0], 1),
        ([0x3C00_0000, 1 << 8, 0, 0], 0),
        ([0x7C01_FFFF, 1 << 0, 0, 0], 1),
        ([0x7C01_FFFF, 1 << 1, 0, 0], 0),
    ];
    let hi_in: Vec<[u32; 4]> = hi_cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_hi_neg, &hi_in, |backend, i, inp, got| {
        let want = hi_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16(-hi({:#010x}), {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

const S_BARRIER: u32 = 0xBF8A_0000;

/// DS `op vdst, addr, data0` at offset 0.
const fn ds(op: u8, vdst: u8, addr: u8, data0: u8) -> [u32; 2] {
    [
        0x36 << 26 | (op as u32) << 18,
        (vdst as u32) << 24 | (data0 as u32) << 8 | addr as u32,
    ]
}

const DS_WRITE_B32: u8 = 0x0D;
const DS_MIN_F32: u8 = 0x12;
const DS_MAX_F32: u8 = 0x13;
const DS_MIN_RTN_F32: u8 = 0x32;
const DS_MAX_RTN_F32: u8 = 0x33;
const DS_READ_B32: u8 = 0x36;

/// What RDNA2 writes for `ds_max_f32` / `ds_min_f32` over a dword holding
/// `old` with `data` (bits), as measured on gfx1036 with f32 denormals kept:
/// an sNaN, data first, then memory, is written quieted; otherwise a NaN
/// loses to a number (two quiet NaNs give the data); numbers compare as
/// floats with -0 below +0.
fn lds_min_max_ref(old: u32, data: u32, max: bool) -> u32 {
    let nan = |x: u32| x & 0x7FFF_FFFF > 0x7F80_0000;
    let snan = |x: u32| nan(x) && x & 0x40_0000 == 0;
    let key = |x: u32| {
        let b = x as i32;
        if b < 0 { b ^ 0x7FFF_FFFF } else { b }
    };
    if snan(data) {
        return data | 0x40_0000;
    }
    if snan(old) {
        return old | 0x40_0000;
    }
    match (nan(data), nan(old)) {
        (true, true) | (false, true) => return data,
        (true, false) => return old,
        (false, false) => {}
    }
    let take = if max {
        key(data) > key(old)
    } else {
        key(data) < key(old)
    };
    if take { data } else { old }
}

#[test]
fn ds_min_max_f32_order_floats_and_return_the_old_value() {
    let values = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        2.5,
        -2.5,
        1e-30,
        -1e-30,
        f32::from_bits(1), // denormal
        f32::from_bits(0x8000_0001),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
    ];
    // Quiet and signalling NaNs of both signs, in memory and as data.
    let nans = [
        0x7FC0_0000u32,
        0xFFC0_0000,
        0x7F80_0001,
        0xFFA0_0000,
        0x7FBF_FFFF,
    ];
    let bits: Vec<u32> = values.iter().map(|v| v.to_bits()).chain(nans).collect();
    let mut cases: Vec<[u32; 4]> = Vec::new();
    for &old in &bits {
        for &data in &bits {
            cases.push([old, data, 0, 0]);
        }
    }
    for (rtn, plain, max, what) in [
        (DS_MAX_RTN_F32, DS_MAX_F32, true, "ds_max"),
        (DS_MIN_RTN_F32, DS_MIN_F32, false, "ds_min"),
    ] {
        // v5 = 0; LDS[v5] = v1; the atomic with v2; v0 = the old value
        // (rtn) or the dword afterwards (read back).
        let mut returns_old = vec![vop1(0x001, 5, 0x80)];
        returns_old.extend(ds(DS_WRITE_B32, 0, 5, 1));
        returns_old.extend(ds(rtn, 0, 5, 2));
        let mut returns_new = vec![vop1(0x001, 5, 0x80)];
        returns_new.extend(ds(DS_WRITE_B32, 0, 5, 1));
        returns_new.extend(ds(plain, 0, 5, 2));
        returns_new.extend(ds(DS_READ_B32, 0, 5, 0));
        for (body, old_result) in [(returns_old, true), (returns_new, false)] {
            // One thread with LDS declares WaveSize(32), which WARP's
            // pipeline creation refuses: hardware only (the 64-thread test
            // below runs on WARP too).
            let shader = build_cs(&body, 1, 512);
            run_cases_on(
                &shader,
                &cases,
                &[("hardware", false)],
                |backend, _, inp, got| {
                    let (old, data) = (f32::from_bits(inp[0]), f32::from_bits(inp[1]));
                    let want = if old_result {
                        inp[0]
                    } else {
                        lds_min_max_ref(inp[0], inp[1], max)
                    };
                    assert_eq!(
                        got,
                        want,
                        "{backend}: {what}_f32 on {old:e} [{:#010x}] with {data:e} [{:#010x}] gave {got:#010x}, want {want:#010x} ({})",
                        inp[0],
                        inp[1],
                        if old_result {
                            "old value"
                        } else {
                            "dword after"
                        },
                    );
                },
            );
        }
    }
}

#[test]
fn ds_min_max_f32_are_atomic_across_waves() {
    // 64 threads (two host waves of 32 or one of 64): LDS[0] = v1, then
    // every thread applies (thread - 32) as a float; v0 = LDS[0].
    for (op, max, init, want) in [
        (DS_MAX_F32, true, f32::NEG_INFINITY, 31.0f32),
        (DS_MAX_F32, true, 100.0, 100.0),
        (DS_MIN_F32, false, f32::INFINITY, -32.0),
        (DS_MIN_F32, false, -0.0, -32.0),
    ] {
        let mut body = vec![vop1(0x001, 5, 0x80)];
        body.extend(ds(DS_WRITE_B32, 0, 5, 1));
        body.push(S_BARRIER);
        body.push(vop2(0x26, 6, vgpr(0), 3)); // v_sub_nc_u32 v6, v0, v3
        body.push(vop1(0x005, 6, vgpr(6))); // v_cvt_f32_i32
        body.extend(ds(op, 0, 5, 6));
        body.push(S_BARRIER);
        body.extend(ds(DS_READ_B32, 0, 5, 0));
        let shader = build_cs(&body, 64, 512);
        run_cases(
            &shader,
            &[[init.to_bits(), 0, 32, 0]],
            |backend, _, _, got| {
                assert_eq!(
                    f32::from_bits(got),
                    want,
                    "{backend}: {} from {init} over 64 threads",
                    if max { "max" } else { "min" },
                );
            },
        );
    }
}

#[test]
fn m0_relative_swaps_and_moves_index_source_and_destination_apart() {
    // v1..v4 = 10, 20, 30, 40. v_movrelsd_2_b32 and v_swaprel_b32 offset
    // their source by M0[9:0] and their destination by M0[25:16]; other M0
    // bits are ignored. Each case reads one VGPR back into v0.
    const SWAPREL: u16 = 0x068;
    const MOVRELSD_2: u16 = 0x048;
    let cases: &[(u16, u8, u32, u8, u32)] = &[
        // (opcode, vdst, M0, register read back, expected)
        // v[1 + 2] <-> v[1 + 0]
        (SWAPREL, 1, 2 << 16, 1, 30),
        (SWAPREL, 1, 2 << 16, 3, 10),
        // v[1 + 0] <-> v[1 + 3]
        (SWAPREL, 1, 3, 4, 10),
        (SWAPREL, 1, 3, 1, 40),
        // Both name v2: it keeps its value.
        (SWAPREL, 1, 1 << 16 | 1, 2, 20),
        // v[5 + 1] = v[1 + 2]
        (MOVRELSD_2, 5, 1 << 16 | 2, 6, 30),
        // The same with every ignored M0 bit set.
        (MOVRELSD_2, 5, 0xFC00_FC00 | 1 << 16 | 2, 6, 30),
    ];
    for &(op, vdst, m0, back, want) in cases {
        let shader = build_valu_cs(&[
            s_mov_b32(124, 0xFF), // s_mov_b32 m0, literal
            m0,
            vop1(op, vdst, vgpr(1)),
            vop1(0x001, 0, vgpr(back)), // v_mov_b32 v0, v[back]
        ]);
        run_cases(&shader, &[[10, 20, 30, 40]], |backend, _, _, got| {
            assert_eq!(
                got, want,
                "{backend}: op {op:#05x} v{vdst}, v1 with M0 {m0:#010x}, v{back} = {got}, want {want}",
            );
        });
    }
}

/// Builds and lowers a 1-thread synthetic RDNA2 compute shader with root constants
/// in `s0..s3`, a UAV table at `s4`, and a Root SRV descriptor in `s6..s7`.
/// `body` leaves its result in `v0`, which is stored to dword 0 of the UAV.
fn build_root_srv_cs(body: &[u32]) -> HostShader {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 for VALU address expressions.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
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
            RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 6,
                slot: 2,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 6,
                slot: 2,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 8 << 1, // 8 user SGPRs: s0..s7
            threads: [1, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let res = resource::track(&p, &native.layout, native.stage, None).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let st = structure::structure(&p).expect("structure");
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
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

/// Runs a Root SRV compute shader on available SM 6.6 devices (hardware GPU and WARP).
fn run_root_srv_cases(
    shader: &HostShader,
    srv_data: &[u32],
    cases: &[[u32; 4]],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(ud)),
                            (
                                1,
                                Root::View {
                                    srv: true,
                                    data: srv_data.to_vec(),
                                },
                            ),
                        ],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, out[0][0]);
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn root_srv_s_lshl2_add_u32_with_root_constant_dword0() {
    // s_lshl2_add_u32 s12, s0, s6; s_add_u32 s12, s12, s1; s_addc_u32 s13, s7, 0;
    // v_mov_b32 v5, 0; global_load_dword v0, v5, s[12:13]
    let shader = build_root_srv_cs(&[
        sop2(0x2F, 12, 0, 6),
        sop2(0x00, 12, 12, 1),
        sop2(0x04, 13, 7, 0x80),
        vop1(0x001, 5, 0x80),
        global_load_dword(0, 5, 12, 0)[0],
        global_load_dword(0, 5, 12, 0)[1],
    ]);
    let srv_data: Vec<u32> = (0..16).map(|i| 0x1000_0000 + i * 0x0101_0101).collect();
    let cases = [[0, 0, 0, 0], [3, 0, 0, 0], [5, 8, 0, 0], [11, 16, 0, 0]];
    run_root_srv_cases(&shader, &srv_data, &cases, |backend, _, inp, got| {
        let want = srv_data[(inp[0] + inp[1] / 4) as usize];
        assert_eq!(
            got, want,
            "{backend}: s_lshl2_add_u32 index {} + {}B = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn root_srv_two_term_v_add_co_and_v_lshl_add() {
    // v_add_co_u32 v5, vcc, s6, v1
    // v_add_co_ci_u32 v6, vcc, s7, 0, vcc (VOP3B op 0x128)
    // v_lshl_add_u32 v5, v2, 2, v5
    // global_load_dword v0, v[5:6], off
    let add_co = vop3b(0x30F, 5, 106, sgpr(6), vgpr(1), 0);
    let add_ci = vop3b(0x128, 6, 106, sgpr(7), 0x80, sgpr(106));
    let lshl_add = vop3(0x346, 5, vgpr(2), 0x82, vgpr(5));
    let load = global_load_dword(0, 5, 125, 0);
    let shader = build_root_srv_cs(&[
        add_co[0],
        add_co[1],
        add_ci[0],
        add_ci[1],
        lshl_add[0],
        lshl_add[1],
        load[0],
        load[1],
    ]);
    let srv_data: Vec<u32> = (0..24).map(|i| 0xA000_0000 ^ (i * 0x0011_2233)).collect();
    let cases = [[0, 0, 0, 0], [8, 1, 0, 0], [4, 5, 0, 0], [16, 7, 0, 0]];
    run_root_srv_cases(&shader, &srv_data, &cases, |backend, _, inp, got| {
        let idx = (inp[0] / 4 + inp[1]) as usize;
        let want = srv_data[idx];
        assert_eq!(
            got, want,
            "{backend}: two-term v_add_co + v_lshl_add (off={}, idx={}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn root_srv_loop_carried_pointer_induction_phi() {
    // s12, s13 = root_srv + (s0 << 2); s14 = 0; v0 = 0; v5 = 0;
    // loop:
    //   global_load_dword v6, v5, s[12:13]   (2 dwords)
    //   v_add_nc_u32 v0, v0, v6              (1 dword)
    //   s_add_u32 s12, s12, 4                (1 dword)
    //   s_addc_u32 s13, s13, 0               (1 dword)
    //   s_add_u32 s14, s14, 1                (1 dword)
    //   s_cmp_lt_u32 s14, 4                  (1 dword)
    //   s_cbranch_scc1 -8                    (jumps back 8 dwords -> start of loop)
    let load = global_load_dword(6, 5, 12, 0);
    let shader = build_root_srv_cs(&[
        sop2(0x2F, 12, 0, 6),      // s_lshl2_add_u32 s12, s0, s6
        s_mov_b32(13, 7),          // s_mov_b32 s13, s7
        s_mov_b32(14, 0x80),       // s_mov_b32 s14, 0
        vop1(0x001, 0, 0x80),      // v_mov_b32 v0, 0
        vop1(0x001, 5, 0x80),      // v_mov_b32 v5, 0
        load[0],                   // loop start (-7 before branch)
        load[1],                   // (-6)
        vop2(0x25, 0, vgpr(0), 6), // v_add_nc_u32 v0, v0, v6 (-5)
        sop2(0x00, 12, 12, 0x84),  // s_add_u32 s12, s12, 4 (-4)
        sop2(0x04, 13, 13, 0x80),  // s_addc_u32 s13, s13, 0 (-3)
        sop2(0x00, 14, 14, 0x81),  // s_add_u32 s14, s14, 1 (-2)
        sopc(0x0A, 14, 0x84),      // s_cmp_lt_u32 s14, 4 (-1)
        sopp(0x05, -8),            // s_cbranch_scc1 loop
    ]);
    let srv_data: Vec<u32> = (0..16).map(|i| 100 + i * 17).collect();
    let cases = [[0, 0, 0, 0], [2, 0, 0, 0], [5, 0, 0, 0], [11, 0, 0, 0]];
    run_root_srv_cases(&shader, &srv_data, &cases, |backend, _, inp, got| {
        let start = inp[0] as usize;
        let want: u32 = srv_data[start..start + 4]
            .iter()
            .fold(0u32, |a, &x| a.wrapping_add(x));
        assert_eq!(
            got, want,
            "{backend}: loop phi sum starting at {start} = {got}, want {want}",
        );
    });
}

/// SMEM `s_load_dwordx8 s[dst:dst+7], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx8(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x03 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// MIMG `image_store vdata, vaddr, s[srsrc:srsrc+7] dmask:dmask dim:2D unorm [d16]`.
const fn image_store_2d(vdata: u8, vaddr: u8, srsrc: u8, dmask: u8, d16: bool) -> [u32; 2] {
    let w0 = 0x3C << 26 | 0x08 << 18 | 1 << 12 | (dmask as u32 & 0xF) << 8 | 1 << 3;
    let w1 = (d16 as u32) << 31
        | ((srsrc as u32 >> 2) & 0x1F) << 16
        | (vdata as u32) << 8
        | (vaddr as u32);
    [w0, w1]
}

fn try_build_image_store_cs(dmask: u8, d16: bool) -> Result<HostShader, crate::ShaderError> {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx8(8, 4));
    // Data in v1..v4 from root constants s0..s3; 2D texel coordinate (0, 0) in v5..v6.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.push(vop1(0x001, 5, 0x80)); // v5 = 0
    words.push(vop1(0x001, 6, 0x80)); // v6 = 0
    words.extend_from_slice(&image_store_2d(1, 5, 8, dmask, d16));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
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
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [1, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native)?;
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
    })?;

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        table_views: Vec::new(),
        stride_table: false,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    Ok(shader)
}

#[test]
fn image_store_partial_dmask_writes_expected_channels() {
    // Non-contiguous dmask (e.g. 0x5 = x|z) is rejected because DXIL
    // `textureStore` requires a contiguous mask from channel 0.
    assert!(matches!(
        try_build_image_store_cs(0x5, false),
        Err(crate::ShaderError::Unsupported {
            what: "an image store of some channels",
            ..
        })
    ));

    // (dmask, d16, input root constants, expected channels written)
    let cases: &[(u8, bool, [u32; 4], &[f32])] = &[
        (0x0, false, [1.5f32.to_bits(), 0, 0, 0], &[0.0, 0.0, 0.0, 0.0]),
        (0x1, false, [1.5f32.to_bits(), 0, 0, 0], &[1.5]),
        (
            0x3,
            false,
            [1.5f32.to_bits(), (-2.25f32).to_bits(), 0, 0],
            &[1.5, -2.25],
        ),
        (
            0x7,
            false,
            [
                1.5f32.to_bits(),
                (-2.25f32).to_bits(),
                4.0f32.to_bits(),
                0,
            ],
            &[1.5, -2.25, 4.0],
        ),
        // d16: 1.5 (0x3E00) in low half, -2.0 (0xC000) in high half, 4.0 (0x4400) in next VGPR.
        (0x1, true, [0x0000_3E00, 0, 0, 0], &[1.5]),
        (0x3, true, [0xC000_3E00, 0, 0, 0], &[1.5, -2.0]),
        (0x7, true, [0xC000_3E00, 0x0000_4400, 0, 0], &[1.5, -2.0, 4.0]),
    ];

    let shaders: Vec<HostShader> = cases
        .iter()
        .map(|&(dmask, d16, ..)| {
            try_build_image_store_cs(dmask, d16)
                .unwrap_or_else(|e| panic!("dmask={dmask:#x} d16={d16}: {e}"))
        })
        .collect();

    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for (&(dmask, d16, inputs, want), shader) in cases.iter().zip(&shaders) {
            let (root, pso) = host
                .compute_pipeline(shader)
                .unwrap_or_else(|e| panic!("{label}: dmask={dmask:#x} d16={d16}: {e}"));
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
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
                        views: &[View::TextureUav(1, 1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            let got: Vec<f32> = out[0][..want.len()]
                .iter()
                .map(|&b| f32::from_bits(b))
                .collect();
            assert_eq!(
                got.as_slice(),
                want,
                "{label}: image_store dmask={dmask:#x} d16={d16} wrote {got:?}, want {want:?}"
            );
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP supports SM 6.6");
}

/// EXP `exp target, v0, v1, v2, v3 [done] [vm]` (`en`: 4-bit channel mask).
const fn exp_inst(
    target: u8,
    en: u8,
    v: [u8; 4],
    done: bool,
    vm: bool,
) -> [u32; 2] {
    let w0 = 0x3E << 26
        | (vm as u32) << 12
        | (done as u32) << 11
        | (target as u32 & 0x3F) << 4
        | (en as u32 & 0xF);
    let w1 = (v[3] as u32) << 24 | (v[2] as u32) << 16 | (v[1] as u32) << 8 | (v[0] as u32);
    [w0, w1]
}

#[test]
fn vs_rt_and_viewport_index_exports_validate_and_create_pso() {
    use super::container::SigElement;
    use crate::dxbc::{Container, FourCC};

    let build_stage = |stage: HwStage,
                       words: &[u32],
                       layout: Vec<RootParam>,
                       stage_io: StageIo,
                       outputs: Vec<SigElement>|
     -> HostShader {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage,
            wave64: false,
            vgprs: 8,
            layout,
            used: Vec::new(),
            io: stage_io,
            code,
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
            outputs,
            prims: Vec::new(),
            mesh_stride: 0,
        })
        .expect("lower");
        let shader = HostShader {
            stage,
            dxil: lowered.dxil,
            user_data: lowered.user_data,
            samplers: lowered.samplers,
            static_samplers: Vec::new(),
            root_cbvs: lowered.root_cbvs,
            root_srvs: lowered.root_srvs,
            table_views: Vec::new(),
            stride_table: false,
        };
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            let mut copy = shader.dxil.clone();
            let ours = copy[4..20].to_vec();
            v.validate(&mut copy).expect("DXIL validation failed");
            assert_eq!(copy[4..20], ours[..], "container hash mismatch");
        }
        shader
    };

    // Minimal pixel shader writing SV_Target0 = (1, 0, 0, 1).
    let ps_words = [
        vop1(0x001, 0, 0xF2), // v0 = 1.0
        vop1(0x001, 1, 0x80), // v1 = 0.0
        vop1(0x001, 2, 0x80), // v2 = 0.0
        vop1(0x001, 3, 0xF2), // v3 = 1.0
        exp_inst(0, 0xF, [0, 1, 2, 3], true, true)[0],
        exp_inst(0, 0xF, [0, 1, 2, 3], true, true)[1],
        S_ENDPGM,
    ];
    let ps = build_stage(
        HwStage::Ps,
        &ps_words,
        Vec::new(),
        StageIo::Ps {
            inputs: Vec::new(),
            rsrc2: 0,
            input_addr: 0,
            db_shader_control: 0,
            patches: Vec::new(),
        },
        vec![SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }],
    );

    // Three VS variants: RT index only, viewport index only, and both + clip distance in pos2.
    let configs = [
        (true, false, false),
        (false, true, false),
        (true, true, true),
    ];
    let mut vs_shaders = Vec::new();
    for (has_rt, has_vp, has_clip) in configs {
        let mut words = vec![
            vop1(0x001, 0, 0x80),    // pos.x = 0.0
            vop1(0x001, 1, 0x80),    // pos.y = 0.0
            vop1(0x001, 2, 0x80),    // pos.z = 0.0
            vop1(0x001, 3, 0xF2),    // pos.w = 1.0
            vop1(0x001, 4, sgpr(0)), // pos1.z = packed (viewport << 16) | rt
            vop1(0x001, 5, sgpr(1)), // pos2.x = clip distance
        ];
        words.extend_from_slice(&exp_inst(12, 0xF, [0, 1, 2, 3], false, false)); // pos0
        words.extend_from_slice(&exp_inst(13, 0x4, [0, 0, 4, 0], !has_clip, false)); // pos1.z
        if has_clip {
            words.extend_from_slice(&exp_inst(14, 0x1, [5, 0, 0, 0], true, false)); // pos2.x
        }
        words.push(S_ENDPGM);

        let mut outputs = vec![SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 4,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }];
        let mut next_row = 1u32;
        if has_rt {
            outputs.push(SigElement {
                name: "SV_RenderTargetArrayIndex".into(),
                index: 0,
                kind: 4,
                comp: 5,
                interp: 1,
                row: Some(next_row),
                col: 0,
                cols: 1,
                usage: 0,
                half: false,
            });
            next_row += 1;
        }
        if has_vp {
            outputs.push(SigElement {
                name: "SV_ViewportArrayIndex".into(),
                index: 0,
                kind: 5,
                comp: 5,
                interp: 1,
                row: Some(next_row),
                col: 0,
                cols: 1,
                usage: 0,
                half: false,
            });
            next_row += 1;
        }
        if has_clip {
            outputs.push(SigElement {
                name: "SV_ClipDistance".into(),
                index: 0,
                kind: 6,
                comp: 9,
                interp: 2,
                row: Some(next_row),
                col: 0,
                cols: 1,
                usage: 0,
                half: false,
            });
        }

        let vs = build_stage(
            HwStage::Vs,
            &words,
            vec![RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            StageIo::Vs {
                inputs: Vec::new(),
                outputs: Vec::new(),
            },
            outputs,
        );
        let c = Container::parse(&vs.dxil).expect("parse emitted VS DXIL");
        let sfi0 = u64::from_le_bytes(
            c.part(FourCC::SFI0).expect("SFI0").data[..8]
                .try_into()
                .unwrap(),
        );
        assert_ne!(
            sfi0 & 0x2000,
            0,
            "SFI0 should set ViewPortAndRTArrayIndexFromAnyShaderFeedingRasterizer"
        );
        vs_shaders.push(vs);
    }

    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        for (i, vs) in vs_shaders.iter().enumerate() {
            host.pipeline(vs, &ps)
                .unwrap_or_else(|e| panic!("{label}: graphics pipeline case {i} failed: {e}"));
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP supports SM 6.6");
}

#[test]
fn reachability_overwritten_missing_phis_match_on_host_and_warp() {
    // 1. Correlated overwrite of a missing descriptor word (`s13`) across two joins:
    //    when s0 == 0: load s[12:15], read UAV via raw buffer_load_dword, then overwrite s13 = 29;
    //    when s0 != 0: set s13 = 17, skip overwrite. Result v0 = s13.
    let desc_word_shader = build_valu_cs(&[
        0x7E0C_0280, // v_mov_b32 v6, 0
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0002, // s_cbranch_scc1 +2
        0xBE8D_0391, // s_mov_b32 s13, 17
        0xBF82_0004, // s_branch +4
        s_load_dwordx4(12, 4)[0],
        s_load_dwordx4(12, 4)[1],
        0xE030_0000,
        0x8003_0506, // buffer_load_dword v5, v6, s[12:15], 0
        0xBF84_0001, // s_cbranch_scc0 +1 (skip overwrite only when s0 != 0)
        0xBE8D_039D, // s_mov_b32 s13, 29
        vop1(0x001, 0, sgpr(13)),
    ]);
    // 2. Correlated overwrite of a VGPR missing value (`s_memtime` -> `v5` -> Missing) across two joins:
    //    when s0 == 0: sample `s_memtime` into v5, then overwrite v5 = 29;
    //    when s0 != 0: v5 = 17, skip overwrite. Result v0 = v5.
    let unknown_stride_shader = build_valu_cs(&[
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0002, // s_cbranch_scc1 +2
        0x7E0A_0291, // v_mov_b32 v5, 17
        0xBF82_0003, // s_branch +3
        0xF490_0180,
        0,           // s_memtime s[6:7]
        vop1(0x001, 5, sgpr(6)),
        0xBF84_0001, // s_cbranch_scc0 +1 (skip overwrite only when s0 != 0)
        0x7E0A_029D, // v_mov_b32 v5, 29
        vop1(0x001, 0, vgpr(5)),
    ]);
    // 3. Loop-invariant infeasible s_memtime phi (latch uses VCC to preserve SCC):
    //    when s0 == 0: loop runs s_memtime -> s13, then overwrites s13 = 29 after loop;
    //    when s0 != 0: loop keeps s13 = 17, skips overwrite after loop.
    let loop_shader = build_valu_cs(&[
        0xBE8D_0391, // s_mov_b32 s13, 17
        0x7E0E_0280, // v_mov_b32 v7, 0
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF84_0003, // s_cbranch_scc0 +3
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0xBE8D_0306, // s_mov_b32 s13, s6
        0x4A0E_0E81, // v_add_nc_u32 v7, 1, v7
        0x7D88_0E82, // v_cmp_gt_u32 vcc_lo, 2, v7
        0xBF87_FFF9, // s_cbranch_vccnz -7
        0xBF84_0001, // s_cbranch_scc0 +1
        0xBE8D_039D, // s_mov_b32 s13, 29
        vop1(0x001, 0, sgpr(13)),
    ]);

    let cases = [([0, 0, 0, 0], 29u32), ([1, 0, 0, 0], 17u32)];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    for (name, shader) in [
        ("desc_word", &desc_word_shader),
        ("unknown_stride", &unknown_stride_shader),
        ("loop_memtime", &loop_shader),
    ] {
        run_cases(shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend} {name} s0={}: got {got}, want {want}",
                inp[0]
            );
        });
    }
}

/// SMEM `s_load_dwordx4` (count 2) or `s_load_dwordx8` (count 3) with byte offset.
const fn smem_load(count_code: u8, dst: u8, base: u8, byte_off: u32) -> [u32; 2] {
    [
        0x3D << 26 | (count_code as u32) << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25 | (byte_off & 0x1F_FFFF),
    ]
}

/// MIMG instruction (`dim` 3 = CUBE).
const fn mimg_cube(
    op: u8,
    vdata: u8,
    vaddr: u8,
    srsrc: u8,
    ssamp: u8,
    dmask: u8,
    a16: bool,
) -> [u32; 2] {
    [
        0x3C << 26 | (op as u32) << 18 | (dmask as u32) << 8 | 3 << 3,
        (vaddr as u32)
            | (vdata as u32) << 8
            | ((srsrc as u32 >> 2) & 0x1F) << 16
            | ((ssamp as u32 >> 2) & 0x1F) << 21
            | (a16 as u32) << 30,
    ]
}

/// Builds a 1-thread compute shader with root constants `s0..s3` copied into
/// `v1..v4`, SRV 0 (`s[8:15]`) at view heap index 0, raw UAV (`s[20:23]`) at
/// view heap index 1, and point-clamp sampler (`s[16:19]`) at sampler index 0.
/// `body` writes `uav_dwords` result dwords starting at `v0`.
fn build_cube_cs(body: &[u32], uav_dwords: u8) -> HostShader {
    let mut words = Vec::new();
    // Copy user SGPR s5 (sampler table) into s6 before zeroing s5 for s[4:5].
    words.push(s_mov_b32(6, 5));
    words.push(s_mov_b32(7, 0x80));
    words.push(s_mov_b32(5, 0x80));
    words.extend_from_slice(&smem_load(0x03, 8, 4, 0));
    words.extend_from_slice(&smem_load(0x02, 16, 6, 0));
    words.extend_from_slice(&smem_load(0x02, 20, 4, 32));
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i)));
    }
    words.extend_from_slice(body);
    for k in 0..uav_dwords {
        words.extend_from_slice(&buffer_store_dword(k, 20, 4 * u16::from(k)));
    }
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 16,
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
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 5,
                slot: 2,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 4,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 8,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Sampler,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 5,
                slot: 2,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 6 << 1,
            threads: [1, 1, 1],
        }),
        code,
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
        table_views: Vec::new(),
        stride_table: false,
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

/// Runs `shader` on every SM 6.6 adapter (`Host::adapters`, including hardware
/// GPUs and WARP) with `srv` at heap index 0 and a raw UAV of `uav_dwords` at
/// heap index 1.
fn run_cube_cases(
    shader: &HostShader,
    srv: View,
    uav_dwords: u32,
    cases: &[[u32; 4]],
    mut check: impl FnMut(&str, usize, [u32; 4], &[u32]),
) {
    use windows::Win32::Graphics::Direct3D12::*;
    let samplers = [D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        MaxLOD: f32::MAX,
        ..Default::default()
    }];
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let adapters = Host::adapters();
    assert!(
        !adapters.is_empty(),
        "no SM 6.6 hardware or WARP device available"
    );
    let views = [srv, View::RawUav(uav_dwords)];
    for (label, host) in &adapters {
        eprintln!("cube test adapter: {label}");
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1 | 2) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &views,
                        samplers: &samplers,
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, &out[0]);
        }
    }
}

#[test]
fn cube_array_samples_layers_faces_and_clamped_ids() {
    // 2-cube array (12 faces, 4x4 per face): each texel encodes layer, face
    // and (x, y) in RGBA so both layer/face selection and s/t direction
    // reconstruction are checked on every face of every layer.
    let mut texels = Vec::with_capacity(12 * 16);
    for layer in 0..2u32 {
        for face in 0..6u32 {
            for y in 0..4u32 {
                for x in 0..4u32 {
                    texels.push([
                        (10 * layer + face) as f32,
                        x as f32,
                        y as f32,
                        (100 + layer) as f32,
                    ]);
                }
            }
        }
    }

    // v1 = s, v2 = t, v3 = layer, v4 = face:
    // v_fma_f32 v3, v3, 8.0, v4  (VOP3 0x14b, src1 = literal 0x4100_0000)
    // image_sample_lz v[0:3], v[1:3], s[8:15], s[16:19] dmask:0xf dim:CUBE
    let [fma_w0, fma_w1] = vop3(0x14b, 3, vgpr(3), 255, vgpr(4));
    let [img_w0, img_w1] = mimg_cube(0x27, 0, 1, 8, 16, 0xF, false);
    let shader = build_cube_cs(&[fma_w0, fma_w1, 0x4100_0000, img_w0, img_w1], 4);

    // Centre of texel (x=2, y=1) in [1, 2] face space: 1.0 + (idx + 0.5) / 4.0.
    let (s, t) = (1.625f32.to_bits(), 1.375f32.to_bits());
    let mut cases = Vec::new();
    let mut expected = Vec::new();
    for layer in 0..2u32 {
        // Faces 0..=5 plus out-of-range 6 and 7, which clamp to face 5 (-Z)
        // of the same layer on RDNA2.
        for face in 0..8u32 {
            cases.push([s, t, (layer as f32).to_bits(), (face as f32).to_bits()]);
            expected.push((layer, face.min(5)));
        }
    }
    run_cube_cases(
        &shader,
        View::CubeArray(4, 2, &texels),
        4,
        &cases,
        |backend, i, inp, got| {
            let (layer, clamped_face) = expected[i];
            let want_r = (10 * layer + clamped_face) as f32;
            let got_rgba = [got[0], got[1], got[2], got[3]].map(f32::from_bits);
            assert_eq!(
                got_rgba,
                [want_r, 2.0, 1.0, (100 + layer) as f32],
                "{backend}: layer={} face={} got {got_rgba:?}",
                f32::from_bits(inp[2]),
                f32::from_bits(inp[3]),
            );
        },
    );

    // Issue #720's literal-operand reproducer: v3 = 10.0 (layer 1, face 2 = +Y)
    // with no v_fma in the shader still infers TextureCubeArray and reads 12.0.
    let const_face = build_cube_cs(
        &[
            0x7E02_02FF,
            0x3FC0_0000, // v_mov_b32 v1, 1.5
            0x7E04_02FF,
            0x3FC0_0000, // v_mov_b32 v2, 1.5
            0x7E06_02FF,
            0x4120_0000, // v_mov_b32 v3, 10.0
            mimg_cube(0x27, 0, 1, 8, 16, 0x1, false)[0],
            mimg_cube(0x27, 0, 1, 8, 16, 0x1, false)[1],
        ],
        1,
    );
    run_cube_cases(
        &const_face,
        View::CubeArray(4, 2, &texels),
        1,
        &[[0; 4]],
        |backend, _, _, got| {
            assert_eq!(
                f32::from_bits(got[0]),
                12.0,
                "{backend}: constant face 10.0"
            );
        },
    );
}

#[test]
fn cube_samples_bare_faces_on_plain_cube_view() {
    // Plain 1-cube view (`D3D12_SRV_DIMENSION_TEXTURECUBE`): a bare face
    // operand stays `TextureCube` (kind 5) without regressing single cubes.
    let mut texels = Vec::with_capacity(6 * 16);
    for face in 0..6u32 {
        for _ in 0..16 {
            texels.push([face as f32, 0.0, 0.0, 1.0]);
        }
    }
    let [img_w0, img_w1] = mimg_cube(0x27, 0, 1, 8, 16, 0x1, false);
    let shader = build_cube_cs(&[img_w0, img_w1], 1);
    let mid = 1.5f32.to_bits();
    let cases: Vec<[u32; 4]> = (0..8u32)
        .map(|f| [mid, mid, (f as f32).to_bits(), 0])
        .collect();
    run_cube_cases(
        &shader,
        View::Cube(4, &texels),
        1,
        &cases,
        |backend, f, _, got| {
            let want = f.min(5) as f32;
            assert_eq!(
                f32::from_bits(got[0]),
                want,
                "{backend}: plain cube face {f}"
            );
        },
    );
}

#[test]
fn cube_array_gather_and_resinfo_match_rdna2() {
    // 3-cube array (18 faces, 4x4 per face): texel = 100 * layer + 10 * face + y * 4 + x.
    let mut texels = Vec::with_capacity(18 * 16);
    for layer in 0..3u32 {
        for face in 0..6u32 {
            for y in 0..4u32 {
                for x in 0..4u32 {
                    texels.push([(100 * layer + 10 * face + y * 4 + x) as f32; 4]);
                }
            }
        }
    }

    // `image_gather4_lz` (0x47, dmask 1) with `v_fma_f32 v3, v3, 8.0, v4`.
    let [fma_w0, fma_w1] = vop3(0x14b, 3, vgpr(3), 255, vgpr(4));
    let [gat_w0, gat_w1] = mimg_cube(0x47, 0, 1, 8, 16, 0x1, false);
    let gather_cs = build_cube_cs(&[fma_w0, fma_w1, 0x4100_0000, gat_w0, gat_w1], 4);
    // At s = t = 1.5 (centre of the 4x4 face, footprint across texels (1,1)..(2,2)),
    // all four gathered texels come from the requested (layer, face) and sum to
    // 4 * (100 * layer + 10 * face) + (5 + 6 + 9 + 10) = 400 * layer + 40 * face + 30.
    let mid = 1.5f32.to_bits();
    let gather_cases = [
        ([mid, mid, 0.0f32.to_bits(), 0.0f32.to_bits()], 0, 0),
        ([mid, mid, 1.0f32.to_bits(), 2.0f32.to_bits()], 1, 2),
        ([mid, mid, 2.0f32.to_bits(), 5.0f32.to_bits()], 2, 5),
    ];
    let g_in: Vec<[u32; 4]> = gather_cases.iter().map(|c| c.0).collect();
    run_cube_cases(
        &gather_cs,
        View::CubeArray(4, 3, &texels),
        4,
        &g_in,
        |backend, i, _, got| {
            let (_, layer, face) = gather_cases[i];
            let vals = [got[0], got[1], got[2], got[3]].map(f32::from_bits);
            let sum: f32 = vals.iter().sum();
            let want_sum = (400 * layer + 40 * face + 30) as f32;
            assert_eq!(
                sum, want_sum,
                "{backend}: gather4 layer={layer} face={face} got {vals:?}"
            );
        },
    );

    // Standalone `image_get_resinfo v[0:3], v1, s[8:15] dmask:0xf dim:CUBE`:
    // RDNA2 returns [width, height, 6 * cubes, levels] = [4, 4, 18, 1] for 3 cubes,
    // and [4, 4, 12, 1] for 2 cubes.
    let [res_w0, res_w1] = mimg_cube(0x0e, 0, 1, 8, 0, 0xF, false);
    let resinfo_cs = build_cube_cs(&[res_w0, res_w1], 4);
    run_cube_cases(
        &resinfo_cs,
        View::CubeArray(4, 3, &texels),
        4,
        &[[0; 4]],
        |backend, _, _, got| {
            assert_eq!(&got[..4], &[4, 4, 18, 1], "{backend}: 3-cube resinfo");
        },
    );
    run_cube_cases(
        &resinfo_cs,
        View::CubeArray(4, 2, &texels[..12 * 16]),
        4,
        &[[0; 4]],
        |backend, _, _, got| {
            assert_eq!(&got[..4], &[4, 4, 12, 1], "{backend}: 2-cube resinfo");
        },
    );
}


#[test]
fn glc_heap_uav_loads_stores_and_atomics_execute_on_gpu() {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // Compute shader with `RootType::ResourceHeap` (`s2`) and 2 root constants
    // (`s0..s1`), 2 threads per group (`v0` = thread id in group), dispatched
    // across 2 groups (`s3` = group id X):
    // - Heap index 0 (`0x0`): raw UAV buffer (16 dwords)
    // - Heap index 1 (`0x20`): 2x2 RGBA32F texture UAV
    // - Heap index 2 (`0x40`): 2x1 R32_UINT texture UAV
    // Exercises heap UAV inference (#724) and `IsGloballyCoherent` `0x4000`
    // (#725) across `buffer_store_dword glc`, `s_barrier`, reloaded descriptor
    // `s_buffer_load_dword glc`, `buffer_load_dword` (plain and `glc`),
    // `buffer_atomic_add glc`, `image_store glc`, `image_load glc`, and
    // `image_atomic_add glc`.
    let lshl_add = vop3(0x346, 1, sgpr(3), 0x81, vgpr(0)); // v_lshl_add_u32 v1, s3, 1, v0
    let lshl_byte = vop3(0x11A, 2, 0x82, vgpr(1), 0); // v_lshlrev_b32 v2, 2, v1
    let add_s0_v1 = vop3(0x125, 3, sgpr(0), vgpr(1), 0); // v_add_nc_u32 v3, s0, v1
    let add_v4_v5 = vop3(0x125, 4, vgpr(4), vgpr(5), 0); // v_add_nc_u32 v4, v4, v5
    let add_s6_v4 = vop3(0x125, 4, sgpr(6), vgpr(4), 0); // v_add_nc_u32 v4, s6, v4
    let words = [
        s_mov_b32(4, 2), // s4 = resource heap base (s2)
        s_mov_b32(5, 0x80),
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0   (raw UAV @ 0)
        0xF40C_0302,
        0xFA00_0020, // s_load_dwordx8 s[12:19], s[4:5], 0x20 (RGBA32F UAV @ 1)
        0xF40C_0502,
        0xFA00_0040, // s_load_dwordx8 s[20:27], s[4:5], 0x40 (R32_UINT UAV @ 2)
        // v1 = flat index = group_id_x * 2 + thread_id_x; v2 = v1 * 4 (byte offset).
        lshl_add[0],
        lshl_add[1],
        lshl_byte[0],
        lshl_byte[1],
        // v3 = s0 + v1; store with `glc` at byte offset v2.
        add_s0_v1[0],
        add_s0_v1[1],
        0xE070_5000,
        0x8002_0302, // buffer_store_dword v3, v2, s[8:11], 0 offen glc
        0xBF8A_0000, // s_barrier
        // Reload raw UAV @ 0 into s[28:31], read dword 0 via scalar `glc` and
        // lane's own slot via vector `glc` + plain load, and write sum to slot 4 + flat.
        0xF408_0702,
        0xFA00_0000, // s_load_dwordx4 s[28:31], s[4:5], 0x0
        0xF421_018E,
        0xFA00_0000, // s_buffer_load_dword s6, s[28:31], 0x0 glc
        0xE030_5000,
        0x8007_0402, // buffer_load_dword v4, v2, s[28:31], 0 offen glc
        0xE030_1000,
        0x8007_0502, // buffer_load_dword v5, v2, s[28:31], 0 offen
        add_v4_v5[0],
        add_v4_v5[1],
        add_s6_v4[0],
        add_s6_v4[1],
        0xE070_5010,
        0x8007_0402, // buffer_store_dword v4, v2, s[28:31], 0 offen offset:16 glc
        // Atomic add s1 into dword 8 of raw UAV with `glc`.
        vop1(0x001, 6, sgpr(1)), // v_mov_b32 v6, s1
        vop1(0x001, 7, 0x80),    // v_mov_b32 v7, 0
        0xE0C8_5020,
        0x8007_0607, // buffer_atomic_add v6, v7, s[28:31], 0 offen offset:32 glc
        // Texture UAV (2x2): store (1.0, 2.0, 3.0, 4.0) at (v0, s3) with `glc`,
        // load back with `glc`, and atomic_add 5 at (s3, 0) on R32U texture UAV.
        vop1(0x001, 8, vgpr(0)), // x = thread_id_x (0..1)
        vop1(0x001, 9, sgpr(3)), // y = group_id_x (0..1)
        vop1(0x001, 10, 0xF2),   // 1.0f
        vop1(0x001, 11, 0xF4),   // 2.0f
        vop1(0x001, 12, 0xF6),   // 4.0f
        vop1(0x001, 13, 0xF0),   // 0.5f
        0xF020_2F08,
        0x0003_0A08, // image_store v[10:13], v[8:9], s[12:19] dmask:0xf 2d glc
        0xBF8A_0000, // s_barrier
        0xF000_2F08,
        0x0003_0E08, // image_load v[14:17], v[8:9], s[12:19] dmask:0xf 2d glc
        0xE070_5024,
        0x8007_0E02, // buffer_store_dword v14, v2, s[28:31], 0 offen offset:36 glc
        vop1(0x001, 18, sgpr(3)), // x = group_id_x
        vop1(0x001, 19, 0x80), // y = 0
        vop1(0x001, 20, 0x85), // 5
        0xF044_2108,
        0x0005_1412, // image_atomic_add v20, v[18:19], s[20:27] dmask:0x1 2d glc
        S_ENDPGM,
    ];
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 24,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::ResourceHeap,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 2,
                slot: 1,
            },
        ],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo {
            // 3 user SGPRs (s0..s2) + TGID_X enabled (bit 7 -> s3).
            rsrc2: (3 << 1) | (1 << 7),
            threads: [2, 1, 1],
        }),
        code,
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

    let mut v = validator::Validator::load(validator::Validator::default_path().as_deref())
        .expect("dxil.dll validator must load");
    let mut copy = lowered.dxil.clone();
    let ours = copy[4..20].to_vec();
    v.validate(&mut copy).expect("DXIL validation failed");
    assert_eq!(copy[4..20], ours[..], "container hash mismatch");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        table_views: Vec::new(),
        stride_table: false,
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
    };

    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let inputs = [100u32, 7u32];
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = inputs[u.dword as usize];
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[
                        View::RawUav(16),
                        View::TextureUav(2, 2),
                        View::UintTextureUav(2, 1),
                    ],
                    samplers: &[],
                    groups: [2, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        // Slots 0..4 hold 100..103.
        assert_eq!(&out[0][0..4], &[100, 101, 102, 103], "{label}: raw stores");
        // Group 0 saw its own s_barrier before reading dword 0 (100):
        // slot 4 = 100 + 100 + 100 = 300, slot 5 = 101 + 101 + 100 = 302.
        assert_eq!(&out[0][4..6], &[300, 302], "{label}: reloaded glc reads");
        // Slot 8 accumulated 4 atomic adds of 7 = 28.
        assert_eq!(out[0][8], 28, "{label}: buffer_atomic_add glc");
        // Slots 9..13 hold 1.0f read back through `image_load glc`.
        assert_eq!(
            &out[0][9..13],
            &[1.0f32.to_bits(); 4],
            "{label}: image_load glc readback"
        );
        // 2x2 RGBA32F texture UAV holds (1.0, 2.0, 4.0, 0.5) at all 4 texels.
        for px in out[1].as_chunks::<4>().0 {
            assert_eq!(
                *px,
                [1.0f32, 2.0, 4.0, 0.5].map(f32::to_bits),
                "{label}: texture store glc"
            );
        }
        // 2x1 R32_UINT texture UAV: 2 threads per group * 5 = 10 at (0,0) and (1,0).
        assert_eq!(&out[2][0..2], &[10, 10], "{label}: image_atomic_add glc");
        ran += 1;
    }
    assert!(
        ran == 2,
        "expected both hardware GPU and WARP to run, got {ran}"
    );
}

#[test]
fn structured_buffer_flat_byte_offset_loads_reach_elements_past_the_first() {
    use super::host::Stride;

    // 1-thread compute shader:
    //   s0: flat byte offset (root constant)
    //   s4: table holding raw UAV at +0 dwords and StructuredBuffer<56> SRV at +8 dwords
    // Reads the dword at `s0` via both `s_buffer_load_dword` and flat `buffer_load_dword offen`
    // (no `idxen`), storing both results to UAV[0] and UAV[1].
    let mut words = Vec::new();
    words.push(s_mov_b32(5, 0x80)); // s5 = 0
    words.extend_from_slice(&s_load_dwordx4(8, 4)); // s[8:11] = UAV at table+0
    // s_load_dwordx4 s[12:15], s[4:5], 32 (table+8 dwords = slot 1)
    words.extend_from_slice(&[
        0x3D << 26 | 0x02 << 18 | (12 << 6) | (4 >> 1),
        125 << 25 | 32,
    ]);
    // s_buffer_load_dword s16, s[12:15], s0
    words.extend_from_slice(&[0x3D << 26 | 0x08 << 18 | (16 << 6) | (12 >> 1), 0]);
    words.push(vop1(0x001, 1, sgpr(0))); // v1 = s0 (flat byte offset)
    // buffer_load_dword v2, v1, s[12:15], 0 offen
    words.extend_from_slice(&[
        0x38 << 26 | 0x0C << 18 | 0x1000,
        0x80 << 24 | ((12 >> 2) << 16) | (2 << 8) | 1,
    ]);
    words.push(vop1(0x001, 0, sgpr(16))); // v0 = s16
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.extend_from_slice(&buffer_store_dword(2, 8, 4));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 1,
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
        used: vec![
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 8,
                lower_bound: 0,
                stride: 56,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code,
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
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }

    // 3 elements * 56 bytes = 42 distinct dwords.
    let data: Vec<u32> = (0..42u32).map(|i| 0xA000_0000 | (i * 0x101 + 7)).collect();
    let views = [
        View::RawUav(2),
        View::Buffer {
            data: &data,
            stride: Stride::Structured(56),
        },
    ];

    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for &byte_off in &[0u32, 24, 52, 56, 80, 108, 112, 136, 164] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => byte_off,
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
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            let want = data[(byte_off / 4) as usize];
            assert_eq!(
                out[0][0], want,
                "{label}: s_buffer_load_dword at byte {byte_off} = {:#x}, want {want:#x}",
                out[0][0]
            );
            assert_eq!(
                out[0][1], want,
                "{label}: buffer_load_dword offen at byte {byte_off} = {:#x}, want {want:#x}",
                out[0][1]
            );
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn v_ldexp_f16_matches_rdna2() {
    // VOP2 `v_ldexp_f16 v0, v1, v2` (opcode 0x3b), preserving upper half from v3.
    let vop2_cs = build_valu_cs(&[
        vop1(0x001, 0, vgpr(3)),   // v_mov_b32 v0, v3 (sentinel upper half)
        vop2(0x3b, 0, vgpr(1), 2), // v_ldexp_f16 v0, v1, v2
    ]);
    // VOP3 `v_ldexp_f16 v0, |v1|, v2 clamp` (opcode 0x13b) with op_sel:[1, 0, 0, 1]
    // (reads high half of v1, low half of v2, writes high half of v0).
    let [mut w0, w1] = vop3(0x13b, 0, vgpr(1), vgpr(2), 0);
    w0 |= (1 << 8) | (1 << 11) | (1 << 14) | (1 << 15); // abs[0], opsel[0], opsel[3], clamp
    let vop3_cs = build_valu_cs(&[
        vop1(0x001, 0, vgpr(3)), // v_mov_b32 v0, v3 (sentinel lower half)
        w0,
        w1,
    ]);

    let sentinel = 0xCAFE_0000u32;
    // (f16 input, i16 exponent, expected f16 output)
    let cases: &[(u16, i16, u16)] = &[
        (0x3C00, 0, 0x3C00),         // 1.0 * 2^0 = 1.0
        (0x3E00, 3, 0x4A00),         // 1.5 * 2^3 = 12.0
        (0xC200, -2, 0xBA00),        // -3.0 * 2^-2 = -0.75
        (0x3C00, -15, 0x0200),       // 1.0 * 2^-15 = f16 subnormal 0x0200
        (0x0001, 24, 0x3C00),        // min f16 subnormal (2^-24) * 2^24 = 1.0
        (0x7BFF, 0, 0x7BFF),         // max finite f16 * 2^0 stays 65504 (no premature inf)
        (0x3C00, 16, 0x7C00),        // 1.0 * 2^16 = 65536 -> +inf (RTE overflow)
        (0xBC00, 1000, 0xFC00),      // -1.0 * 2^1000 -> -inf
        (0x0001, i16::MAX, 0x7C00),  // min subnormal * 2^32767 -> +inf
        (0x7BFF, i16::MIN, 0x0000),  // max finite * 2^-32768 -> +0.0
        (0xFBFF, i16::MIN, 0x8000),  // -max finite * 2^-32768 -> -0.0
        (0x0000, i16::MAX, 0x0000),  // +0.0 * 2^32767 = +0.0 (not NaN)
        (0x8000, i16::MAX, 0x8000),  // -0.0 * 2^32767 = -0.0 (not NaN)
        (0x7C00, i16::MIN, 0x7C00),  // +inf * 2^-32768 = +inf (not NaN)
        (0xFC00, i16::MIN, 0xFC00),  // -inf * 2^-32768 = -inf (not NaN)
    ];
    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(a, e, _)| [u32::from(a), u32::from(e as u16), sentinel, 0])
        .collect();
    run_cases(&vop2_cs, &inputs, |backend, i, _, got| {
        let (a, e, want_h) = cases[i];
        let want = sentinel | u32::from(want_h);
        assert_eq!(
            got, want,
            "{backend}: v_ldexp_f16({a:#06x}, {e}) = {got:#010x}, want {want:#010x}",
        );
    });

    // NaN stays NaN (upper half preserved).
    run_cases(&vop2_cs, &[[0x7E00, 4, sentinel, 0]], |backend, _, _, got| {
        assert_eq!(got & 0xFFFF_0000, sentinel);
        let lo = (got & 0x7FFF) as u16;
        assert!(lo > 0x7C00, "{backend}: v_ldexp_f16(NaN, 4) lo={lo:#06x}, want NaN");
    });

    // VOP3 abs[0] + opsel:[1,0,0,1] + clamp:
    // |-0.25(=0xB400)| * 2^1 = 0.5 (0x3800) into high half; |-1.0(=0xBC00)| * 2^2 = 4.0 -> clamped 1.0 (0x3C00)
    let lo_sentinel = 0x0000_1234u32;
    let vop3_cases: &[([u32; 4], u32)] = &[
        ([0xB400_0000, 1, lo_sentinel, 0], (0x3800u32 << 16) | lo_sentinel),
        ([0xBC00_0000, 2, lo_sentinel, 0], (0x3C00u32 << 16) | lo_sentinel),
    ];
    let vop3_in: Vec<[u32; 4]> = vop3_cases.iter().map(|c| c.0).collect();
    run_cases(&vop3_cs, &vop3_in, |backend, i, _, got| {
        let want = vop3_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_ldexp_f16_e64 case {i} = {got:#010x}, want {want:#010x}",
        );
    });
}

#[test]
fn v_pk_fmac_f16_matches_rdna2() {
    // VOP2 `v_pk_fmac_f16 v0, v1, v2` (opcode 0x3c), initialized from v3 (root constant s2).
    let vop2_cs = build_valu_cs(&[
        vop1(0x001, 0, vgpr(3)),          // v_mov_b32 v0, v3 (accumulator)
        vop2(0x3c, 0, vgpr(1), 2),        // v_pk_fmac_f16 v0, v1, v2
    ]);
    // VOP2 with inline constant 1.0 (0xF2): GFX10 puts 1.0 in lo f16 and 0.0 in hi f16.
    let vop2_inline_cs = build_valu_cs(&[
        vop1(0x001, 0, vgpr(3)),          // v_mov_b32 v0, v3
        vop2(0x3c, 0, 0xF2, 2),           // v_pk_fmac_f16 v0, 1.0, v2
    ]);
    // VOP3 `v_pk_fmac_f16 v0, |v1|, -v2 clamp` (opcode 0x13c): abs[0], neg[1], clamp.
    let [mut w0, mut w1] = vop3(0x13c, 0, vgpr(1), vgpr(2), 0);
    w0 |= (1 << 8) | (1 << 15); // abs[0], clamp
    w1 |= 1 << 30;              // neg[1]
    let vop3_mod_cs = build_valu_cs(&[
        vop1(0x001, 0, vgpr(3)), // v_mov_b32 v0, v3
        w0,
        w1,
    ]);

    let pack = |lo: u16, hi: u16| u32::from(lo) | (u32::from(hi) << 16);
    // f16 constants: 0.0=0x0000, 0.5=0x3800, 1.0=0x3C00, 2.0=0x4000, 3.0=0x4200,
    // 4.0=0x4400, 5.0=0x4500, 7.0=0x4700, 10.0=0x4900, -2.0=0xC000, -3.0=0xC200.
    let vop2_cases: &[([u32; 4], u32)] = &[
        // lo: 2.0 * 3.0 + 1.0 = 7.0; hi: -2.0 * 4.0 + 10.0 = 2.0
        (
            [pack(0x4000, 0xC000), pack(0x4200, 0x4400), pack(0x3C00, 0x4900), 0],
            pack(0x4700, 0x4000),
        ),
        // lo: 0.5 * 4.0 + 3.0 = 5.0; hi: 0.0 * 7.0 + (-3.0) = -3.0
        (
            [pack(0x3800, 0x0000), pack(0x4400, 0x4700), pack(0x4200, 0xC200), 0],
            pack(0x4500, 0xC200),
        ),
    ];
    let vop2_in: Vec<[u32; 4]> = vop2_cases.iter().map(|c| c.0).collect();
    run_cases(&vop2_cs, &vop2_in, |backend, i, inp, got| {
        let want = vop2_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_pk_fmac_f16({:#010x}, {:#010x}, acc={:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });

    // Inline 1.0: lo = 1.0 * 3.0 + 2.0 = 5.0; hi = 0.0 * 4.0 + 7.0 = 7.0 (unchanged).
    run_cases(
        &vop2_inline_cs,
        &[[0, pack(0x4200, 0x4400), pack(0x4000, 0x4700), 0]],
        |backend, _, _, got| {
            let want = pack(0x4500, 0x4700);
            assert_eq!(
                got, want,
                "{backend}: v_pk_fmac_f16(1.0, ...) = {got:#010x}, want {want:#010x}",
            );
        },
    );

    // VOP3 abs[0], neg[1], clamp:
    // lo: |-2.0| * (-0.5) + 1.5(=0x3E00) = 0.5 (in [0, 1]) -> 0x3800
    // hi: |-2.0| * (-2.0) + 1.0 = -3.0 -> clamped to 0.0 (0x0000)
    // case 2: lo: |-2.0| * (-(-1.0)) + 0.5 = 2.5 -> clamped to 1.0 (0x3C00)
    //         hi: |0.5| * (-(-0.5)) + 0.25(=0x3400) = 0.5 (0x3800)
    let vop3_cases: &[([u32; 4], u32)] = &[
        (
            [pack(0xC000, 0xC000), pack(0x3800, 0x4000), pack(0x3E00, 0x3C00), 0],
            pack(0x3800, 0x0000),
        ),
        (
            [pack(0xC000, 0x3800), pack(0xBC00, 0xB800), pack(0x3800, 0x3400), 0],
            pack(0x3C00, 0x3800),
        ),
    ];
    let vop3_in: Vec<[u32; 4]> = vop3_cases.iter().map(|c| c.0).collect();
    run_cases(&vop3_mod_cs, &vop3_in, |backend, i, inp, got| {
        let want = vop3_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_pk_fmac_f16_e64(|{:#010x}|, -{:#010x}, acc={:#010x}) clamp = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn v_div_fixup_f32_matches_rdna2() {
    // VOP3 v_div_fixup_f32 v0, v1 (quot), v2 (denom), v3 (num) — opcode 0x15f
    let shader = build_valu_cs(&vop3(0x15f, 0, vgpr(1), vgpr(2), vgpr(3)));
    let p_zero = 0.0f32.to_bits();
    let n_zero = (-0.0f32).to_bits();
    let p_inf = f32::INFINITY.to_bits();
    let n_inf = f32::NEG_INFINITY.to_bits();
    let qnan = 0x7FC0_0000u32;
    let snan1 = 0x7F80_0001u32;
    let snan2 = 0x7F80_0002u32;
    let f_pos = 3.5f32.to_bits();
    let f_neg = (-3.5f32).to_bits();
    let two = 2.0f32.to_bits();
    let m_two = (-2.0f32).to_bits();
    let seven = 7.0f32.to_bits();
    let m_seven = (-7.0f32).to_bits();

    // (quot, denom, num) -> expected exact bits (None = any NaN)
    let cases: &[([u32; 4], Option<u32>)] = &[
        // Normal quotients: sign = sign(denom) ^ sign(num).
        ([f_pos, two, seven, 0], Some(f_pos)),
        ([f_pos, m_two, seven, 0], Some(f_neg)),
        ([f_neg, m_two, m_seven, 0], Some(f_pos)),
        // NaN quotient with finite non-zero denom/num -> ±inf with sign_out.
        ([qnan, two, seven, 0], Some(p_inf)),
        ([qnan, m_two, seven, 0], Some(n_inf)),
        // Division by ±0 -> ±inf with sign_out.
        ([f_pos, p_zero, seven, 0], Some(p_inf)),
        ([f_pos, n_zero, seven, 0], Some(n_inf)),
        ([f_pos, n_zero, m_seven, 0], Some(p_inf)),
        // ±inf / finite -> ±inf with sign_out.
        ([f_pos, m_two, p_inf, 0], Some(n_inf)),
        ([f_pos, m_two, n_inf, 0], Some(p_inf)),
        // finite / ±inf and ±0 / finite -> ±0.0 with sign_out.
        ([f_pos, n_inf, seven, 0], Some(n_zero)),
        ([f_pos, n_inf, m_seven, 0], Some(p_zero)),
        ([f_pos, m_two, p_zero, 0], Some(n_zero)),
        ([f_pos, m_two, n_zero, 0], Some(p_zero)),
        // Indeterminate forms: 0/0 and inf/inf -> qNaN.
        ([f_pos, p_zero, p_zero, 0], None),
        ([f_pos, n_zero, p_zero, 0], None),
        ([f_pos, p_inf, n_inf, 0], None),
        // NaN in denom or num propagates quieted NaN (denom takes priority).
        ([f_pos, snan1, snan2, 0], Some(0x7FC0_0001)),
        ([f_pos, two, snan2, 0], Some(0x7FC0_0002)),
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        match cases[i].1 {
            Some(want) => assert_eq!(
                got, want,
                "{backend}: v_div_fixup_f32({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
                inp[0], inp[1], inp[2],
            ),
            None => assert!(
                f32::from_bits(got).is_nan(),
                "{backend}: v_div_fixup_f32({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want NaN",
                inp[0], inp[1], inp[2],
            ),
        }
    });
}

#[test]
fn v_div_fixup_f16_matches_rdna2() {
    // VOP3 v_div_fixup_f16 v0, v1 (quot), v2 (denom), v3 (num) — opcode 0x35f
    let shader = build_valu_cs(&vop3(0x35f, 0, vgpr(1), vgpr(2), vgpr(3)));
    // f16 bit patterns: +0=0x0000, -0=0x8000, +1=0x3C00, -2=0xC000, +3.5=0x4300, -3.5=0xC300,
    // +inf=0x7C00, -inf=0xFC00, qNaN=0x7E00
    let cases: &[([u32; 4], Option<u16>)] = &[
        ([0x4300, 0xC000, 0x3C00, 0], Some(0xC300)), // copysign(3.5, -2 ^ +1) = -3.5
        ([0xC300, 0xC000, 0xBC00, 0], Some(0x4300)), // copysign(-3.5, -2 ^ -1) = +3.5
        ([0x7E00, 0xC000, 0x3C00, 0], Some(0xFC00)), // NaN quot -> -inf
        ([0x4300, 0x8000, 0x3C00, 0], Some(0xFC00)), // +1 / -0 -> -inf
        ([0x4300, 0xFC00, 0x3C00, 0], Some(0x8000)), // +1 / -inf -> -0.0
        ([0x4300, 0xC000, 0x0000, 0], Some(0x8000)), // +0 / -2 -> -0.0
        ([0x4300, 0x0000, 0x0000, 0], None),         // 0 / 0 -> NaN
        ([0x4300, 0x7C00, 0xFC00, 0], None),         // -inf / +inf -> NaN
        ([0x4300, 0x7E00, 0x3C00, 0], None),         // NaN denom -> NaN
    ];
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let got_h = got as u16;
        match cases[i].1 {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_div_fixup_f16({:#06x}, {:#06x}, {:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0], inp[1], inp[2],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_div_fixup_f16({:#06x}, {:#06x}, {:#06x}) = {got_h:#06x}, want NaN",
                inp[0], inp[1], inp[2],
            ),
        }
    });
}
