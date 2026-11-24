//! Synthetic GFX10 mesh shader tests for `GS_ALLOC_REQ` `M0` static upper-bound
//! narrowing, runtime `setMeshOutputCounts` lowering, the 32 KB DXIL output
//! limit, D3D12 Mesh Shader PSO creation (RTX 5070 Ti + WARP), and negative
//! soundness cases.

use super::container::{self, SigElement};
use super::{HostShader, emit};
use crate::dxbc::{Container, FourCC};
use crate::native::{
    HwStage, MeshIo, NativeProgram, RootParam, RootType, StageIo, Usage, VsOutput,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_SENDMSG_GS_ALLOC_REQ: u32 = 0xBF90_0009;
const M0: u8 = 124;
const VCC_LO: u8 = 106;
const EXEC_LO: u8 = 126;
// `HwStage::Ms` has 8 system SGPRs (`s0..s7`), so user data starts at `s8`.
const UD0: u8 = 8;
const UD1: u8 = 9;
const TMP0: u8 = 10;
const TMP1: u8 = 11;
const TMP2: u8 = 12;

const fn inline_u32(n: u8) -> u8 {
    128 + n
}

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0300 | ((sdst as u32) << 16) | (src0 as u32)
}

const fn s_mov_b32_lit(sdst: u8, lit: u32) -> [u32; 2] {
    [s_mov_b32(sdst, 0xFF), lit]
}

const fn s_and_saveexec_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_3C00 | ((sdst as u32) << 16) | (src0 as u32)
}

const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | ((op as u32 & 0x7F) << 23)
        | ((sdst as u32 & 0x7F) << 16)
        | ((src1 as u32) << 8)
        | (src0 as u32)
}

const fn sop2_lit(op: u8, sdst: u8, src0: u8, lit: u32) -> [u32; 2] {
    [sop2(op, sdst, src0, 0xFF), lit]
}

const fn s_min_u32(sdst: u8, src0: u8, src1: u8) -> u32 {
    sop2(0x07, sdst, src0, src1)
}

const fn s_min_u32_lit(sdst: u8, src0: u8, lit: u32) -> [u32; 2] {
    sop2_lit(0x07, sdst, src0, lit)
}

const fn s_cselect_b32(sdst: u8, src0: u8, src1: u8) -> u32 {
    sop2(0x0A, sdst, src0, src1)
}

const fn s_and_b32_lit(sdst: u8, src0: u8, lit: u32) -> [u32; 2] {
    sop2_lit(0x0E, sdst, src0, lit)
}

const fn s_or_b32(sdst: u8, src0: u8, src1: u8) -> u32 {
    sop2(0x10, sdst, src0, src1)
}

const fn s_lshl_b32(sdst: u8, src0: u8, src1: u8) -> u32 {
    sop2(0x1E, sdst, src0, src1)
}

const fn s_bfe_u32_lit(sdst: u8, src0: u8, lit: u32) -> [u32; 2] {
    sop2_lit(0x27, sdst, src0, lit)
}

const fn s_cmp_eq_u32(src0: u8, src1: u8) -> u32 {
    0xBF06_0000 | ((src1 as u32) << 8) | (src0 as u32)
}

const fn s_branch(simm16: i16) -> u32 {
    0xBF82_0000 | (simm16 as u16 as u32)
}

const fn s_cbranch_scc1(simm16: i16) -> u32 {
    0xBF85_0000 | (simm16 as u16 as u32)
}

const fn s_cbranch_execz(simm16: i16) -> u32 {
    0xBF88_0000 | (simm16 as u16 as u32)
}

const fn v_mov_b32(vdst: u8, src0: u8) -> u32 {
    0x7E00_0200 | ((vdst as u32) << 17) | (src0 as u32)
}

const fn v_mov_b32_lit(vdst: u8, lit: u32) -> [u32; 2] {
    [v_mov_b32(vdst, 0xFF), lit]
}

const fn v_cmp_eq_u32_v(src0: u8, vsrc1: u8) -> u32 {
    0x7C00_0000 | (0x0C2 << 17) | ((vsrc1 as u32) << 9) | (src0 as u32)
}

const fn v_mbcnt_lo_u32_b32(vdst: u8, src0: u16, src1: u16) -> [u32; 2] {
    [
        0x35 << 26 | (0x365 << 16) | (vdst as u32),
        (src0 as u32 & 0x1FF) | ((src1 as u32 & 0x1FF) << 9),
    ]
}

const fn exp_4(target: u8, v0: u8, v1: u8, v2: u8, v3: u8, done: bool) -> [u32; 2] {
    [
        0xF800_0000 | ((done as u32) << 11) | ((target as u32) << 4) | 0xF,
        ((v3 as u32) << 24) | ((v2 as u32) << 16) | ((v1 as u32) << 8) | (v0 as u32),
    ]
}

const fn exp_prim(v0: u8, done: bool) -> [u32; 2] {
    [
        0xF800_0000 | ((done as u32) << 11) | (20 << 4) | 0x1,
        v0 as u32,
    ]
}

fn sig_vec4(name: &str, index: u32, kind: u8, row: u32) -> SigElement {
    SigElement {
        name: name.into(),
        index,
        kind,
        comp: 9,
        interp: 0,
        row: Some(row),
        col: 0,
        cols: 4,
        usage: 0xF,
        half: false,
    }
}

/// Builds a synthetic GFX10 mesh shader with `prologue` instructions followed by
/// `extra_vert_rows` float4 vertex attributes (beyond `SV_Position`) and
/// `prim_rows` float4 per-primitive attributes.
fn build_mesh_shader(
    prologue: &[u32],
    threads: u32,
    ms_max_v: u32,
    ms_max_p: u32,
    extra_vert_rows: u8,
    prim_rows: u8,
    root_constants: u8,
) -> HostShader {
    let mut words = prologue.to_vec();
    // v0 = 1.0f (vertex/prim payload), v1 = triangle indices (0, 1, 2).
    words.push(v_mov_b32(0, 0xF2));
    words.extend_from_slice(&v_mov_b32_lit(1, (1 << 10) | (2 << 20)));
    words.extend_from_slice(&exp_4(12, 0, 0, 0, 0, false)); // exp pos0

    let mut ms_outputs = Vec::new();
    let mut osg = vec![sig_vec4("SV_Position", 0, 3, 0)];
    for k in 0..extra_vert_rows {
        let param = k;
        let row = u32::from(k) + 1;
        osg.push(sig_vec4("TEXCOORD", u32::from(k), 0, row));
        ms_outputs.push(VsOutput {
            usage: Usage::Generic,
            row: row as u8,
            param,
            fp16: false,
            pack2: false,
        });
        words.extend_from_slice(&exp_4(32 + param, 0, 0, 0, 0, false));
    }

    let mut psg = Vec::new();
    for k in 0..prim_rows {
        let param = extra_vert_rows + k;
        psg.push(sig_vec4("COLOR", u32::from(k), 0, u32::from(k)));
        ms_outputs.push(VsOutput {
            usage: Usage::Generic,
            row: k,
            param: 0x80 | param,
            fp16: false,
            pack2: false,
        });
        words.extend_from_slice(&exp_4(32 + param, 0, 0, 0, 0, false));
    }

    words.extend_from_slice(&exp_prim(1, true));
    words.push(S_ENDPGM);

    let layout = if root_constants > 0 {
        vec![RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: u32::from(root_constants),
            base_register: 0,
            user_data: 0,
            slot: 0,
        }]
    } else {
        Vec::new()
    };

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Ms,
        wave64: false,
        vgprs: 8,
        layout,
        used: Vec::new(),
        io: StageIo::Ms(MeshIo {
            outputs: ms_outputs,
            clip_cntl: 0,
            threads,
            max_vertices: ms_max_v,
            max_primitives: ms_max_p,
            topology: 2,
            lds_bytes: 0,
        }),
        code,
    };

    let osg1 = container::signature(&osg, false);
    let psg1 = container::signature(&psg, false);
    let mut parts = vec![(FourCC::OSG1, &osg1[..])];
    if !psg.is_empty() {
        parts.push((FourCC::PSG1, &psg1[..]));
    }
    let bytes = crate::dxbc::build(&parts);
    let c = Container::parse(&bytes).unwrap();
    let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    emit(&c, &native, &p, &st, &io, &res).unwrap()
}

fn psv_mesh_bounds(shader: &HostShader) -> (u16, u16) {
    let host = Container::parse(&shader.dxil).unwrap();
    let psv = host.part(FourCC::PSV0).unwrap().data;
    let max_v = u16::from_le_bytes([psv[4 + 12], psv[4 + 13]]);
    let max_p = u16::from_le_bytes([psv[4 + 14], psv[4 + 15]]);
    (max_v, max_p)
}

#[cfg(windows)]
fn validate_dxil(shader: &HostShader) -> Result<(), String> {
    let path = super::validator::Validator::default_path();
    let Ok(mut v) = super::validator::Validator::load(path.as_deref()) else {
        return Ok(());
    };
    let mut copy = shader.dxil.clone();
    let ours = copy[4..20].to_vec();
    v.validate(&mut copy)?;
    assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    Ok(())
}

#[cfg(windows)]
fn assert_pso_on_gpu_and_warp(ms: &HostShader, draw_cases: &[Vec<u32>]) {
    use super::host::{Draw, Host, Root, stub_ps};
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ps = stub_ps();
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .mesh_pipeline(ms, &ps)
            .unwrap_or_else(|e| panic!("{label}: mesh_pipeline failed: {e}"));
        for ud in draw_cases {
            host.draw(
                &root,
                &pso,
                &Draw {
                    roots: vec![(0, Root::Constants(ud.clone()))],
                    textures: &[],
                    samplers: &[],
                    vertices: &[],
                    stride: 0,
                    size: (8, 8),
                    mesh_groups: 1,
                },
            )
            .unwrap_or_else(|e| panic!("{label}: mesh draw ({ud:?}) failed: {e}"));
        }
    }
}

/// 7 vertex float4 rows (112 B) + 2 prim float4 rows (32 B) = 144 B/slot.
/// At 256/256 this is 36,864 B (> 32,768 B DXIL limit) and must be rejected
/// when no `GS_ALLOC_REQ` narrows it; at 64/64 it is 9,216 B and validates.
#[test]
#[cfg(windows)]
fn large_mesh_shader_exceeds_32kb_without_narrowing_and_passes_with_literal_m0() {
    let path = super::validator::Validator::default_path();
    if super::validator::Validator::load(path.as_deref()).is_err() {
        eprintln!("no dxil.dll; skipped");
        return;
    }

    // Without GS_ALLOC_REQ: stays 256/256 -> > 32,768 B -> dxil.dll rejects.
    let unnarrowed = build_mesh_shader(&[], 64, 256, 256, 6, 2, 0);
    assert_eq!(psv_mesh_bounds(&unnarrowed), (256, 256));
    assert!(
        validate_dxil(&unnarrowed).is_err(),
        "256/256 with 7 vert + 2 prim vec4s must exceed 32 KB limit"
    );

    // With literal M0 proving 64 verts / 64 prims: narrows to 64/64 -> validates & creates PSO.
    let mut prologue = Vec::new();
    prologue.extend_from_slice(&s_mov_b32_lit(M0, 64 | (64 << 12)));
    prologue.push(S_SENDMSG_GS_ALLOC_REQ);
    let narrowed = build_mesh_shader(&prologue, 64, 256, 256, 6, 2, 0);
    assert_eq!(psv_mesh_bounds(&narrowed), (64, 64));
    validate_dxil(&narrowed).expect("narrowed literal-M0 64/64 mesh shader must validate");
    assert_pso_on_gpu_and_warp(&narrowed, &[Vec::new()]);
}

#[test]
#[cfg(windows)]
fn s_min_u32_dynamic_m0_narrows_below_32kb_and_creates_pso() {
    // s10 = min(s8, 64); s11 = min(s9, 64); m0 = s10 | (s11 << 12); s_sendmsg(GS_ALLOC_REQ)
    let prologue = [
        s_min_u32(TMP0, UD0, inline_u32(64)),
        s_min_u32(TMP1, UD1, inline_u32(64)),
        s_lshl_b32(TMP1, TMP1, inline_u32(12)),
        s_or_b32(M0, TMP0, TMP1),
        S_SENDMSG_GS_ALLOC_REQ,
    ];
    let ms = build_mesh_shader(&prologue, 64, 256, 256, 6, 2, 2);
    assert_eq!(psv_mesh_bounds(&ms), (64, 64));
    assert_eq!(
        ms.user_data.len(),
        2,
        "runtime M0 must read both root constants"
    );
    validate_dxil(&ms).expect("s_min_u32-bounded dynamic M0 must validate under 32 KB");
    // Exercise runtime M0 counts (0/0, 3/1, and clamped >64) as well as PSO creation.
    assert_pso_on_gpu_and_warp(&ms, &[vec![0, 0], vec![3, 1], vec![100, 100]]);
}

#[test]
#[cfg(windows)]
fn guarded_lane0_dynamic_m0_materializes_and_creates_pso() {
    // Lane 0 computes M0 = min(s8, 64) | (min(s9, 48) << 12) inside an EXEC-guarded
    // branch; lowering must hoist the pure scalar computation to block 0 so
    // `setMeshOutputCounts` dominates all exports and passes dxil.dll + D3D12 PSO.
    let mut prologue = Vec::new();
    prologue.extend_from_slice(&v_mbcnt_lo_u32_b32(0, 193, 128)); // v_mbcnt_lo_u32_b32 v0, -1, 0
    prologue.push(v_cmp_eq_u32_v(inline_u32(0), 0)); // vcc = (v0 == 0)
    prologue.push(s_and_saveexec_b32(TMP2, VCC_LO));
    // Skip 5 dwords (s_min_u32, s_min_u32, s_lshl_b32, s_or_b32, s_sendmsg) if exec == 0.
    prologue.push(s_cbranch_execz(5));
    prologue.push(s_min_u32(TMP0, UD0, inline_u32(64)));
    prologue.push(s_min_u32(TMP1, UD1, inline_u32(48)));
    prologue.push(s_lshl_b32(TMP1, TMP1, inline_u32(12)));
    prologue.push(s_or_b32(M0, TMP0, TMP1));
    prologue.push(S_SENDMSG_GS_ALLOC_REQ);
    prologue.push(s_or_b32(EXEC_LO, EXEC_LO, TMP2));

    let ms = build_mesh_shader(&prologue, 64, 256, 256, 6, 2, 2);
    assert_eq!(psv_mesh_bounds(&ms), (64, 48));
    assert_eq!(
        ms.user_data.len(),
        2,
        "hoisted M0 must read both root constants"
    );
    validate_dxil(&ms).expect("lane-0 guarded dynamic M0 must validate");
    assert_pso_on_gpu_and_warp(&ms, &[vec![3, 1]]);
}

#[test]
fn and_bfe_select_and_phi_joins_narrow_soundly() {
    // s_bfe_u32 (6 bits at offset 0 -> <= 63) | (s_and_b32(..., 0x1F) << 12 -> <= 31).
    let mut bfe_and = Vec::new();
    bfe_and.extend_from_slice(&s_bfe_u32_lit(TMP0, UD0, 6 << 16));
    bfe_and.extend_from_slice(&s_and_b32_lit(TMP1, UD1, 0x1F));
    bfe_and.push(s_lshl_b32(TMP1, TMP1, inline_u32(12)));
    bfe_and.push(s_or_b32(M0, TMP0, TMP1));
    bfe_and.push(S_SENDMSG_GS_ALLOC_REQ);
    let ms_bfe = build_mesh_shader(&bfe_and, 64, 256, 256, 0, 0, 2);
    assert_eq!(psv_mesh_bounds(&ms_bfe), (63, 31));
    assert_eq!(ms_bfe.user_data.len(), 2);
    #[cfg(windows)]
    validate_dxil(&ms_bfe).unwrap();

    // s_cselect_b32 between (32, 48) and (64, 16) -> (64, 48).
    let mut csel = Vec::new();
    csel.push(s_cmp_eq_u32(UD0, inline_u32(0)));
    csel.extend_from_slice(&s_mov_b32_lit(TMP0, 32 | (48 << 12)));
    csel.extend_from_slice(&s_mov_b32_lit(TMP1, 64 | (16 << 12)));
    csel.push(s_cselect_b32(M0, TMP0, TMP1));
    csel.push(S_SENDMSG_GS_ALLOC_REQ);
    let ms_csel = build_mesh_shader(&csel, 64, 256, 256, 0, 0, 1);
    assert_eq!(psv_mesh_bounds(&ms_csel), (64, 48));
    #[cfg(windows)]
    validate_dxil(&ms_csel).unwrap();

    // Diamond Phi joining m0 = (24, 64) and m0 = (min(s8, 48), 32) -> (48, 64).
    let mut phi = Vec::new();
    phi.push(s_cmp_eq_u32(UD0, inline_u32(0)));
    phi.push(s_cbranch_scc1(3)); // skip fallthrough arm (2 + 1 words) -> else arm
    phi.extend_from_slice(&s_mov_b32_lit(M0, 24 | (64 << 12)));
    phi.push(s_branch(4)); // skip else arm (4 words)
    phi.push(s_min_u32(TMP0, UD0, inline_u32(48)));
    phi.push(s_mov_b32(TMP1, inline_u32(32)));
    phi.push(s_lshl_b32(TMP1, TMP1, inline_u32(12)));
    phi.push(s_or_b32(M0, TMP0, TMP1));
    phi.push(S_SENDMSG_GS_ALLOC_REQ);
    let ms_phi = build_mesh_shader(&phi, 64, 256, 256, 0, 0, 1);
    assert_eq!(psv_mesh_bounds(&ms_phi), (48, 64));
    #[cfg(windows)]
    validate_dxil(&ms_phi).unwrap();
}

#[test]
fn negative_unbounded_dynamic_m0_keeps_full_bounds() {
    // 1. Direct unbounded s8 -> M0 keeps full (128, 96) and still validates with clamped runtime M0.
    let ms_direct = build_mesh_shader(
        &[s_mov_b32(M0, UD0), S_SENDMSG_GS_ALLOC_REQ],
        64,
        128,
        96,
        0,
        0,
        1,
    );
    assert_eq!(psv_mesh_bounds(&ms_direct), (128, 96));
    assert_eq!(ms_direct.user_data.len(), 1);
    #[cfg(windows)]
    validate_dxil(&ms_direct).unwrap();

    // 2. Unbounded vertex count OR'd with bounded prim count keeps full (128, 96).
    let ms_half_unbounded = build_mesh_shader(
        &[
            s_min_u32(TMP1, UD1, inline_u32(32)),
            s_lshl_b32(TMP1, TMP1, inline_u32(12)),
            s_or_b32(M0, UD0, TMP1),
            S_SENDMSG_GS_ALLOC_REQ,
        ],
        64,
        128,
        96,
        0,
        0,
        2,
    );
    assert_eq!(psv_mesh_bounds(&ms_half_unbounded), (128, 96));

    // 3. 10-bit mask (& 0x3FF -> up to 1023 > 256) is not bounded below 256 -> keeps full (256, 256).
    let mut masked_10bit = Vec::new();
    masked_10bit.extend_from_slice(&s_and_b32_lit(TMP0, UD0, 0x3FF));
    masked_10bit.push(s_min_u32(TMP1, UD1, inline_u32(64)));
    masked_10bit.push(s_lshl_b32(TMP1, TMP1, inline_u32(12)));
    masked_10bit.push(s_or_b32(M0, TMP0, TMP1));
    masked_10bit.push(S_SENDMSG_GS_ALLOC_REQ);
    let ms_10bit = build_mesh_shader(&masked_10bit, 64, 256, 256, 0, 0, 2);
    assert_eq!(psv_mesh_bounds(&ms_10bit), (256, 256));

    // 4. Phi joining bounded (64, 64) with unbounded s8 keeps full (128, 96).
    let mut phi_unbounded = Vec::new();
    phi_unbounded.push(s_cmp_eq_u32(UD0, inline_u32(0)));
    phi_unbounded.push(s_cbranch_scc1(3));
    phi_unbounded.extend_from_slice(&s_mov_b32_lit(M0, 64 | (64 << 12)));
    phi_unbounded.push(s_branch(1));
    phi_unbounded.push(s_mov_b32(M0, UD0));
    phi_unbounded.push(S_SENDMSG_GS_ALLOC_REQ);
    let ms_phi_unbounded = build_mesh_shader(&phi_unbounded, 64, 128, 96, 0, 0, 1);
    assert_eq!(psv_mesh_bounds(&ms_phi_unbounded), (128, 96));

    // 5. Clamping: proven (200, 150) with ms=(128, 96) clamps to (128, 96).
    let mut clamp_ms = Vec::new();
    clamp_ms.extend_from_slice(&s_min_u32_lit(TMP0, UD0, 200));
    clamp_ms.extend_from_slice(&s_min_u32_lit(TMP1, UD1, 150));
    clamp_ms.push(s_lshl_b32(TMP1, TMP1, inline_u32(12)));
    clamp_ms.push(s_or_b32(M0, TMP0, TMP1));
    clamp_ms.push(S_SENDMSG_GS_ALLOC_REQ);
    let ms_clamped = build_mesh_shader(&clamp_ms, 64, 128, 96, 0, 0, 2);
    assert_eq!(psv_mesh_bounds(&ms_clamped), (128, 96));
}

#[test]
fn negative_multiple_gs_alloc_req_sites_take_max_and_require_all_proven() {
    // Two sites in two branches: (32, 80) and (64, 48) -> must take max (64, 80).
    let mut two_sites = Vec::new();
    two_sites.push(s_cmp_eq_u32(UD0, inline_u32(0)));
    two_sites.push(s_cbranch_scc1(4)); // skip site 1 arm (s_mov_b32_lit [2] + sendmsg [1] + s_branch [1] = 4)
    two_sites.extend_from_slice(&s_mov_b32_lit(M0, 32 | (80 << 12)));
    two_sites.push(S_SENDMSG_GS_ALLOC_REQ);
    two_sites.push(s_branch(3)); // skip site 2 arm (s_mov_b32_lit [2] + sendmsg [1] = 3)
    two_sites.extend_from_slice(&s_mov_b32_lit(M0, 64 | (48 << 12)));
    two_sites.push(S_SENDMSG_GS_ALLOC_REQ);

    let ms_max = build_mesh_shader(&two_sites, 64, 256, 256, 0, 0, 1);
    assert_eq!(psv_mesh_bounds(&ms_max), (64, 80));
    #[cfg(windows)]
    validate_dxil(&ms_max).unwrap();

    // Two sites where site 1 is bounded (32, 32) and site 2 is unbounded s8 -> must keep full (256, 256).
    let mut one_unproven = Vec::new();
    one_unproven.push(s_cmp_eq_u32(UD0, inline_u32(0)));
    one_unproven.push(s_cbranch_scc1(4));
    one_unproven.extend_from_slice(&s_mov_b32_lit(M0, 32 | (32 << 12)));
    one_unproven.push(S_SENDMSG_GS_ALLOC_REQ);
    one_unproven.push(s_branch(2));
    one_unproven.push(s_mov_b32(M0, UD0));
    one_unproven.push(S_SENDMSG_GS_ALLOC_REQ);

    let ms_unproven = build_mesh_shader(&one_unproven, 64, 256, 256, 0, 0, 1);
    assert_eq!(psv_mesh_bounds(&ms_unproven), (256, 256));
}

#[test]
#[cfg(windows)]
fn mixed_unbounded_vertex_and_small_const_prim_guards_primitive_exports() {
    // Unbounded vertex count (s8) | small constant primitive count (16) with threads=64:
    // overall bounds stay (256, 256), while setMeshOutputCounts emits (v, 16), so
    // body storePrimitiveOutput and emitIndices must still be guarded by `flat < 16`.
    let mut prologue = Vec::new();
    prologue.extend_from_slice(&s_and_b32_lit(TMP0, UD0, 0x3FF));
    prologue.push(s_mov_b32(TMP1, inline_u32(16)));
    prologue.push(s_lshl_b32(TMP1, TMP1, inline_u32(12)));
    prologue.push(s_or_b32(M0, TMP0, TMP1));
    prologue.push(S_SENDMSG_GS_ALLOC_REQ);

    // Include 1 prim COLOR output (prim_rows = 1) so storePrimitiveOutput + emitIndices both run.
    let ms = build_mesh_shader(&prologue, 64, 256, 256, 0, 1, 1);
    assert_eq!(psv_mesh_bounds(&ms), (256, 256));
    assert_eq!(ms.user_data.len(), 1);
    validate_dxil(&ms).expect("mixed unbounded-vertex + const-16 prim must validate");
    assert_pso_on_gpu_and_warp(&ms, &[vec![3], vec![64], vec![512]]);
}
