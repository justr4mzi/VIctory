//! Tests of descriptors that no instruction names alone (cont.):
use super::*;
use crate::native::{ComputeIo, RootParam, StageIo};

/// Read the size dword of a 2D texture SRV and store it with the red
/// channel of its first texel. The scalar read comes before the image use.
fn texture_size_word(texture_offset: u32, reload: [u32; 2]) -> Result<HostShader, ShaderError> {
    let words = [
        0xBE81_0380, // s_mov_b32 s1, 0 (table address high)
        reload[0],
        reload[1],
        0xF40C_0200,
        0xFA00_0000 | texture_offset, // s_load_dwordx8 s[8:15], s[0:1], texture_offset
        0xF408_0600,
        0xFA00_0040, // s_load_dwordx4 s[24:27], s[0:1], 64 (output UAV)
        0x7E00_0280,
        0x7E02_0280, // v0, v1 = 0
        0xF000_1108,
        0x0002_0200, // image_load v2, v[0:1], s[8:15] dmask:1 dim:2D unorm
        0x7E06_0212, // v_mov_b32 v3, s18 (size dword)
        0xE070_1000,
        0x8006_0300, // buffer_store_dword v3, v0, s[24:27], 0 offen
        0xE070_1004,
        0x8006_0200, // buffer_store_dword v2, v0, s[24:27], 0 offen offset:4
        0xBF81_0000, // s_endpgm
    ];
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2,
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs)?;
    let st = structure::structure(&p)?;
    let io = io::raise(&p, &native)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let bytes = crate::dxbc::build(&[]);
    emit(&Container::parse(&bytes)?, &native, &p, &st, &io, &res)
}

/// s_load_dword s18, s[0:1], offset (soffset = null).
fn reload_size_word(offset: u32) -> [u32; 2] {
    [0xF400_0480, 0xFA00_0000 | offset]
}

#[test]
fn texture_size_word_matches_the_full_descriptor_reload() {
    for offset in [0, 32] {
        let full = texture_size_word(offset, [0xF40C_0400, 0xFA00_0000 | offset]).unwrap();
        let single = texture_size_word(offset, reload_size_word(offset + 8)).unwrap();
        assert!(full.dxil == single.dxil, "texture at byte offset {offset}");
    }
}

#[test]
fn texture_size_word_of_another_entry_is_refused() {
    let result = texture_size_word(0, reload_size_word(32 + 8));
    assert!(matches!(
        result,
        Err(ShaderError::Unsupported {
            what: "a descriptor read only as data",
            ..
        })
    ));
}

#[cfg(windows)]
#[test]
fn texture_size_word_contains_the_dimensions_of_the_bound_view() {
    use super::host::{Dispatch, Host, Root, View};

    let shader = texture_size_word(32, reload_size_word(32 + 8)).unwrap();
    let texels = [[2.0f32, 0.0, 0.0, 1.0]; 9 * 6];
    let other = [[1.0f32, 0.0, 0.0, 1.0]];
    let views = [
        View::Texture(1, 1, &other),
        View::Texture(9, 6, &texels),
        View::RawUav(2),
    ];
    // Dword 2 has the high bits of width - 1 and all of height - 1.
    let size_word = ((9 - 1) >> 2) | ((6 - 1) << 14);
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (name, warp) in [("hardware", false), ("WARP", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {name} device; skipped");
            continue;
        };
        let (root, pipeline) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{name}: compute pipeline: {e}"));
        let out = host
            .dispatch(
                &root,
                &pipeline,
                &Dispatch {
                    roots: vec![(
                        0,
                        Root::Constants(vec![0; shader.user_data_dwords() as usize]),
                    )],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{name}: dispatch: {e}"));
        assert_eq!(out[0], [size_word, 2.0f32.to_bits()], "{name}");
    }
}

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, validator};
use crate::native::{ResourceType, UsedResource};
use crate::{Container, dxbc, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// A wave32 compute program of one thread: 4 root constants in s0..s3 and, in s4, a table
/// with a texture's descriptor at dword 0 and a raw UAV's at dword 8; `body` follows the
/// loads of the texture's descriptor into s[8:15] and the UAV's into s[16:19].
fn native(body: &[u32]) -> NativeProgram {
    native_of(body, 8, 0)
}

/// [`native`] with a view of `dwords` dwords (8: a texture, 4: a buffer of elements of
/// `stride` bytes) at dword 0, loaded into s8 on.
fn native_of(body: &[u32], dwords: u8, stride: u32) -> NativeProgram {
    let load = match dwords {
        8 => 0xF40C_0202, // s_load_dwordx8 s[8:15], s[4:5], null
        _ => 0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], null
    };
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        load,
        0xFA00_0000,
        0xF408_0402,
        0xFA00_0020, // s_load_dwordx4 s[16:19], s[4:5], 0x20
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    let range = |kind, table_offset_dwords, dwords, stride| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords,
        lower_bound: 0,
        stride,
        dwords,
        user_data: 4,
        slot: 1,
    };
    NativeProgram {
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
        ],
        used: vec![
            range(ResourceType::Srv, 0, dwords, stride),
            range(ResourceType::Uav, 8, 4, 0),
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `native` translated as the title's programs are, and validated.
fn translate(native: &NativeProgram) -> HostShader {
    let container = dxbc::build(&[]);
    let c = Container::parse(&container).expect("container");
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, native).expect("io::raise");
    let shader = super::emit(&c, native, &p, &st, &io, &res).expect("emit");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// Runs `shader` with root constant 0 at `s0`, the texture and a UAV of 4 dwords at heap
/// indices 0 and 1, on the hardware GPU and on WARP (each skipped without a device), and
/// returns per one that ran the UAV's dwords.
fn run(shader: &HostShader, s0: u32, texture: View) -> Vec<(&'static str, Vec<u32>)> {
    let views = [texture, View::RawUav(4)];
    let mut out = Vec::new();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        // The constant is s0; the table starts at heap index 0.
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.dword) {
                (RootType::Constants, 0) => s0,
                _ => 0,
            };
        }
        let words = host
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
        out.push((label, words[0].clone()));
    }
    out
}

#[test]
fn a_texture_read_only_for_its_size_gives_the_views_size() {
    // The console compiler's `GetDimensions` of a texture the program does not sample: width
    // and height from the descriptor's bits, and its last level; stored to the UAV.
    let shader = translate(&native(&[
        0x9014_9E09, // s_lshr_b32 s20, s9, 30
        0x8F15_820A, // s_lshl_b32 s21, s10, 2
        0x8814_1514, // s_or_b32 s20, s20, s21
        0x8714_FF14,
        0x0000_3FFF, // s_and_b32 s20, s20, 0x3fff
        0x8114_8114, // s_add_i32 s20, s20, 1
        0x9396_FF0A,
        0x000E_000E, // s_bfe_u32 s22, s10, 0x000e000e (bits 14-27)
        0x8116_8116, // s_add_i32 s22, s22, 1
        0x9397_FF0B,
        0x0004_0010, // s_bfe_u32 s23, s11, 0x00040010 (bits 16-19)
        0x7E02_0214, // v_mov_b32 v1, s20
        0x7E04_0216, // v_mov_b32 v2, s22
        0x7E06_0217, // v_mov_b32 v3, s23
        0xE070_0000,
        0x8004_0100, // buffer_store_dword v1, off, s[16:19], 0
        0xE070_0004,
        0x8004_0200, // buffer_store_dword v2, off, s[16:19], 0 offset:4
        0xE070_0008,
        0x8004_0300, // buffer_store_dword v3, off, s[16:19], 0 offset:8
    ]));
    let texels = [[0.0f32; 4]; 13 * 7];
    let results = run(&shader, 0, View::Texture(13, 7, &texels));
    for (label, words) in results {
        // 13 x 7 texels, one level (the last is level 0).
        assert_eq!(words, [13, 7, 0, 0], "{label}");
    }
}

#[test]
fn a_buffer_read_only_for_its_size_gives_the_views_element_count() {
    // The title's water simulation: whether a structured buffer has any elements, from its
    // descriptor's record count, the buffer itself not read; stored to the UAV.
    let shader = translate(&native_of(
        &[
            0x7E02_020A, // v_mov_b32 v1, s10
            0xE070_0000,
            0x8004_0100, // buffer_store_dword v1, off, s[16:19], 0
        ],
        4,
        16,
    ));
    let data = [0u32; 4 * 9];
    let buffer = View::Buffer {
        data: &data,
        stride: super::host::Stride::Structured(16),
    };
    for (label, words) in run(&shader, 0, buffer) {
        // nine elements of 16 bytes
        assert_eq!(words[0], 9, "{label}");
    }
}

#[test]
fn a_sampler_each_arm_of_a_uniform_branch_sets_samples_as_that_arm_says() {
    // The title's terrain programs: an immediate sampler that clamps (0x92: clamp in x, y and
    // z) or wraps (0) by a root constant, joined before one sample. Point filtering at u =
    // 1.25 of a 2 x 1 texture: wrapped, texel 0 (red 1.0); clamped, texel 1 (red 2.0).
    let shader = translate(&native(&[
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0xBF85_0002, // s_cbranch_scc1 -> s_mov_b32 s20, 0
        0xB014_0092, // s_movk_i32 s20, 0x92
        0xBF82_0001, // s_branch -> the join
        0xBE94_0380, // s_mov_b32 s20, 0
        0xBE95_03FF,
        0x00C0_0000, // s_mov_b32 s21, 0x00c00000 (levels up to 12)
        0xBE96_0380, // s_mov_b32 s22, 0 (point filtering)
        0xBE97_0380, // s_mov_b32 s23, 0
        0x7E02_02FF,
        0x3FA0_0000, // v_mov_b32 v1, 1.25
        0x7E04_02FF,
        0x3F00_0000, // v_mov_b32 v2, 0.5
        0xF09C_0F08,
        0x00A2_0301, // image_sample_lz v[3:6], v[1:2], s[8:15], s[20:23] dmask:0xf 2D
        0xE070_0000,
        0x8004_0300, // buffer_store_dword v3, off, s[16:19], 0
    ]));
    // One immediate sampler per arm, each the arm's own.
    assert_eq!(shader.static_samplers.len(), 2);
    let texels = [[1.0f32, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]];
    for (s0, red) in [(0, 1.0f32), (1, 2.0)] {
        for (label, words) in run(&shader, s0, View::Texture(2, 1, &texels)) {
            assert_eq!(f32::from_bits(words[0]), red, "{label}, s0 = {s0}");
        }
    }
}
