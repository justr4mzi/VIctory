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
