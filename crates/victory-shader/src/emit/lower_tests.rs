    use super::{Lowered, Numeric, Parts, TableView, buffer_format, half_shift, lower};
    use crate::native::{
        ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
    };
    use crate::{io, ir, resource, structure};

    const S_ENDPGM: u32 = 0xBF81_0000;

    #[test]
    fn packed_pixel_inputs_find_their_second_row() {
        use super::super::container::SigElement;
        use super::half_input_partners;
        use crate::native::{PsInput, Usage};
        let entry = |row, row2, pack2| PsInput {
            usage: Usage::Generic,
            row,
            mask: 0xF,
            attr: row,
            flat: false,
            fp16: true,
            pack2,
            row2,
        };
        let element = |row| SigElement {
            name: "TEXCOORD".into(),
            index: row,
            kind: 0,
            comp: 9,
            interp: 2,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half: true,
        };
        // Rows 0 and 3 pack and leave their second row out; 5 names its
        // own (6). Unnamed 16-bit rows: 1, 4, 6, 7.
        let io = StageIo::Ps {
            inputs: vec![entry(0, 0, true), entry(3, 3, true), entry(5, 6, true)],
            rsrc2: 0,
            input_addr: 0,
            db_shader_control: 0,
            patches: vec![],
        };
        let sig: Vec<_> = [0, 1, 3, 4, 5, 6, 7].map(element).into();
        let p = half_input_partners(&io, &sig);
        assert_eq!(p.get(&0), Some(&1));
        assert_eq!(p.get(&3), Some(&4));
        assert_eq!(p.get(&5), None);
    }

    const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
        0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
    }

    /// `s_load_dwordx4 s[dst:dst+3], s[base:base+1], offset` with `soffset`
    /// (`125` = NULL, `0..105` = SGPR).
    const fn s_load_dwordx4(dst: u8, base: u8, offset: u32, soffset: u8) -> [u32; 2] {
        [
            0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
            (soffset as u32) << 25 | (offset & 0x1F_FFFF),
        ]
    }

    const fn buffer_load_dword(vdata: u8, srsrc: u8, idxen: bool) -> [u32; 2] {
        [
            0x38 << 26 | 0x0C << 18 | (idxen as u32) << 13,
            0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
        ]
    }

    const fn buffer_store_dword(vdata: u8, srsrc: u8, idxen: bool) -> [u32; 2] {
        [
            0x38 << 26 | 0x1C << 18 | (idxen as u32) << 13,
            0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
        ]
    }

    fn lower_table_views(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
        user_sgprs: u8,
    ) -> Vec<TableView> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 8,
            layout,
            used,
            io: StageIo::Cs(ComputeIo {
                rsrc2: u32::from(user_sgprs) << 1,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, false, 8).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
        .unwrap()
        .table_views
    }

    /// Whether a synthetic compute program translates with these input registers.
    fn lowers(words: &[u32], rsrc2: u32) -> Result<(), crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, false, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
        .map(|_| ())
    }

    /// Lowers a compute program with the root parameters `layout` (in user
    /// SGPRs from s0 on) and the used resources `used`.
    fn lower_with(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
    ) -> Result<Lowered, crate::ShaderError> {
        lower_wave(words, layout, used, false)
    }

    /// [`lower_with`] in wave64 if `wave64`.
    fn lower_wave(
        words: &[u32],
        layout: Vec<RootParam>,
        used: Vec<UsedResource>,
        wave64: bool,
    ) -> Result<Lowered, crate::ShaderError> {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64,
            vgprs: 256,
            layout,
            used,
            io: StageIo::Cs(ComputeIo {
                // The layout's user SGPRs (at least two), 512 bytes of LDS.
                rsrc2: 1 << 15 | sgprs.max(2) << 1,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, wave64, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
    }

    /// Root parameter `slot` in user SGPRs from `user_data` on.
    fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs: if root == RootType::Descriptor { 2 } else { 1 },
            base_register: 0,
            user_data,
            slot,
        }
    }

    #[test]
    fn a_scalar_carry_chain_reaches_the_select() {
        // The high dword of a 64-bit add or subtract, then a select on the
        // carry (borrow) the high half leaves in SCC, and a branch on it
        // (only code with an effect needs its operands).
        let add = [
            0xBE80_03C1, // s_mov_b32 s0, -1
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8000_8100, // s_add_u32 s0, s0, 1
            0x8201_8001, // s_addc_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];
        let sub = [
            0xBE80_0380, // s_mov_b32 s0, 0
            0xBE81_0380, // s_mov_b32 s1, 0
            0x8080_8100, // s_sub_u32 s0, s0, 1
            0x8281_8001, // s_subb_u32 s1, s1, 0
            0x8502_0001, // s_cselect_b32 s2, s1, s0
            0xBF07_8002, // s_cmp_lg_u32 s2, 0
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE83_0380, // s_mov_b32 s3, 0
            S_ENDPGM,
        ];

        for words in [add, sub] {
            lowers(&words, 0).unwrap();
        }
    }

    #[test]
    fn m0_relative_moves_with_two_indices_translate() {
        // v1 = 5, v2 = 7, M0 = source offset 0, destination offset 1, then
        // the M0-relative instruction and an LDS store of its v1 (an effect,
        // so a value the emitter cannot produce fails the translation).
        let program = |rel: u32| {
            [
                0x7E02_0285, // v_mov_b32 v1, 5
                0x7E04_0287, // v_mov_b32 v2, 7
                0x7E06_0280, // v_mov_b32 v3, 0
                0xBEFC_03FF, // s_mov_b32 m0, literal
                0x0001_0000,
                rel,
                0xD834_0000, // ds_write_b32 v3, v1
                0x0000_0103,
                S_ENDPGM,
            ]
        };
        for rel in [
            0x7E02_D101, // v_swaprel_b32 v1, v1
            0x7E02_9101, // v_movrelsd_2_b32 v1, v1
        ] {
            lowers(&program(rel), 1 << 15).unwrap_or_else(|e| panic!("{rel:08x}: {e}"));
        }
    }

    fn perm_features(selector: u32) -> u64 {
        // Synthetic compute shader:
        // v0 is the thread ID and can supply a dynamic selector.
        // Store the result in LDS to keep the permutation live.
        let words = [
            0x7E02_02FFu32,
            0x817F_34AB,
            0x7E04_02FF,
            0xFE01_56CD,
            0xD744_0003,
            257 | (258 << 9) | (selector << 18),
            0x3408_0082,
            0xD834_0000,
            0x0000_0304,
            S_ENDPGM,
        ];

        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();

        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 1 << 15,
                threads: [64, 1, 1],
            }),
            code,
        };

        let p = ir::build(&native.code, false, 256).unwrap();

        assert!(
            p.blocks
                .iter()
                .flat_map(|b| &b.insts)
                .any(|i| i.name() == "v_perm_b32")
        );

        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();

        let io = io::raise(&p, &native).unwrap();

        let emitted = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
        .unwrap();

        let container = crate::Container::parse(&emitted.dxil).unwrap();

        let features = container.part(crate::FourCC::SFI0).unwrap();

        u64::from_le_bytes(features.data.try_into().unwrap())
    }

    #[test]
    fn perm_constant_selector_needs_no_int64() {
        assert_eq!(perm_features(128) & 0x8000, 0);
    }

    #[test]
    fn perm_dynamic_selector_needs_no_int64() {
        assert_eq!(
            perm_features(256) & 0x8000,
            0,
            "dynamic v_perm_b32 must not require INT64"
        );
    }

    #[test]
    fn a_descriptor_selected_by_scc_picks_its_heap_index() {
        // Two buffer descriptors loaded from the table at s0, one picked by
        // a root constant (s1) with s_cselect_b64 per half, and loaded
        // through by index and stored to LDS so the load is needed.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xF408_0302,
            0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
            0xBF06_8001, // s_cmp_eq_u32 s1, 0
            0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
            0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
            0xE030_2000,
            0x8004_0100, // buffer_load_dword v1, v0, s[16:19], 0 idxen
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        // Both are structured buffers of dwords in the table's range.
        let used = || {
            vec![UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }]
        };
        lower_with(&words, layout(), used()).unwrap();
    }

    #[test]
    fn a_descriptor_read_only_as_data_takes_the_kind_of_its_entrys_use() {
        // The table entry at s0 loaded twice: the second load's descriptor
        // reads a structured buffer, the first's record count (dword 2) is
        // only read as a number, as the console compiler does for a size.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
            0xF408_0302,
            0xFA00_0000, // s_load_dwordx4 s[12:15], s[4:5], null
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xE030_2000,
            0x8003_0100, // buffer_load_dword v1, v0, s[12:15], 0 idxen
            0x7E04_020A, // v_mov_b32 v2, s10
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            0xD834_0000,
            0x0000_0200, // ds_write_b32 v0, v2
            S_ENDPGM,
        ];
        let (layout, used) = fetch_tables(4, 4);
        lower_with(&words, layout, used).unwrap();
    }

    /// `v_mov_b32 vdst, src0` (`src0`: 128 is the inline zero).
    fn v_mov(vdst: u8, src0: u16) -> u32 {
        0x7E00_0000 | (vdst as u32) << 17 | 0x01 << 9 | (src0 as u32 & 0x1FF)
    }

    /// `ds_write_b32 v[addr], v[data]`.
    fn ds_write_b32(addr: u8, data: u8) -> [u32; 2] {
        [0xD834_0000, u32::from(addr) | u32::from(data) << 8]
    }

    /// A buffer access: MUBUF (`mtbuf` false, opcode in bits 18-25, a
    /// resource and optional `idxen`/`offen`) or MTBUF (`mtbuf` true, opcode
    /// in bits 15-18, the format in bits 19-25), `srsrc` the first of four
    /// resource SGPRs, `vdata`/`vaddr` VGPRs and `offset` a byte offset.
    #[allow(clippy::too_many_arguments)]
    fn buffer(
        opcode: u16,
        mtbuf: bool,
        format: u8,
        offset: u16,
        offen: bool,
        idxen: bool,
        vdata: u8,
        vaddr: u8,
        srsrc: u8,
    ) -> [u32; 2] {
        let class = if mtbuf { 0x3A } else { 0x38 };
        let op_shift = if mtbuf { 16 } else { 18 };
        [
            class << 26
                | u32::from(opcode & if mtbuf { 0x7 } else { 0x7F }) << op_shift
                | u32::from(format) << 19
                | u32::from(offen) << 12
                | u32::from(idxen) << 13
                | u32::from(offset),
            // soffset is the inline zero, TFE and GLC clear; MTBUF keeps
            // the top bit of its opcode here.
            0x80 << 24
                | u32::from(if mtbuf { opcode >> 3 } else { 0 }) << 21
                | u32::from(srsrc >> 2) << 16
                | u32::from(vdata) << 8
                | u32::from(vaddr),
        ]
    }

    /// A program that reads the descriptor at table entry 0 into `s[8:11]`,
    /// runs `fetch` (VGPRs from 0, its address in v3) and writes its VGPRs
    /// to LDS: the three positions of a vertex fetch, or four channels.
    fn fetch_program(fetch: [u32; 2], channels: u8) -> Vec<u32> {
        let mut words = vec![
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000,              // s_load_dwordx4 s[8:11], s[4:5], null
            v_mov(3, 128),            // v_mov_b32 v3, 0 (the fetch's address)
            v_mov(channels + 1, 128), // v_mov_b32 v5, 0 (the LDS address)
        ];
        words.extend_from_slice(&fetch);
        for k in 0..channels {
            words.extend_from_slice(&ds_write_b32(channels + 1, k));
        }
        words.push(S_ENDPGM);
        words
    }

    /// The descriptor at table entry 0: a typed or raw SRV of `stride` (0:
    /// none), read as `dwords` dwords.
    fn fetch_tables(stride: u32, dwords: u8) -> (Vec<RootParam>, Vec<UsedResource>) {
        let layout = vec![root(RootType::Table, 0, 0)];
        let used = vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride,
            dwords,
            user_data: 0,
            slot: 0,
        }];
        (layout, used)
    }

    #[test]
    fn every_buffer_format_names_its_bits_and_numbers() {
        // Every number names a format, 1 to 77, and no number past that
        // does: the element sizes the typed fetches read follow from these.
        for format in 1..=77u8 {
            let (bits, _) = buffer_format(format).unwrap_or_else(|| panic!("format {format}"));
            let total: u32 = bits.iter().sum();
            assert_eq!(total % 8, 0, "format {format}: {bits:?} is whole bytes");
            assert!(
                total == 8 || total == 16 || total >= 32,
                "format {format}: {bits:?} is a dword or a byte/half inside one"
            );
        }
        assert_eq!(buffer_format(78), None);
        // The vertex fetch shapes: float3 is 12 bytes (16_16_16_16 comes
        // first, at 65), half2 and unorm8x4 are 4.
        let bytes = |f| -> u32 { buffer_format(f).unwrap().0.iter().sum::<u32>() / 8 };
        assert_eq!(bytes(74), 12, "32_32_32_FLOAT");
        assert_eq!(bytes(24), 4, "16_16_SNORM");
        assert_eq!(bytes(56), 4, "8_8_8_8_UNORM");
    }

    #[test]
    fn a_typed_fetch_inside_a_dword_reads_the_pair_around_it() {
        // `tbuffer_load_format_xyz` at byte offset 33: the console reads
        // the bytes at 33, so the host reads the dword-aligned pair and
        // shifts (a dword read promised at 33 can return zeros).
        let odd = buffer(0x002, true, 74, 33, false, false, 0, 3, 8);
        let (layout, used) = fetch_tables(0, 4);
        let l = lower_with(&fetch_program(odd, 3), layout, used).unwrap();
        assert_eq!(l.unaligned_typed_loads, 1);
        // The same fetch on a dword boundary is one plain read.
        let even = buffer(0x002, true, 74, 32, false, false, 0, 3, 8);
        let (layout, used) = fetch_tables(0, 4);
        let l = lower_with(&fetch_program(even, 3), layout, used).unwrap();
        assert_eq!(l.unaligned_typed_loads, 0);
    }

    #[test]
    fn a_typed_fetch_of_no_format_is_refused() {
        // Format 0 is not a format: the fetch must stop rather than read
        // whatever the descriptor holds.
        let fetch = buffer(0x002, true, 0, 0, true, false, 0, 3, 8);
        let (layout, used) = fetch_tables(0, 4);
        let err = match lower_with(&fetch_program(fetch, 3), layout, used) {
            Err(e) => e,
            Ok(_) => panic!("a fetch of format 0 lowered"),
        };
        assert!(
            matches!(
                err,
                crate::ShaderError::Unsupported {
                    what: "a buffer format",
                    ..
                }
            ),
            "{err:?}"
        );
    }
    /// Tables of 2D textures at s0 and samplers at s1.
    fn texture_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
        let range = |kind, user_data, slot| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: if kind == ResourceType::Sampler { 4 } else { 8 },
            user_data,
            slot,
        };
        (
            vec![root(RootType::Table, 0, 0), root(RootType::Table, 1, 1)],
            vec![
                range(ResourceType::Srv, 0, 0),
                range(ResourceType::Sampler, 1, 1),
            ],
        )
    }

    #[test]
    fn waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own() {
        use crate::resource::tests::{
            WATERFALL_FETCHED_WAVE32, WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32,
        };
        for (words, wave64) in [
            (WATERFALL_FETCHED_WAVE32, false),
            (WATERFALL_FETCHED_WAVE64, true),
            (WATERFALL_PICKED_WAVE32, false),
        ] {
            let (layout, used) = texture_tables();
            let l = lower_wave(words, layout, used, wave64).unwrap();
            // The loop's reads of the first lane became each lane's own
            // dwords: no wave operation is left.
            assert_eq!(l.nonuniform_handles, 1);
            assert!(!l.wave_ops);
        }
    }

    #[test]
    fn an_image_atomic_after_a_size_query_gets_an_integer_handle() {
        // A size query, then an atomic on the same image UAV: the float
        // handle of the query must not be reused for the u32 atomic
        // (dxil-spirv then declares a float image: OpAtomicUMax %float).
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF40C_0202,
            0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], null
            0x7E02_0300, // v_mov_b32 v1, v0
            0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
            0xF038_0308,
            0x0002_0200, // image_get_resinfo v[2:3], v0, s[8:15] dmask:0x3 dim:2D
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xF05C_0108,
            0x0002_0200, // image_atomic_umax v2, v[0:1], s[8:15] dmask:0x1 dim:2D
            S_ENDPGM,
        ];
        let used = vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        }];
        let l = lower_with(&words, vec![root(RootType::Table, 0, 0)], used).unwrap();
        // The query's f32 x4 handle and the atomic's own u32 x1 one.
        assert_eq!(
            l.heap_props.iter().map(|p| p.1).collect::<Vec<_>>(),
            [0x409, 0x105]
        );
    }

    #[test]
    fn a_console_waterfall_that_only_sizes_reads_each_lanes_own_dimensions() {
        use crate::resource::tests::WATERFALL_SIZED_WAVE32;
        let (layout, used) = texture_and_uav_tables();
        let l = lower_wave(WATERFALL_SIZED_WAVE32, layout, used, false).unwrap();
        // The sizing loop's and the sampling loop's handles, both each
        // lane's own entry.
        assert_eq!(l.nonuniform_handles, 2);
        assert!(!l.wave_ops);
    }

    #[test]
    fn a_biased_sample_of_16_bit_coordinates_reads_the_bias_from_its_own_vgpr() {
        // With a16 the bias takes a whole VGPR, an f16 in its low half, and
        // the packed 16-bit coordinates start at the next one: v2 holds the
        // bias (-1.0 as f16), v3 the coordinates (0, 0).
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE86_0301, // s_mov_b32 s6, s1
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF40C_0202,
            0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], null
            0xF408_0403,
            0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], null
            0x7E04_02FF,
            0x0000_BC00, // v_mov_b32 v2, 0xbc00
            0x7E06_0280, // v_mov_b32 v3, 0
            0xF094_0108,
            0x4082_0102, // image_sample_b v1, v[2:3], s[8:15], s[16:19] dmask:0x1 dim:2D a16
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let (layout, used) = texture_tables();
        lower_with(&words, layout, used).unwrap();
    }

    #[test]
    fn a_gradient_sample_with_a_level_clamp_reads_the_clamp_after_the_coordinates() {
        // image_sample_d_cl on a 2D texture: x gradients, y gradients,
        // coordinates (two each), then the clamp (1.0) in v8.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE86_0301, // s_mov_b32 s6, s1
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF40C_0202,
            0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], null
            0xF408_0403,
            0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], null
            0x7E04_0280, // v_mov_b32 v2, 0
            0x7E06_0280, // v_mov_b32 v3, 0
            0x7E08_0280, // v_mov_b32 v4, 0
            0x7E0A_0280, // v_mov_b32 v5, 0
            0x7E0C_0280, // v_mov_b32 v6, 0
            0x7E0E_0280, // v_mov_b32 v7, 0
            0x7E10_02F2, // v_mov_b32 v8, 1.0
            0xF08C_0108,
            0x0082_0102, // image_sample_d_cl v1, v[2:8], s[8:15], s[16:19] dmask:0x1 dim:2D
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let (layout, used) = texture_tables();
        lower_with(&words, layout, used).unwrap();
    }

    #[test]
    fn a_cube_gradient_sample_takes_face_space_gradients() {
        // image_sample_d_cl on a cube: two face-space gradients per
        // direction (v2..v5), face coordinates s, t (1.5) and face 0
        // (v6..v8), then the clamp (1.0) in v9.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE86_0301, // s_mov_b32 s6, s1
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF40C_0202,
            0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], null
            0xF408_0403,
            0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], null
            0x7E04_0280, // v_mov_b32 v2, 0
            0x7E06_0280, // v_mov_b32 v3, 0
            0x7E08_0280, // v_mov_b32 v4, 0
            0x7E0A_0280, // v_mov_b32 v5, 0
            0x7E0C_02FF,
            0x3FC0_0000, // v_mov_b32 v6, 1.5
            0x7E0E_02FF,
            0x3FC0_0000, // v_mov_b32 v7, 1.5
            0x7E10_0280, // v_mov_b32 v8, 0
            0x7E12_02F2, // v_mov_b32 v9, 1.0
            0xF08C_0118,
            0x0082_0102, // image_sample_d_cl v1, v[2:9], s[8:15], s[16:19] dmask:0x1 dim:CUBE
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let (layout, used) = texture_tables();
        lower_with(&words, layout, used).unwrap();
    }

    /// [`texture_tables`] with a raw buffer UAV at dword 16 of the first.
    pub(crate) fn texture_and_uav_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
        let (layout, mut used) = texture_tables();
        used.push(UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 16,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 0,
            slot: 0,
        });
        (layout, used)
    }

    #[test]
    fn waterfall_loops_over_descriptors_joined_per_lane_sample_with_each_lanes_own() {
        use crate::resource::tests::{WATERFALL_JOINED_MIXED_WAVE32, WATERFALL_JOINED_WAVE32};
        for words in [WATERFALL_JOINED_WAVE32, WATERFALL_JOINED_MIXED_WAVE32] {
            let (layout, used) = texture_and_uav_tables();
            let l = lower_wave(words, layout, used, false).unwrap();
            // One image handle, from the join's per-lane heap index.
            assert_eq!(l.nonuniform_handles, 1);
            assert!(!l.wave_ops);
        }
    }

    #[test]
    fn descriptors_joined_by_nested_control_flow_lower_to_one_uniform_handle() {
        // The view table at s0 (2D textures at entries 0 and 1, a raw UAV at
        // dword 16) and a root constant at s1: entry 0 if s1 is 0, else
        // entry 1 if it is 1, else entry 0 again, the inner if/else joined
        // before the outer one. The join's edges are a load and a join.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0A02,
            0xFA00_0040, // s_load_dwordx4 s[40:43], s[4:5], 0x40
            0xBE88_03FF,
            0x0000_0092, // s_mov_b32 s8, 0x92 (an immediate sampler)
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE8A_0380, // s_mov_b32 s10, 0
            0xBE8B_0380, // s_mov_b32 s11, 0
            0xBF06_8001, // s_cmp_eq_u32 s1, 0
            0xBF84_0003, // s_cbranch_scc0 -> 0x38
            0xF40C_0302,
            0xFA00_0000, // s_load_dwordx8 s[12:19], s[4:5], 0x0
            0xBF82_0008, // s_branch -> 0x58
            0xBF06_8101, // 0x38: s_cmp_eq_u32 s1, 1
            0xBF84_0003, // s_cbranch_scc0 -> 0x4c
            0xF40C_0302,
            0xFA00_0020, // s_load_dwordx8 s[12:19], s[4:5], 0x20
            0xBF82_0002, // s_branch -> 0x54
            0xF40C_0302,
            0xFA00_0000, // 0x4c: s_load_dwordx8 s[12:19], s[4:5], 0x0
            0xBE86_0301, // 0x54: s_mov_b32 s6, s1
            0x7E04_02F0, // 0x58: v_mov_b32 v2, 0.5
            0x7E06_02F0, // v_mov_b32 v3, 0.5
            0x3420_0082, // v_lshlrev_b32 v16, 2, v0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            0xF09C_0F08,
            0x0043_0C02, // image_sample_lz v[12:15], v[2:3], s[12:19], s[8:11] dmask:0xf dim:SQ_RSRC_IMG_2D
            0xE070_1000,
            0x800A_0C10, // buffer_store_dword v12, v16, s[40:43], 0 offen
            S_ENDPGM,
        ];
        let (_, mut used) = texture_and_uav_tables();
        used.retain(|u| u.user_data == 0);
        let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
        let l = lower_with(&words, layout, used).unwrap();
        // One image handle from the joins' heap index, the same in all lanes.
        assert_eq!(l.nonuniform_handles, 0);
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut l.dxil.clone())
                .expect("DXIL validation failed");
        }
    }

    #[test]
    fn scalar_bit_tests_lower() {
        // Each tests one bit of a constant into SCC and branches on it (only
        // code with an effect needs its operands).
        let tail = [
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE84_0380, // s_mov_b32 s4, 0
            S_ENDPGM,
        ];
        let bit_tests: [&[u32]; 4] = [
            &[0xBE81_0387, 0xBF0C_8301], // s_mov_b32 s1, 7; s_bitcmp0_b32 s1, 3
            &[0xBE81_0387, 0xBF0D_8301], // s_mov_b32 s1, 7; s_bitcmp1_b32 s1, 3
            &[
                0xBE80_0387, // s_mov_b32 s0, 7
                0xBE81_0380, // s_mov_b32 s1, 0
                0xBF0E_8300, // s_bitcmp0_b64 s[0:1], 3
            ],
            &[
                0xBE80_0387, // s_mov_b32 s0, 7
                0xBE81_0380, // s_mov_b32 s1, 0
                0xBF0F_8300, // s_bitcmp1_b64 s[0:1], 3
            ],
        ];
        for head in bit_tests {
            let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
            lowers(&words, 0).unwrap();
        }
    }

    #[test]
    fn scalar_64_bit_field_ops_lower() {
        // Each makes a 64-bit field from constants and branches on SCC (only
        // code with an effect needs its operands).
        let tail = [
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE87_0380, // s_mov_b32 s7, 0
            S_ENDPGM,
        ];
        let consts = [
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0380, // s_mov_b32 s1, 0
            0xBE82_0388, // s_mov_b32 s2, 8
        ];
        let field_tests: [&[u32]; 3] = [
            &[0x9484_0200], // s_bfe_u64 s[4:5], s[0:1], s2
            &[0x9504_0200], // s_bfe_i64 s[4:5], s[0:1], s2
            // s_bfm_b64 s[4:5], s0, s1; its result is read by s_or_b32,
            // which sets the SCC the branch uses.
            &[0x9284_0100, 0x8806_0504],
        ];
        for body in field_tests {
            let words: Vec<u32> = consts.iter().chain(body).chain(&tail).copied().collect();
            lowers(&words, 0).unwrap();
        }
    }

    #[test]
    fn a_missing_value_stays_missing_through_two_live_joins() {
        let mut words = [
            0xF490_0180,
            0,           // s_memtime s[6:7]
            0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
            0xBF85_0001, // skip the first overwrite
            0xBE86_0380, // s_mov_b32 s6, 0
            0xBF06_8101, // s_cmp_eq_u32 s1, 1 (s1 = group id Y)
            0xBF85_0001, // skip the second overwrite
            0xBE86_0381, // s_mov_b32 s6, 1
            0xBF07_8006, // s_cmp_lg_u32 s6, 0
            0xBF85_0001, // use the result to branch
            0xBE83_0380,
            S_ENDPGM,
        ];
        let group_ids_xy = 3 << 7;
        // Compare to a direct read so future timestamp support works in both cases.
        let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
            Ok(()) => None,
            Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
            Err(error) => panic!("{error}"),
        };
        let direct = [&words[..2], &words[8..]].concat();
        assert_eq!(refusal(&words), refusal(&direct));
        words[8] = 0xBF07_8000; // read the group id instead; the timestamp is dead
        lowers(&words, group_ids_xy).expect("dead timestamp");
    }

    #[cfg(windows)]
    fn validate_dxil(dxil: &[u8]) {
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            let mut copy = dxil.to_vec();
            let ours = copy[4..20].to_vec();
            v.validate(&mut copy).expect("DXIL validation failed");
            assert_eq!(copy[4..20], ours[..], "container hash mismatch");
        }
    }

    #[cfg(not(windows))]
    fn validate_dxil(_dxil: &[u8]) {}

    #[test]
    fn a_live_descriptor_word_stays_missing_through_two_joins() {
        let mut words = [
            0xBE84_0300, // s_mov_b32 s4, s0 (table)
            0xBE85_0380, // s_mov_b32 s5, 0
            0x7E00_0280, // v_mov_b32 v0, 0
            0xBF06_8001, // s_cmp_eq_u32 s1, 0 (root constant)
            0xBF85_0002, // take the descriptor arm when SCC is set
            0xBE89_0391, // s_mov_b32 s9, 17
            0xBF82_0006, // skip the descriptor arm
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
            0xE030_2000,
            0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            0xBF85_0001, // if SCC is set, keep the descriptor word live
            0xBE89_039D, // s_mov_b32 s9, 29
            0x7E04_0209, // v_mov_b32 v2, s9
            0xD834_0000,
            0x0000_0200, // ds_write_b32 v0, v2
            S_ENDPGM,
        ];
        let run = |words: &[u32]| {
            let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
            let used = vec![UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }];
            lower_wave(words, layout, used, false)
        };
        let reason = |words: &[u32]| match run(words) {
            Ok(_) => None,
            Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
            Err(error) => panic!("{error}"),
        };
        let through_two_joins = reason(&words);
        assert_eq!(
            through_two_joins,
            Some("a descriptor's address or format read as data")
        );
        words[13] = 0xBF82_0001; // skip the overwrite unconditionally: one join
        assert_eq!(through_two_joins, reason(&words));

        // Correlated overwrite: when SCC is set, always overwrite s9 = 29 before reading it.
        words[13] = 0xBF84_0001; // s_cbranch_scc0 +1 (skip overwrite only when SCC is clear)
        let lowered = run(&words).expect("overwritten descriptor word must lower");
        validate_dxil(&lowered.dxil);
    }

    #[test]
    fn unknown_stride_load_stays_missing_when_live_and_lowers_when_overwritten() {
        let mut words = [
            0xBE84_0300, // s_mov_b32 s4, s0 (table)
            0xBE85_0380, // s_mov_b32 s5, 0
            0x7E00_0280, // v_mov_b32 v0, 0
            0xBF06_8001, // s_cmp_eq_u32 s1, 0 (root constant)
            0xBF85_0002, // take the unknown-stride load arm when SCC is set
            0x7E02_0291, // v_mov_b32 v1, 17
            0xBF82_0004, // skip the load arm
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
            0xE030_2000,
            0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen (unknown stride -> Missing)
            0xBF85_0001, // if SCC is set, keep the missing load live
            0x7E02_029D, // v_mov_b32 v1, 29
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let run = |words: &[u32]| {
            let layout = vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
            let used = vec![UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 2, // unsupported stride -> Missing without stride table
                dwords: 4,
                user_data: 0,
                slot: 0,
            }];
            lower_wave(words, layout, used, false)
        };
        let reason = |words: &[u32]| match run(words) {
            Ok(_) => None,
            Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
            Err(error) => panic!("{error}"),
        };
        let two_joins = reason(&words);
        assert_eq!(two_joins, Some("a structured buffer of unknown stride"));
        words[11] = 0xBF82_0001; // s_branch +1: one join
        assert_eq!(two_joins, reason(&words));

        // Correlated overwrite: when SCC is set, always overwrite v1 = 29.
        words[11] = 0xBF84_0001; // s_cbranch_scc0 +1
        let lowered = run(&words).expect("overwritten unknown-stride load must lower");
        validate_dxil(&lowered.dxil);
    }

    #[test]
    fn first_half_of_interpolation_stays_missing_when_live_and_lowers_when_overwritten() {
        let mut words = [
            0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x (Missing)
            0xBF06_8000,    // s_cmp_eq_u32 s0, 0
            0xBF84_0001,    // s_cbranch_scc0 +1 (skip p2 when s0 != 0)
            0xC809_0001,    // v_interp_p2_f32 v2, v1, attr0.x
            0xBF85_0001,    // s_cbranch_scc1 +1 (skip overwrite when s0 == 0)
            0x7E04_0280,    // v_mov_b32 v2, 0
            0xF800_180F,
            0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
            S_ENDPGM,
        ];
        let run = |words: &[u32]| {
            let native = NativeProgram {
                stage: HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![root(RootType::Constants, 0, 0)],
                used: vec![],
                io: StageIo::Ps {
                    inputs: vec![crate::native::PsInput {
                        usage: crate::native::Usage::Generic,
                        row: 0,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: false,
                        pack2: false,
                        row2: 0,
                    }],
                    rsrc2: 0,
                    input_addr: 1 << 1, // PERSP_CENTER
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let p = ir::build(&native.code, false, 4).unwrap();
            let st = structure::structure(&p).unwrap();
            let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = io::raise(&p, &native).unwrap();
            let outputs = vec![super::SigElement {
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
            }];
            let inputs = vec![super::SigElement {
                name: "TEXCOORD".into(),
                index: 0,
                kind: 0,
                comp: 9,
                interp: 2,
                row: Some(0),
                col: 0,
                cols: 4,
                usage: 0,
                half: false,
            }];
            lower(&Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs,
                outputs,
                prims: vec![],
                mesh_stride: 0,
            })
        };
        // Overwritten on the s0 != 0 branch: p1 never reaches exp.
        let lowered = run(&words).expect("overwritten interp half must lower");
        validate_dxil(&lowered.dxil);

        // Flip word 4 to s_cbranch_scc0 +1: when s0 != 0, p1 reaches exp live.
        words[4] = 0xBF84_0001;
        let Err(crate::ShaderError::Unsupported { what, .. }) = run(&words) else {
            panic!("live p1 must refuse");
        };
        assert_eq!(what, "the first half of an interpolation");

        // Unconditional skip (1 join): also refuses identically.
        words[4] = 0xBF82_0001;
        let Err(crate::ShaderError::Unsupported { what, .. }) = run(&words) else {
            panic!("1-join live p1 must refuse");
        };
        assert_eq!(what, "the first half of an interpolation");
    }

    #[test]
    fn guarded_write_and_exec_mask_phi_lower_soundly() {
        // 1. EXEC phi fed by a scalar SGPR in one branch, then read via s_bcnt1_i32_b32.
        let exec_phi_words = [
            0xBE84_03BF, // s_mov_b32 s4, 0x3F
            0xBF06_8000, // s_cmp_eq_u32 s0, 0
            0xBF84_0001, // s_cbranch_scc0 +1
            0xBEFE_0304, // s_mov_b32 exec_lo, s4
            0xBE82_0F7E, // s_bcnt1_i32_b32 s2, exec_lo
            0x7E00_0202, // v_mov_b32 v0, s2
            0xD834_0000,
            0x0000_0000, // ds_write_b32 v0, v0
            S_ENDPGM,
        ];
        let lowered = lower_wave(
            &exec_phi_words,
            vec![root(RootType::Constants, 0, 0)],
            vec![],
            false,
        )
        .expect("EXEC phi read via mask_ok must lower");
        validate_dxil(&lowered.dxil);

        // 2. Guarded write (`write_placed`) preserving `17` on `!guard` lanes where `guard`
        //    lanes held an unknown-stride load (and overwriting `guard` lanes with `29`).
        let guarded_words = [
            0xBE84_0300, // s_mov_b32 s4, s0 (table)
            0xBE85_0380, // s_mov_b32 s5, 0
            0x7E00_0280, // v_mov_b32 v0, 0
            0x7E02_0291, // v_mov_b32 v1, 17
            0x7D84_0080, // v_cmp_eq_u32 vcc_lo, 0, v0
            0xBE86_3C6A, // s_and_saveexec_b32 s6, vcc_lo
            0xBF88_0004, // s_cbranch_execz +4
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
            0xE030_2000,
            0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen (unknown stride)
            0xBEFE_0306, // s_mov_b32 exec_lo, s6
            0xBE86_3C6A, // s_and_saveexec_b32 s6, vcc_lo
            0x7E02_029D, // v_mov_b32 v1, 29 (guarded by v0 == 0, keeps v1 = 17 on v0 != 0)
            0xBEFE_0306, // s_mov_b32 exec_lo, s6
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let lowered = lower_wave(
            &guarded_words,
            vec![root(RootType::Table, 0, 0)],
            vec![UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }],
            false,
        )
        .expect("guarded overwrite over unknown-stride arm must lower");
        validate_dxil(&lowered.dxil);
    }

    #[test]
    fn loop_missing_value_refuses_on_back_edge_and_lowers_when_infeasible() {
        // Loop header (word 4): s2 starts at 17, v7 = 0. Inside loop: if s0 == 0, s_memtime -> s2.
        // Latch uses VCC so loop-invariant scc (s0 == 0) survives across iterations.
        // After loop: if s0 == 0, overwrite s2 = 29, else read P_s2.
        let mut words = [
            0x7E00_0280, // 0: v_mov_b32 v0, 0
            0xBE82_0391, // 1: s_mov_b32 s2, 17
            0x7E0E_0280, // 2: v_mov_b32 v7, 0
            0xBF06_8000, // 3: s_cmp_eq_u32 s0, 0 (loop-invariant, preserved in SCC)
            // loop header (word 4):
            0xBF84_0003, // 4: s_cbranch_scc0 +3 (skip s_memtime when s0 != 0)
            0xF490_0180,
            0,           // 5-6: s_memtime s[6:7]
            0xBE82_0306, // 7: s_mov_b32 s2, s6
            // latch (word 8):
            0x4A0E_0E81, // 8: v_add_nc_u32 v7, 1, v7
            0x7D88_0E82, // 9: v_cmp_gt_u32 vcc_lo, 2, v7
            0xBF87_FFF9, // 10: s_cbranch_vccnz -7 (back to word 4)
            // after loop (word 11):
            0xBF84_0001, // 11: s_cbranch_scc0 +1 (skip overwrite when s0 != 0)
            0xBE82_039D, // 12: s_mov_b32 s2, 29
            0x7E02_0202, // 13: v_mov_b32 v1, s2
            0xD834_0000,
            0x0000_0100, // 14-15: ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let run =
            |words: &[u32]| lower_wave(words, vec![root(RootType::Constants, 0, 0)], vec![], false);
        let lowered = run(&words).expect("loop-invariant infeasible missing edge must lower");
        validate_dxil(&lowered.dxil);

        // Flip word 11 to s_cbranch_scc1 +1: s_memtime from back-edge reaches ds_write when s0 == 0.
        words[11] = 0xBF85_0001;
        let Err(crate::ShaderError::Unsupported { what, .. }) = run(&words) else {
            panic!("live loop back-edge missing value must refuse");
        };
        assert_eq!(what, "a scalar memory read as data");
    }

    #[test]
    fn old_register_missing_values_and_cached_floats_stay_refused() {
        let srv_used = || {
            vec![UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 2,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }]
        };
        let expect_unsupported = |res: Result<Lowered, crate::ShaderError>, want: &str| match res {
            Ok(_) => panic!("expected refusal `{want}`, but lowered"),
            Err(crate::ShaderError::Unsupported { what, .. }) => assert_eq!(what, want),
            Err(error) => panic!("{error}"),
        };

        // 1a. Direct Val::Missing on `old` surviving into `!guard` of `write_placed`.
        let direct_old_guard = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0x7E00_0280, // v_mov_b32 v0, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
            0xE030_2000,
            0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen (unknown stride)
            0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0 (v0 < 16)
            0xBE86_3C6A, // s_and_saveexec_b32 s6, vcc_lo
            0x7E02_029D, // v_mov_b32 v1, 29 (v0 >= 16 keeps old unknown-stride v1)
            0xBEFE_0306, // s_mov_b32 exec_lo, s6
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        expect_unsupported(
            lower_wave(
                &direct_old_guard,
                vec![root(RootType::Table, 0, 0)],
                srv_used(),
                false,
            ),
            "a structured buffer of unknown stride",
        );

        // 1b. Phi Val::Missing on `old` surviving into `!guard` of `write_placed`.
        let phi_old_guard = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0x7E00_0280, // v_mov_b32 v0, 0
            0x7E02_0291, // v_mov_b32 v1, 17
            0x7D84_0080, // v_cmp_eq_u32 vcc_lo, 0, v0 (v0 == 0)
            0xBE86_3C6A, // s_and_saveexec_b32 s6, vcc_lo
            0xBF88_0004, // s_cbranch_execz +4
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
            0xE030_2000,
            0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
            0xBEFE_0306, // s_mov_b32 exec_lo, s6
            0x7D8A_0080, // v_cmp_ne_u32 vcc_lo, 0, v0 (v0 != 0 -> !guard keeps v0 == 0!)
            0xBE86_3C6A, // s_and_saveexec_b32 s6, vcc_lo
            0x7E02_029D, // v_mov_b32 v1, 29
            0xBEFE_0306, // s_mov_b32 exec_lo, s6
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        expect_unsupported(
            lower_wave(
                &phi_old_guard,
                vec![root(RootType::Table, 0, 0)],
                srv_used(),
                false,
            ),
            "a structured buffer of unknown stride",
        );

        // 1c. Live Val::Missing on `old` kept by `write_16` (`keep == true`).
        let write16_old = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0x7E00_0280, // v_mov_b32 v0, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
            0xE030_2000,
            0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
            0x6402_00F2, // v_add_f16 v1, 1.0, v0 (keeps high 16 bits of v1)
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        expect_unsupported(
            lower_wave(
                &write16_old,
                vec![root(RootType::Table, 0, 0)],
                srv_used(),
                false,
            ),
            "a structured buffer of unknown stride",
        );

        // 1d. Live Val::Missing on `old` kept by `place_bits` (SDWA `dst_unused: UNUSED_PRESERVE`).
        let sdwa_old = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0x7E00_0280, // v_mov_b32 v0, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
            0xE030_2000,
            0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
            0x7E02_02F9,
            0x0006_1400, // v_mov_b32_sdwa v1, v0 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        expect_unsupported(
            lower_wave(
                &sdwa_old,
                vec![root(RootType::Table, 0, 0)],
                srv_used(),
                false,
            ),
            "a structured buffer of unknown stride",
        );

        // 1e. Direct Val::Missing on `outside` surviving `!guard` of `ds_add_rtn_u32`.
        let ds_rtn_outside = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0x7E00_0280, // v_mov_b32 v0, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0
            0xE030_2000,
            0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 idxen
            0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0 (v0 < 16)
            0xBE86_3C6A, // s_and_saveexec_b32 s6, vcc_lo
            0xD880_0000,
            0x0100_0000, // ds_add_rtn_u32 v1, v0, v0 (v0 >= 16 keeps outside = unknown-stride v1)
            0xBEFE_0306, // s_mov_b32 exec_lo, s6
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        expect_unsupported(
            lower_wave(
                &ds_rtn_outside,
                vec![root(RootType::Table, 0, 0)],
                srv_used(),
                false,
            ),
            "a structured buffer of unknown stride",
        );

        // 2. Live missing phi flowing through chained float ops (`self.floats` cache hit) to `ds_write`.
        let chained_float_phi = [
            0x7E00_0280, // v_mov_b32 v0, 0
            0xBF06_8000, // s_cmp_eq_u32 s0, 0
            0xBF85_0002, // take s_memtime arm when s0 == 0
            0xBE82_0391, // s_mov_b32 s2, 17
            0xBF82_0003, // skip s_memtime
            0xF490_0180,
            0,           // s_memtime s[6:7]
            0xBE82_0306, // s_mov_b32 s2, s6
            0x0604_0002, // v_add_f32 v2, s2, v0 (inserts v2 into self.floats)
            0x0A06_04F4, // v_mul_f32 v3, 2.0, v2 (hits self.floats[&v2], inserts v3 into self.floats)
            0xD834_0000,
            0x0000_0300, // ds_write_b32 v0, v3
            S_ENDPGM,
        ];
        expect_unsupported(
            lower_wave(
                &chained_float_phi,
                vec![root(RootType::Constants, 0, 0)],
                vec![],
                false,
            ),
            "a scalar memory read as data",
        );
    }

    #[test]
    fn v_readlane_and_writelane_mask_lane_and_ignore_exec() {
        // v_writelane_b32 under narrowed EXEC ignores EXEC and writes target lane.
        // v_readlane_b32 with lane > 31 masks to lane & 31.
        let words = [
            0xBE80_0380, // s_mov_b32 s0, 0
            0xBEFE_0380, // s_mov_b32 exec_lo, 0
            0xD761_0000,
            0x0001_0000, // v_writelane_b32 v0, s0, 0 (VOP3 0x361)
            0xD760_0000,
            0x0001_5100, // v_readlane_b32 s0, v0, 40 (lane 40 inline const 168) (VOP3 0x360)
            S_ENDPGM,
        ];
        let layout = || vec![];
        let used = || vec![];
        let l = lower_with(&words, layout(), used()).unwrap();
        #[cfg(windows)]
        if let Ok(mut v) = crate::emit::validator::Validator::load(
            crate::emit::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut l.dxil.clone()).unwrap();
        }
    }

    #[test]
    fn buffer_formats_by_number() {
        // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM,
        // 2_10_10_10_UNORM, 16_FLOAT, 32_32_32_32_FLOAT.
        assert_eq!(buffer_format(56), Some((&[8, 8, 8, 8][..], Numeric::Unorm)));
        assert_eq!(buffer_format(24), Some((&[16, 16][..], Numeric::Snorm)));
        assert_eq!(
            buffer_format(50),
            Some((&[10, 10, 10, 2][..], Numeric::Unorm))
        );
        assert_eq!(buffer_format(13), Some((&[16][..], Numeric::Float)));
        assert_eq!(
            buffer_format(77),
            Some((&[32, 32, 32, 32][..], Numeric::Float))
        );
        // Invalid, past the table.
        assert_eq!(buffer_format(0), None);
        assert_eq!(buffer_format(78), None);
    }

    #[test]
    fn packed_buffer_formats_by_number() {
        // LLVM's `UfmtGFX10`: 10_11_11 is 30-36, 11_11_10 37-43,
        // 10_10_10_2 44-49. Channels from the lowest bits: R11G11B10_FLOAT
        // is 10_11_11 (Mesa), 10_10_10_2 has the 2-bit channel first.
        assert_eq!(buffer_format(30), Some((&[11, 11, 10][..], Numeric::Unorm)));
        assert_eq!(buffer_format(36), Some((&[11, 11, 10][..], Numeric::Float)));
        assert_eq!(buffer_format(37), Some((&[10, 11, 11][..], Numeric::Unorm)));
        assert_eq!(buffer_format(43), Some((&[10, 11, 11][..], Numeric::Float)));
        assert_eq!(
            buffer_format(44),
            Some((&[2, 10, 10, 10][..], Numeric::Unorm))
        );
        assert_eq!(
            buffer_format(45),
            Some((&[2, 10, 10, 10][..], Numeric::Snorm))
        );
        assert_eq!(
            buffer_format(46),
            Some((&[2, 10, 10, 10][..], Numeric::Uscaled))
        );
        assert_eq!(
            buffer_format(47),
            Some((&[2, 10, 10, 10][..], Numeric::Sscaled))
        );
        assert_eq!(
            buffer_format(48),
            Some((&[2, 10, 10, 10][..], Numeric::Uint))
        );
        assert_eq!(
            buffer_format(49),
            Some((&[2, 10, 10, 10][..], Numeric::Sint))
        );
    }

    /// A half's value (sign, 5-bit exponent biased 15, 10-bit mantissa), as
    /// `legacyF16ToF32` widens it.
    fn half(h: u32) -> f32 {
        let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 1023);
        let v = match e {
            0 => m as f32 * 2f32.powi(-24),
            31 if m == 0 => f32::INFINITY,
            31 => f32::NAN,
            _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
        };
        if s == 1 { -v } else { v }
    }

    #[test]
    fn small_floats_widen_as_halves() {
        // Unsigned 11- and 10-bit floats: a 5-bit exponent biased 15 over a
        // 6- or 5-bit mantissa, no sign; values from the D3D/GL spec.
        let f11 = |x: u32| half(x << half_shift(11));
        let f10 = |x: u32| half(x << half_shift(10));
        assert_eq!(f11(0), 0.0);
        assert_eq!(f11(0x001), 2f32.powi(-20)); // smallest denormal
        assert_eq!(f11(0x03F), 63.0 * 2f32.powi(-20)); // largest denormal
        assert_eq!(f11(0x040), 2f32.powi(-14)); // smallest normal
        assert_eq!(f11(0x3C0), 1.0);
        assert_eq!(f11(0x3E0), 1.5);
        assert_eq!(f11(0x7BF), 65024.0); // largest
        assert_eq!(f11(0x7C0), f32::INFINITY);
        assert!(f11(0x7C1).is_nan());
        assert_eq!(f10(0x001), 2f32.powi(-19));
        assert_eq!(f10(0x020), 2f32.powi(-14));
        assert_eq!(f10(0x1E0), 1.0);
        assert_eq!(f10(0x3DF), 64512.0);
        assert_eq!(f10(0x3E0), f32::INFINITY);
        assert!(f10(0x3FF).is_nan());
        // A 16-bit float is a half already, sign included.
        assert_eq!(half(0xBC00 << half_shift(16)), -1.0);
    }

    #[test]
    fn image_sample_d_cl_and_gather4_l_translate() {
        use crate::native::{RootParam, RootType};
        let param = |root: RootType, user_data: u8, sgprs: u32| RootParam {
            root,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data,
            slot: 0,
        };
        let layout = [param(RootType::Table, 8, 1), param(RootType::Table, 9, 1)];

        // 1. image_sample_d_cl v[0:3], v[2:8], s[0:7], s[8:11]
        let sample_d_cl_words = [
            0xBE82_0309, // s_mov_b32 s2, s9
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE83_0380, // s_mov_b32 s3, 0
            0xF408_0204,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[8:9], null
            0xF40C_0001,
            0xFA00_0000, // s_load_dwordx8 s[0:7], s[2:3], null
            0xF08C_0F08,
            0x0040_0002, // image_sample_d_cl v[0:3], v[2:8], s[0:7], s[8:11]
            S_ENDPGM,
        ];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: layout.to_vec(),
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [1, 1, 1],
            }),
            code: sample_d_cl_words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect(),
        };
        let p = ir::build(&native.code, false, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
        .unwrap();

        // 2. image_gather4_l v[0:3], v[2:4], s[0:7], s[8:11]
        let gather4_l_words = [
            0xBE82_0309, // s_mov_b32 s2, s9
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE83_0380, // s_mov_b32 s3, 0
            0xF408_0204,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[8:9], null
            0xF40C_0001,
            0xFA00_0000, // s_load_dwordx8 s[0:7], s[2:3], null
            0xF110_0108,
            0x0040_0002, // image_gather4_l v[0:3], v[2:4], s[0:7], s[8:11]
            S_ENDPGM,
        ];
        let native_cs = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: layout.to_vec(),
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [1, 1, 1],
            }),
            code: gather4_l_words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect(),
        };
        let p = ir::build(&native_cs.code, false, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native_cs.layout, native_cs.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native_cs).unwrap();
        lower(&Parts {
            native: &native_cs,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
        .unwrap();
    }

    #[test]
    fn spilled_inactive_mask_does_not_poison_other_lanes() {
        let words = [
            0xBE80_0385, // s_mov_b32 s0, 5
            0x8902_C17E, // s_xor_b32 s2, exec_lo, -1
            0xD761_0000,
            0x0001_0002, // v_writelane_b32 v0, s2, 0
            0xD761_0000,
            0x0001_0200, // v_writelane_b32 v0, s0, 1
            0xD760_0004,
            0x0001_0300, // v_readlane_b32 s4, v0, 1
            0xBF06_8504, // s_cmp_eq_u32 s4, 5
            0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
            0xBE85_0380, // s_mov_b32 s5, 0
            S_ENDPGM,
        ];
        lowers(&words, 0).unwrap();
    }

    #[test]
    fn descriptor_data_handles_s_buffer_load_srv() {
        use crate::native::{ResourceType, RootParam, RootType, UsedResource};
        let words = [
            0xBE82_0309, // s_mov_b32 s2, s9
            0xBE83_0380, // s_mov_b32 s3, 0
            0xF408_0001,
            0xFA00_0000, // s_load_dwordx4 s[0:3], s[2:3], null
            0xBF06_8002, // s_cmp_eq_u32 s2, 0
            0xBF85_0002, // s_cbranch_scc1 -> s_endpgm
            0xF420_0100,
            0xFA00_0000, // s_buffer_load_dword s4, s[0:3], null
            S_ENDPGM,
        ];
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: false,
            vgprs: 256,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 9,
                slot: 0,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 9,
                slot: 0,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [1, 1, 1],
            }),
            code,
        };
        let p = ir::build(&native.code, false, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![],
            prims: vec![],
            mesh_stride: 0,
        })
        .unwrap();
    }

    #[test]
    fn table_views_record_srv_uav_structured_and_skip_dynamic() {
        let layout = vec![
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
        ];
        let used = vec![
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
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
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 16,
                lower_bound: 1,
                stride: 32,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
        ];
        let mut words = vec![s_mov_b32(5, 0x80)];
        words.extend_from_slice(&s_load_dwordx4(8, 4, 0, 125));
        words.extend_from_slice(&s_load_dwordx4(12, 4, 32, 125));
        words.extend_from_slice(&s_load_dwordx4(16, 4, 64, 125));
        words.push(0x7E28_0500); // v_readfirstlane_b32 s20, v0: an offset from no root parameter
        words.extend_from_slice(&s_load_dwordx4(20, 4, 96, 20)); // dynamic soffset s20
        words.extend_from_slice(&buffer_load_dword(0, 8, false));
        words.extend_from_slice(&buffer_store_dword(0, 12, false));
        words.extend_from_slice(&buffer_store_dword(0, 16, true));
        words.extend_from_slice(&buffer_store_dword(0, 20, false));
        words.push(S_ENDPGM);

        assert_eq!(
            lower_table_views(&words, layout, used, 5),
            vec![
                TableView {
                    slot: 1,
                    dword: 0,
                    props: (11, 0),
                },
                TableView {
                    slot: 1,
                    dword: 8,
                    props: (11 | 0x1000, 0),
                },
                TableView {
                    slot: 1,
                    dword: 16,
                    props: (12 | 0x1000, 32),
                },
            ]
        );
    }

    #[test]
    fn table_views_prefer_uav_and_record_phi_arms() {
        let layout = vec![
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
        ];
        // Same table slot read as SRV and written as UAV: UAV wins in either order.
        for load_first in [true, false] {
            let mut words = vec![s_mov_b32(5, 0x80)];
            words.extend_from_slice(&s_load_dwordx4(8, 4, 32, 125));
            if load_first {
                words.extend_from_slice(&buffer_load_dword(0, 8, false));
                words.extend_from_slice(&buffer_store_dword(0, 8, false));
            } else {
                words.extend_from_slice(&buffer_store_dword(0, 8, false));
                words.extend_from_slice(&buffer_load_dword(0, 8, false));
            }
            words.push(S_ENDPGM);
            assert_eq!(
                lower_table_views(&words, layout.clone(), vec![], 5),
                vec![TableView {
                    slot: 1,
                    dword: 8,
                    props: (11 | 0x1000, 0),
                }]
            );
        }

        // Control-flow join (`Source::Phi`) records both incoming table slots.
        let mut phi_words = vec![
            s_mov_b32(5, 0x80),
            0xBF07_8000, // s_cmp_lg_u32 s0, 0
            0xBF85_0003, // s_cbranch_scc1 +3 dwords -> arm 2
        ];
        phi_words.extend_from_slice(&s_load_dwordx4(8, 4, 32, 125)); // arm 1 (2 dwords)
        phi_words.push(0xBF82_0002); // s_branch +2 dwords -> join (1 dword)
        phi_words.extend_from_slice(&s_load_dwordx4(8, 4, 64, 125)); // arm 2 (2 dwords)
        phi_words.extend_from_slice(&buffer_store_dword(0, 8, false));
        phi_words.push(S_ENDPGM);
        assert_eq!(
            lower_table_views(&phi_words, layout, vec![], 5),
            vec![
                TableView {
                    slot: 1,
                    dword: 8,
                    props: (11 | 0x1000, 0),
                },
                TableView {
                    slot: 1,
                    dword: 16,
                    props: (11 | 0x1000, 0),
                },
            ]
        );
    }

    #[test]
    fn v_cvt_f32_f64_lowers_with_double_flag() {
        let words = [
            0x7E00_1F00, // v_cvt_f32_f64 v0, v[0:1]
            0xD834_0000, // ds_write_b32 v0, v0 keeps the conversion live
            0,
            S_ENDPGM,
        ];
        let emitted = lower_with(&words, vec![], vec![]).unwrap();
        let container = crate::Container::parse(&emitted.dxil).unwrap();
        let features = container.part(crate::FourCC::SFI0).unwrap();
        let features = u64::from_le_bytes(features.data.try_into().unwrap());
        assert_eq!(features & 1, 1, "SFI0 double-precision feature");
        assert_eq!(super::flags::DOUBLE, 1 << 2, "DXIL double-precision flag");
    }

    #[test]
    fn cube_array_encodings_and_linked_resinfo_lower() {
        // Common prologue: SRV at s[8:15] (table 0 offset 0), sampler at
        // s[16:19] (table 1 offset 0), raw UAV at s[20:23] (table 0 offset 64).
        let head = [
            0xBE84_0300u32, // s_mov_b32 s4, s0
            0xBE85_0380,    // s_mov_b32 s5, 0
            0xBE86_0301,    // s_mov_b32 s6, s1
            0xBE87_0380,    // s_mov_b32 s7, 0
            0xF40C_0202,
            0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0
            0xF408_0403,
            0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], 0
            0xF408_0502,
            0xFA00_0040, // s_load_dwordx4 s[20:23], s[4:5], 64
            0x7E02_02FF,
            0x3FC0_0000, // v_mov_b32 v1, 1.5
            0x7E04_02FF,
            0x3FC0_0000, // v_mov_b32 v2, 1.5
        ];
        let store_v4 = [
            0xE070_0000u32,
            0x8005_0400, // buffer_store_dword v4, off, s[20:23], 0
            S_ENDPGM,
        ];
        let cases: &[&[u32]] = &[
            // Constant face >= 8.0 (layer 1, face 2 = 10.0).
            &[
                0x7E06_02FF,
                0x4120_0000, // v_mov_b32 v3, 10.0
                0xF09C_0118,
                0x0082_0401, // image_sample_lz v4, v[1:3], s[8:15], s[16:19] dmask:1 dim:CUBE
            ],
            // SGPR 8.0 multiplier in v_fma_f32, followed by image_get_resinfo
            // on a second load of the same table slot.
            &[
                0xBE98_03FF,
                0x4100_0000, // s_mov_b32 s24, 8.0
                0x7E06_0282, // v_mov_b32 v3, 2
                0xD54B_0003,
                0x04CE_3000, // v_fma_f32 v3, v0, s24, v3
                0xF09C_0118,
                0x0082_0401, // image_sample_lz v4, v[1:3], s[8:15], s[16:19] dmask:1 dim:CUBE
                0xF40C_0702,
                0xFA00_0000, // s_load_dwordx8 s[28:35], s[4:5], 0
                0x7E0A_0280, // v_mov_b32 v5, 0
                0xF038_0418,
                0x0007_0605, // image_get_resinfo v6, v5, s[28:35] dmask:4 dim:CUBE
                0x1008_0C04, // v_add_f32 v4, v4, v6
            ],
            // v_fmamk_f32 by 8.0, then image_sample_l with explicit LOD in v4.
            &[
                0x7E06_02F2, // v_mov_b32 v3, 1.0
                0x5806_0700,
                0x4100_0000, // v_fmamk_f32 v3, v0, 8.0, v3
                0x7E08_0280, // v_mov_b32 v4, 0
                0xF090_0118,
                0x0082_0401, // image_sample_l v4, v[1:4], s[8:15], s[16:19] dmask:1 dim:CUBE
            ],
            // v_fmac_f32 by 8.0 with comparison reference before coordinates
            // (image_sample_c_lz).
            &[
                0x7E08_02F2, // v_mov_b32 v4, 1.0
                0x5608_00FF,
                0x4100_0000, // v_fmac_f32 v4, 8.0, v0
                0x7E06_02FF,
                0x3FC0_0000, // v_mov_b32 v3, 1.5 (ref=v1, s=v2, t=v3, face=v4)
                0xF0BC_0118,
                0x0082_0401, // image_sample_c_lz v4, v[1:4], s[8:15], s[16:19] dmask:1 dim:CUBE
            ],
            // v_mul_f32 with omod *2 (4.0 * 2 = 8.0) + v_add_f32.
            &[
                0xD508_0003,
                0x0801_EC00, // v_mul_f32 v3, v0, 4.0 mul:2
                0x0606_06F4, // v_add_f32 v3, 2.0, v3
                0xF09C_0118,
                0x0082_0401, // image_sample_lz v4, v[1:3], s[8:15], s[16:19] dmask:1 dim:CUBE
            ],
            // a16 coordinates: v_cvt_pkrtz_f16_f32 + v_cvt_f16_f32 of 8.0 * slice + face.
            &[
                0x5E02_0501, // v_cvt_pkrtz_f16_f32 v1, v1, v2
                0x7E06_02F4, // v_mov_b32 v3, 2.0
                0x5606_00FF,
                0x4100_0000, // v_fmac_f32 v3, 8.0, v0
                0x7E04_1503, // v_cvt_f16_f32 v2, v3
                0xF09C_0118,
                0x4082_0401, // image_sample_lz v4, v[1:2], s[8:15], s[16:19] dmask:1 dim:CUBE a16
            ],
        ];
        for body in cases {
            let words: Vec<u32> = head.iter().chain(*body).chain(&store_v4).copied().collect();
            let (layout, used) = texture_and_uav_tables();
            let l = lower_with(&words, layout, used).unwrap();
            #[cfg(windows)]
            if let Ok(mut v) = super::super::validator::Validator::load(
                super::super::validator::Validator::default_path().as_deref(),
            ) {
                let mut dxil = l.dxil.clone();
                v.validate(&mut dxil).expect("DXIL validation failed");
            }
            let _ = l;
        }

        // Pixel-stage `image_sample_b` (implicit LOD + bias before coordinates)
        // with `v_madak_f32` by 8.0.
        let ps_words = [
            0xBE84_0300u32, // s_mov_b32 s4, s0
            0xBE85_0380,    // s_mov_b32 s5, 0
            0xBE86_0301,    // s_mov_b32 s6, s1
            0xBE87_0380,    // s_mov_b32 s7, 0
            0xF40C_0202,
            0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0
            0xF408_0403,
            0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], 0
            0xBE98_03FF,
            0x4100_0000, // s_mov_b32 s24, 8.0
            0x7E00_02F2, // v_mov_b32 v0, 1.0 (slice)
            0x7E02_0280, // v_mov_b32 v1, 0.0 (bias)
            0x7E04_02FF,
            0x3FC0_0000, // v_mov_b32 v2, 1.5 (s)
            0x7E06_02FF,
            0x3FC0_0000, // v_mov_b32 v3, 1.5 (t)
            0x4208_0018,
            0x4000_0000, // v_madak_f32 v4, s24, v0, 2.0
            0xF094_0118,
            0x0082_0401, // image_sample_b v4, v[1:4], s[8:15], s[16:19] dmask:1 dim:CUBE
            0xF800_1801,
            0x0000_0004, // exp mrt0 v4, off, off, off done vm
            S_ENDPGM,
        ];
        let (layout, used) = texture_tables();
        let ps_native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 16,
            layout,
            used,
            io: StageIo::Ps {
                inputs: vec![],
                rsrc2: 0,
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: ps_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&ps_native.code, false, 16).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &ps_native.layout, HwStage::Ps, Some(&st)).unwrap();
        let io = io::raise(&p, &ps_native).unwrap();
        let ps_l = lower(&Parts {
            native: &ps_native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![],
            outputs: vec![super::SigElement {
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
            prims: vec![],
            mesh_stride: 0,
        })
        .unwrap();
        #[cfg(windows)]
        if let Ok(mut v) = super::super::validator::Validator::load(
            super::super::validator::Validator::default_path().as_deref(),
        ) {
            let mut dxil = ps_l.dxil.clone();
            v.validate(&mut dxil).expect("PS DXIL validation failed");
        }
        let _ = ps_l;
    }

    #[test]
    fn heap_uav_inference_and_glc_coherence_set_handle_properties() {
        // #724: one heap descriptor loaded and stored gets UAV handles for both,
        // including across a separate reload of the same heap address and
        // scalar `s_buffer_load_dword` (with and without `glc`), while distinct
        // read-only heap addresses remain SRV and CBV.
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0 (resource heap base)
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0   (UAV @ 0)
            0xF408_0302,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20 (SRV @ 1)
            0xF408_0402,
            0xFA00_0040, // s_load_dwordx4 s[16:19], s[4:5], 0x40 (CBV @ 2)
            0x7E00_0280, // v_mov_b32 v0, 0
            0xE030_1000,
            0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 offen
            0xE030_5004,
            0x8002_0200, // buffer_load_dword v2, v0, s[8:11], 0 offen offset:4 glc
            0xE030_1000,
            0x8003_0300, // buffer_load_dword v3, v0, s[12:15], 0 offen
            0xF420_0188,
            0xFA00_0000, // s_buffer_load_dword s6, s[16:19], 0x0
            // Reload UAV @ 0 into s[20:23] and read it with scalar `glc` and plain load.
            0xF408_0502,
            0xFA00_0000, // s_load_dwordx4 s[20:23], s[4:5], 0x0
            0xF421_01CA,
            0xFA00_0008, // s_buffer_load_dword s7, s[20:23], 0x8 glc
            0xF420_060A,
            0xFA00_000C, // s_buffer_load_dword s24, s[20:23], 0xC
            0x4A02_0501, // v_add_nc_u32 v1, v1, v2
            0x4A02_0701, // v_add_nc_u32 v1, v1, v3
            0x4A02_0206, // v_add_nc_u32 v1, s6, v1
            0x4A02_0207, // v_add_nc_u32 v1, s7, v1
            0x4A02_0218, // v_add_nc_u32 v1, s24, v1
            0xE070_1010,
            0x8002_0100, // buffer_store_dword v1, v0, s[8:11], 0 offen offset:16
            S_ENDPGM,
        ];
        let l = lower_with(&words, vec![root(RootType::ResourceHeap, 0, 0)], vec![]).unwrap();
        validate_dxil(&l.dxil);
        assert_eq!(
            l.heap_props,
            vec![
                (0x100B, 0),    // UAV @ 0 handle (descriptor_data + load + store)
                (0x000B, 0),    // read-only SRV @ 1 handle (descriptor_data + load)
                (0x500B, 0),    // buffer_load_dword glc on UAV @ 0
                (0x000B, 0),    // s_buffer_load_dword on ResourceHeap reads raw SRV (#1409)
                (0x100B, 0),    // descriptor_data + plain s_buffer_load_dword on reloaded UAV @ 0
                (0x500B, 0),    // s_buffer_load_dword glc on reloaded UAV @ 0
            ]
        );
    }

    #[test]
    fn glc_only_heap_spin_wait_and_table_uavs_are_globally_coherent() {
        // #725: a heap descriptor read only by a `glc` load in a spin loop is
        // inferred as a UAV in `descriptor_data` (0x100B) and globally coherent
        // on the `glc` load itself (0x500B).
        let spin = [
            0xBE84_0300, // s_mov_b32 s4, s0 (resource heap base)
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
            0x7E00_0280, // v_mov_b32 v0, 0
            // loop:
            0xE030_5000,
            0x8002_0100, // buffer_load_dword v1, v0, s[8:11], 0 offen glc
            0x7D84_0280, // v_cmp_eq_u32 vcc_lo, 0, v1
            0xBF87_FFFC, // s_cbranch_vccnz -> loop
            0xBF8A_0000, // s_barrier
            0xD834_0000,
            0x0000_0100, // ds_write_b32 v0, v1
            S_ENDPGM,
        ];
        let l = lower_with(&spin, vec![root(RootType::ResourceHeap, 0, 0)], vec![]).unwrap();
        validate_dxil(&l.dxil);
        assert_eq!(l.heap_props, vec![(0x100B, 0), (0x500B, 0)]);

        // Table resources: structured buffer UAV (`idxen glc`), typed buffer
        // UAV (`glc`), 2D image UAV (`image_load glc`, `image_store glc`,
        // `image_atomic_add glc`), and a read-only table SRV whose `glc` load
        // must stay an SRV (0xB).
        let table = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0    (structured UAV)
            0xF408_0302,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20  (typed UAV)
            0xF40C_0402,
            0xFA00_0040, // s_load_dwordx8 s[16:23], s[4:5], 0x40  (image UAV)
            0xF40C_0602,
            0xFA00_0060, // s_load_dwordx8 s[24:31], s[4:5], 0x60  (uint image UAV)
            0xF408_0802,
            0xFA00_0080, // s_load_dwordx4 s[32:35], s[4:5], 0x80  (raw SRV)
            0x7E00_0280, // v_mov_b32 v0, 0
            0x7E02_0280, // v_mov_b32 v1, 0
            0xE030_6000,
            0x8002_0200, // buffer_load_dword v2, v0, s[8:11], 0 idxen glc
            0xE00C_4000,
            0x8003_0400, // buffer_load_format_xyzw v[4:7], v0, s[12:15], 0 glc
            0xE01C_4000,
            0x8003_0400, // buffer_store_format_xyzw v[4:7], v0, s[12:15], 0 glc
            0xF000_2F08,
            0x0004_0800, // image_load v[8:11], v[0:1], s[16:23] dmask:0xf 2d glc
            0xF020_2F08,
            0x0004_0800, // image_store v[8:11], v[0:1], s[16:23] dmask:0xf 2d glc
            0xF044_2108,
            0x0006_0200, // image_atomic_add v2, v[0:1], s[24:31] dmask:0x1 2d glc
            0xE030_5000,
            0x8008_0300, // buffer_load_dword v3, v0, s[32:35], 0 offen glc
            0x4A04_0702, // v_add_nc_u32 v2, v2, v3
            0xD834_0000,
            0x0000_0200, // ds_write_b32 v0, v2
            S_ENDPGM,
        ];
        let used = vec![
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 4,
                dwords: 4,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 8,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 16,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 24,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 0,
                slot: 0,
            },
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 32,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            },
        ];
        let lt = lower_with(&table, vec![root(RootType::Table, 0, 0)], used).unwrap();
        validate_dxil(&lt.dxil);
        assert_eq!(
            lt.heap_props,
            vec![
                (0x100C, 4),     // descriptor_data for structured UAV
                (0x100A, 0x409), // descriptor_data for typed UAV
                (0x1002, 0x409), // descriptor_data for 2D image UAV
                (0x1002, 0x409), // descriptor_data for 2D uint image UAV
                (0x000B, 0),     // descriptor_data + glc load for table SRV
                (0x500C, 4),     // structured UAV glc
                (0x500A, 0x409), // typed UAV glc (shared by load and store)
                (0x5002, 0x409), // 2D image UAV glc (shared by load and store)
                (0x5002, 0x105), // 2D uint image atomic glc
            ]
        );
    }

    /// `s_buffer_load_dword s8, s[4:7], 0` on the descriptor loaded with
    /// `s_load_dwordx4 s[4:7], s[2:3], 0x40` from the root parameter at s0.
    const SCALAR_LOAD_FROM_S0: &[u32] = &[
        0xBE82_0300, // s_mov_b32 s2, s0
        0xBE83_0380, // s_mov_b32 s3, 0
        0xF408_0101,
        0xFA00_0040, // s_load_dwordx4 s[4:7], s[2:3], 0x40
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xF420_0202,
        0xFA00_0000, // s_buffer_load_dword s8, s[4:7], 0
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        S_ENDPGM,
    ];

    #[test]
    fn scalar_loads_from_a_resource_heap_descriptor_are_raw() {
        // No used-resource entry names the descriptor: loaded from the
        // resource heap it is a buffer view the host makes raw, so the load
        // must not be a constant-buffer read (a CBV handle on a raw slot).
        let l = lower_with(
            SCALAR_LOAD_FROM_S0,
            vec![root(RootType::ResourceHeap, 0, 0)],
            vec![],
        )
        .unwrap();
        assert_eq!(l.heap_raw_loads, 1);
        // The same load from a table stays a constant-buffer read.
        let l = lower_with(
            SCALAR_LOAD_FROM_S0,
            vec![root(RootType::Table, 0, 0)],
            vec![],
        )
        .unwrap();
        assert_eq!(l.heap_raw_loads, 0);
    }
