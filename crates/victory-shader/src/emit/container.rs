//! The PC container around emitted DXIL: feature flags (`SFI0`), the
//! signatures (`ISG1`, `OSG1`), pipeline state validation (`PSV0`) and the
//! program (`DXIL`), laid out as validator version 1.6 lays them out, plus
//! the container hash that D3D12 checks.

/// One signature element: one row (D3D12 links stages by semantic).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SigElement {
    pub name: String,
    pub index: u32,
    /// DXIL semantic kind (0 arbitrary, 1 vertex id, 3 position, 16 target...).
    pub kind: u8,
    /// DXIL component type (4 i32, 5 u32, 9 f32, 1 i1).
    pub comp: u8,
    /// DXIL interpolation mode (0 for none).
    pub interp: u8,
    /// Start row, if the element has a register.
    pub row: Option<u32>,
    pub col: u8,
    pub cols: u8,
    /// Components read (inputs) or written (outputs), relative to `col`.
    pub usage: u8,
    /// 16-bit on the console (declared 32-bit here).
    pub half: bool,
}

impl SigElement {
    fn system_value(&self) -> u32 {
        match self.kind {
            1 => 6,  // vertex id
            2 => 8,  // instance id
            3 => 1,  // position
            4 => 4,  // render target array index
            5 => 5,  // viewport array index
            6 => 2,  // clip distance
            7 => 3,  // cull distance
            10 => 7, // primitive id
            12 => 10,
            13 => 9, // front face
            14 => 66,
            15 => 70,
            16 => 64, // target
            17 => 65, // depth
            18 => 68,
            19 => 67,
            20 => 69,
            28 => 23, // barycentrics
            30 => 25, // cull primitive
            _ => 0,
        }
    }

    fn sig_comp(&self) -> u32 {
        match self.comp {
            1 | 5 => 1, // bool and u32
            4 => 2,
            9 => 3,
            _ => 0,
        }
    }

    fn mask(&self) -> u8 {
        ((1u8 << self.cols) - 1) << self.col
    }
}

/// The console packs elements of different interpolation modes into one
/// row (a constant front face beside a linear coordinate, say); DXIL needs
/// one mode per row. Each later mode of a row moves to a free row of its
/// own, shared by that row's elements of the mode. Code addresses elements
/// by id and stages link by semantic, so only the signatures change.
/// Elements without a mode (0) are left where they are.
pub(crate) fn separate_interpolation(elements: &mut [SigElement]) {
    let mut next = elements
        .iter()
        .filter_map(|e| e.row)
        .max()
        .map_or(0, |r| r + 1);
    // (row, the row's first mode)
    let mut first: Vec<(u32, u8)> = Vec::new();
    // ((old row, mode), new row)
    let mut moved: Vec<((u32, u8), u32)> = Vec::new();
    for e in elements.iter_mut() {
        let (Some(row), mode) = (e.row, e.interp) else {
            continue;
        };
        if mode == 0 {
            continue;
        }
        match first.iter().find(|f| f.0 == row) {
            None => first.push((row, mode)),
            Some(&(_, m)) if m == mode => {}
            Some(_) => {
                let to = match moved.iter().find(|m| m.0 == (row, mode)) {
                    Some(&(_, to)) => to,
                    None => {
                        moved.push(((row, mode), next));
                        next += 1;
                        next - 1
                    }
                };
                e.row = Some(to);
            }
        }
    }
}

/// `ISG1`/`OSG1`: 32-byte rows, then the names.
pub(crate) fn signature(elements: &[SigElement], input: bool) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend((elements.len() as u32).to_le_bytes());
    out.extend(8u32.to_le_bytes());
    // Validator 1.6 writes one name per element in element order, repeats
    // included, except that system values share theirs (it keys names by
    // the semantic's string, which is one static string per system value);
    // the rows sorted by register (unallocated last), then by name offset,
    // not column (its `sort_sig`): rows of one register keep element order,
    // apart from system values.
    let mut offsets: Vec<u32> = Vec::new();
    let mut next = 8 + 32 * elements.len() as u32;
    let mut names = Vec::new();
    for (i, e) in elements.iter().enumerate() {
        let shared = (e.kind != 0)
            .then(|| (0..i).find(|&j| elements[j].kind == e.kind))
            .flatten();
        match shared {
            Some(j) => offsets.push(offsets[j]),
            None => {
                offsets.push(next);
                next += e.name.len() as u32 + 1;
                names.push(&e.name);
            }
        }
    }
    let mut order: Vec<usize> = (0..elements.len()).collect();
    order.sort_by_key(|&i| (elements[i].row.unwrap_or(u32::MAX), offsets[i]));
    for i in order {
        let (e, offset) = (&elements[i], offsets[i]);
        let used = e.usage << e.col;
        let rw = if input { used } else { !used & 0xf };
        for v in [
            0,
            offset,
            e.index,
            e.system_value(),
            e.sig_comp(),
            e.row.unwrap_or(u32::MAX),
        ] {
            out.extend(v.to_le_bytes());
        }
        out.extend([e.mask(), rw, 0, 0]);
        out.extend(0u32.to_le_bytes());
    }
    for name in names {
        out.extend(name.as_bytes());
        out.push(0);
    }
    out
}

/// A resource bound by register, as `PSV0` lists it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Binding {
    /// PSV resource type: 1 sampler, 2 CBV, 4 raw SRV, 7 raw UAV.
    pub kind: u32,
    pub space: u32,
    pub register: u32,
    /// DXIL resource kind (11 raw buffer, 13 cbuffer, 14 sampler).
    pub dxil_kind: u32,
}

pub(crate) struct Psv<'a> {
    /// PSV shader kind: 0 pixel, 1 vertex, 5 compute.
    pub stage: u8,
    /// Vertex: writes `SV_Position`. Pixel: writes depth.
    pub flag: bool,
    pub inputs: &'a [SigElement],
    pub outputs: &'a [SigElement],
    /// Mesh: per-primitive outputs.
    pub prims: &'a [SigElement],
    /// Mesh: its state and the group-shared bytes it declares.
    pub mesh: Option<(super::shader::Mesh, u32)>,
    pub gs: Option<super::shader::Geometry>,
    pub bindings: &'a [Binding],
    /// Compute: thread-group size and required wave size.
    pub threads: [u32; 3],
    pub wave_size: Option<u32>,
}

/// `PSV0` with runtime info version 2 (validator 1.6).
pub(crate) fn psv(p: &Psv) -> Vec<u8> {
    let vectors = |els: &[SigElement]| {
        els.iter()
            .filter_map(|e| e.row)
            .map(|r| r + 1)
            .max()
            .unwrap_or(0)
    };
    let (in_vectors, out_vectors) = (vectors(p.inputs), vectors(p.outputs));
    let mut out = Vec::new();
    let u32s = |out: &mut Vec<u8>, vs: &[u32]| {
        for v in vs {
            out.extend(v.to_le_bytes());
        }
    };
    u32s(&mut out, &[48]);
    let mut info = [0u8; 48];
    info[0] = u8::from(p.flag);
    if p.stage == 0 {
        // PSInfo.SampleFrequency: an input interpolated per sample, or the
        // sample index.
        info[1] = u8::from(
            p.inputs
                .iter()
                .any(|e| matches!(e.interp, 6 | 7) || e.kind == 12),
        );
    }
    let (min, max) = p.wave_size.map_or((0, u32::MAX), |n| (n, n));
    info[16..20].copy_from_slice(&min.to_le_bytes());
    info[20..24].copy_from_slice(&max.to_le_bytes());
    info[24] = p.stage;
    for (k, n) in p.threads.into_iter().enumerate() {
        info[36 + 4 * k..40 + 4 * k].copy_from_slice(&n.to_le_bytes());
    }
    info[28] = p.inputs.len() as u8;
    info[29] = p.outputs.len() as u8;
    info[31] = in_vectors as u8;
    info[32] = out_vectors as u8;
    if let Some((mesh, shared)) = p.mesh {
        info[0..4].copy_from_slice(&shared.to_le_bytes());
        info[12..14].copy_from_slice(&(mesh.max_vertices as u16).to_le_bytes());
        info[14..16].copy_from_slice(&(mesh.max_primitives as u16).to_le_bytes());
        info[26] = vectors(p.prims) as u8;
        info[27] = mesh.topology;
        info[30] = p.prims.len() as u8;
    }
    // Geometry: input primitive, output topology, stream mask, position
    // output; the maximum vertex count.
    if let Some(gs) = p.gs {
        info[0..4].copy_from_slice(&gs.input.to_le_bytes());
        info[4..8].copy_from_slice(&gs.topology.to_le_bytes());
        info[8..12].copy_from_slice(&1u32.to_le_bytes());
        info[12] = u8::from(p.flag);
        info[26..28].copy_from_slice(&(gs.max_vertices as u16).to_le_bytes());
    }
    out.extend(info);
    u32s(&mut out, &[p.bindings.len() as u32]);
    if !p.bindings.is_empty() {
        u32s(&mut out, &[24]);
        for b in p.bindings {
            u32s(
                &mut out,
                &[b.kind, b.space, b.register, b.register, b.dxil_kind, 0],
            );
        }
    }
    // Names of arbitrary semantics (system values use ""), then indices.
    let mut strings = vec![0u8];
    let mut indices: Vec<u32> = Vec::new();
    let mut elements = Vec::new();
    for e in p.inputs.iter().chain(p.outputs).chain(p.prims) {
        let name = if e.kind == 0 {
            let needle: Vec<u8> = e.name.bytes().chain([0]).collect();
            match find(&strings, &needle) {
                Some(at) => at,
                None => {
                    strings.extend(&needle);
                    strings.len() - needle.len()
                }
            }
        } else {
            0
        };
        let index = match indices.iter().position(|&i| i == e.index) {
            Some(at) => at,
            None => {
                indices.push(e.index);
                indices.len() - 1
            }
        };
        let (start, alloc) = match e.row {
            Some(r) => (r as u8, 0x40 | (e.col << 4)),
            None => (0, 0),
        };
        elements.push((name as u32, index as u32, start, e.cols | alloc, e));
    }
    strings.resize(strings.len().next_multiple_of(4), 0);
    u32s(&mut out, &[strings.len() as u32]);
    out.extend(&strings);
    u32s(&mut out, &[indices.len() as u32]);
    u32s(&mut out, &indices);
    if !elements.is_empty() {
        u32s(&mut out, &[16]);
        for (name, index, start, cols, e) in elements {
            u32s(&mut out, &[name, index]);
            out.extend([1, start, cols, e.kind, e.sig_comp() as u8, e.interp, 0, 0]);
        }
    }
    // Input-to-output dependencies, as the view-id state declares them:
    // every output scalar on every input scalar.
    if in_vectors > 0 && out_vectors > 0 {
        let scalars = |els: &[SigElement]| {
            els.iter()
                .filter_map(|e| e.row.map(|r| r * 4 + u32::from(e.col + e.cols)))
                .max()
                .unwrap_or(0)
        };
        let (ni, no) = (scalars(p.inputs), scalars(p.outputs));
        let per = (out_vectors * 4).div_ceil(32);
        for i in 0..in_vectors * 4 {
            for k in 0..per {
                let bits = if i < ni {
                    no.saturating_sub(32 * k).min(32)
                } else {
                    0
                };
                let mask = if bits == 32 {
                    u32::MAX
                } else {
                    (1u32 << bits) - 1
                };
                u32s(&mut out, &[mask]);
            }
        }
    }
    out
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// The container: parts in order, hashed.
pub(crate) fn build(parts: &[(&[u8; 4], &[u8])]) -> Vec<u8> {
    let table = 0x20 + 4 * parts.len();
    let total = table + parts.iter().map(|(_, d)| 8 + d.len()).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    out.extend(b"DXBC");
    out.extend([0; 16]);
    out.extend(1u16.to_le_bytes());
    out.extend(0u16.to_le_bytes());
    out.extend((total as u32).to_le_bytes());
    out.extend((parts.len() as u32).to_le_bytes());
    let mut offset = table;
    for (_, d) in parts {
        out.extend((offset as u32).to_le_bytes());
        offset += 8 + d.len();
    }
    for (fourcc, d) in parts {
        out.extend(*fourcc);
        out.extend((d.len() as u32).to_le_bytes());
        out.extend(*d);
    }
    let digest = hash(&out[20..]);
    out[4..20].copy_from_slice(&digest);
    out
}

/// The container hash D3D12 checks: MD5 whose last block carries the bit
/// count first and `2 × bytes + 1` last instead of standard padding.
pub(crate) fn hash(data: &[u8]) -> [u8; 16] {
    let len = data.len();
    let left = len % 64;
    let mut tail = Vec::with_capacity(128);
    if left < 56 {
        tail.extend(((len as u32) << 3).to_le_bytes());
        tail.extend(&data[len - left..]);
        tail.push(0x80);
        tail.resize(60, 0);
    } else {
        tail.extend(&data[len - left..]);
        tail.push(0x80);
        tail.resize(64, 0);
        tail.extend(((len as u32) << 3).to_le_bytes());
        tail.resize(124, 0);
    }
    tail.extend(((len as u32) << 1 | 1).to_le_bytes());
    let mut state = [0x6745_2301u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    for block in data[..len - left]
        .as_chunks::<64>()
        .0
        .iter()
        .chain(tail.as_chunks::<64>().0)
    {
        md5_block(&mut state, block);
    }
    let mut out = [0; 16];
    for (o, s) in out.as_chunks_mut::<4>().0.iter_mut().zip(state) {
        o.copy_from_slice(&s.to_le_bytes());
    }
    out
}

fn md5_block(state: &mut [u32; 4], block: &[u8; 64]) {
    const S: [u32; 16] = [7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21];
    let x: Vec<u32> = block
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let [mut a, mut b, mut c, mut d] = *state;
    for i in 0..64 {
        let (f, g) = match i / 16 {
            0 => ((b & c) | (!b & d), i),
            1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
            2 => (b ^ c ^ d, (3 * i + 5) % 16),
            _ => (c ^ (b | !d), (7 * i) % 16),
        };
        let k = ((i as f64 + 1.0).sin().abs() * 4_294_967_296.0) as u32;
        let rotated = a
            .wrapping_add(f)
            .wrapping_add(k)
            .wrapping_add(x[g])
            .rotate_left(S[(i / 16) * 4 + i % 4]);
        (a, b, c, d) = (d, b.wrapping_add(rotated), b, c);
    }
    for (s, v) in state.iter_mut().zip([a, b, c, d]) {
        *s = s.wrapping_add(v);
    }
}

#[cfg(test)]
mod tests {
    use super::super::shader::Mesh;
    use super::*;

    fn words(bytes: &[u8]) -> Vec<u32> {
        let (words, remainder) = bytes.as_chunks::<4>();
        assert!(remainder.is_empty());
        words.iter().copied().map(u32::from_le_bytes).collect()
    }

    fn element(name: &str, row: Option<u32>, col: u8, cols: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row,
            col,
            cols,
            usage: 1,
            half: false,
        }
    }

    fn empty_psv(stage: u8) -> Psv<'static> {
        Psv {
            stage,
            flag: false,
            inputs: &[],
            outputs: &[],
            prims: &[],
            mesh: None,
            gs: None,
            bindings: &[],
            threads: [0; 3],
            wave_size: None,
        }
    }

    #[test]
    fn md5_core_matches_the_standard() {
        // Standard MD5 of "abc" with standard padding, through md5_block.
        let mut block = b"abc".to_vec();
        block.push(0x80);
        block.resize(56, 0);
        block.extend(24u64.to_le_bytes());
        let mut state = [0x6745_2301u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
        md5_block(&mut state, block.as_slice().try_into().unwrap());
        let hex: String = state
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(hex, "900150983cd24fb0d6963f7d28e17f72");
    }

    fn interp_element(name: &str, row: u32, col: u8, interp: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp,
            row: Some(row),
            col,
            cols: 1,
            usage: 1,
            half: false,
        }
    }

    #[test]
    fn rows_of_mixed_interpolation_are_separated() {
        let mut e = vec![
            interp_element("A", 0, 0, 2),
            interp_element("B", 0, 1, 1),
            interp_element("C", 1, 0, 2),
            interp_element("D", 0, 2, 1),
            interp_element("E", 0, 3, 2),
            interp_element("F", 1, 1, 0),
        ];
        separate_interpolation(&mut e);
        let rows: Vec<u32> = e.iter().map(|e| e.row.unwrap()).collect();
        // B and D (constant) share a new row; A and E stay; F has no mode.
        assert_eq!(rows, [0, 2, 1, 2, 0, 1]);
    }

    #[test]
    fn rows_of_one_register_keep_element_order_not_column_order() {
        let element = |name: &str, index, kind, row, col, cols| SigElement {
            name: name.into(),
            index,
            kind,
            comp: 9,
            interp: if kind == 0 { 2 } else { 4 },
            row: Some(row),
            col,
            cols,
            usage: (1 << cols) - 1,
            half: false,
        };
        // As a vertex program declares them: a w channel before the xyz of
        // the same register.
        let elements = [
            element("SV_Position", 0, 3, 0, 0, 4),
            element("TEXCOORD", 3, 0, 1, 3, 1),
            element("NORMAL", 0, 0, 1, 0, 3),
        ];
        let part = signature(&elements, false);
        let rows: Vec<(u32, u32)> = (0..3)
            .map(|r| {
                let word = |k: usize| {
                    let at = 8 + 32 * r + 4 * k;
                    u32::from_le_bytes(part[at..at + 4].try_into().unwrap())
                };
                // Semantic index and register.
                (word(2), word(5))
            })
            .collect();
        // Register order, then name offset (element order here), as the
        // validator writes them: TEXCOORD3 before NORMAL0 in register 1.
        assert_eq!(rows, [(0, 0), (3, 1), (0, 1)]);
    }

    #[test]
    fn hash_matches_synthetic_vectors_at_padding_boundaries() {
        // Calculated independently by applying MD5 compression to the
        // retail-format padded byte sequences 0, 1, ..., len - 1.
        let vectors = [
            (
                0,
                [
                    0x14, 0x0d, 0x60, 0xf6, 0xb7, 0x75, 0xe2, 0xba, 0x4e, 0x4a, 0xbe, 0xd4, 0x01,
                    0xb2, 0xe9, 0xa1,
                ],
            ),
            (
                55,
                [
                    0x84, 0x2e, 0x55, 0x53, 0x4b, 0xa9, 0x3d, 0xaa, 0x93, 0xe9, 0x4b, 0xd5, 0xaf,
                    0x9b, 0x0c, 0x03,
                ],
            ),
            (
                56,
                [
                    0x00, 0xf9, 0xcc, 0x96, 0x4f, 0xf2, 0xec, 0x81, 0x95, 0x9d, 0x4a, 0x3f, 0x09,
                    0x2c, 0xe6, 0x3f,
                ],
            ),
            (
                63,
                [
                    0x44, 0x00, 0xd4, 0x28, 0xa4, 0x1d, 0x0e, 0x75, 0xdc, 0x98, 0xb9, 0x36, 0x69,
                    0x3f, 0x7b, 0xfb,
                ],
            ),
            (
                64,
                [
                    0xf4, 0x2e, 0xb0, 0x6c, 0xa9, 0x21, 0xc8, 0x78, 0xe4, 0x35, 0xe3, 0xb8, 0xd4,
                    0xf9, 0x2b, 0x13,
                ],
            ),
            (
                119,
                [
                    0xba, 0x3a, 0x67, 0xec, 0xd4, 0x64, 0x1c, 0xe4, 0xed, 0x08, 0xf2, 0x91, 0x89,
                    0x6e, 0x1b, 0xc6,
                ],
            ),
            (
                120,
                [
                    0x88, 0x2f, 0xce, 0x4f, 0x4c, 0x95, 0x19, 0x43, 0x92, 0xe7, 0xad, 0xef, 0xd5,
                    0x3f, 0xe3, 0x43,
                ],
            ),
        ];
        let data: Vec<u8> = (0..120).collect();
        for (len, expected) in vectors {
            assert_eq!(hash(&data[..len]), expected, "length {len}");
        }
    }

    #[test]
    fn empty_container_has_only_the_signed_header() {
        let bytes = build(&[]);
        assert_eq!(&bytes[..4], b"DXBC");
        assert_eq!(&bytes[20..], &[1, 0, 0, 0, 32, 0, 0, 0, 0, 0, 0, 0]);
        assert_eq!(
            words(&bytes[4..20]),
            [0x4c07_86fe, 0xe9c2_2f40, 0xa34f_0f0e, 0xc05b_06bf]
        );
    }

    #[test]
    fn container_offsets_are_contiguous_after_an_unpadded_signature() {
        let bytes = build(&[
            (b"ISG1", &[0x11, 0x22, 0x33]),
            (b"OSG1", &[]),
            (b"PRIV", &[]),
        ]);
        assert_eq!(&bytes[..4], b"DXBC");
        assert_eq!(words(&bytes[20..44]), [1, 71, 3, 44, 55, 63]);
        assert_eq!(&bytes[44..55], b"ISG1\x03\0\0\0\x11\x22\x33");
        assert_eq!(&bytes[55..63], b"OSG1\0\0\0\0");
        assert_eq!(&bytes[63..], b"PRIV\0\0\0\0");
        assert_eq!(
            words(&bytes[4..20]),
            [0xe8e8_9b79, 0x8857_cd6c, 0xf7ec_2ab3, 0x137d_e43d]
        );
    }

    #[test]
    fn signature_sorts_rows_and_preserves_legacy_name_offsets_and_masks() {
        assert_eq!(words(&signature(&[], true)), [0, 8]);
        let elements = [
            SigElement {
                index: 7,
                ..element("COLOR", Some(1), 2, 2)
            },
            SigElement {
                index: 1,
                kind: 16,
                usage: 15,
                ..element("SV_Target", Some(3), 0, 4)
            },
            SigElement {
                index: 8,
                comp: 4,
                usage: 2,
                ..element("COLOR", Some(1), 0, 2)
            },
            SigElement {
                kind: 16,
                usage: 15,
                ..element("SV_Target", Some(2), 0, 4)
            },
            SigElement {
                kind: 17,
                ..element("SV_Depth", None, 0, 1)
            },
        ];
        let input = signature(&elements, true);
        let output = signature(&elements, false);
        let rows = [
            [0, 168, 7, 0, 3, 1, 0x040c, 0],
            [0, 184, 8, 0, 2, 1, 0x0203, 0],
            [0, 174, 0, 64, 3, 2, 0x0f0f, 0],
            [0, 174, 1, 64, 3, 3, 0x0f0f, 0],
            [0, 190, 0, 65, 3, u32::MAX, 0x0101, 0],
        ];
        assert_eq!(words(&input[..8]), [5, 8]);
        assert_eq!(words(&output[..8]), [5, 8]);
        assert_eq!(&input[168..], b"COLOR\0SV_Target\0COLOR\0SV_Depth\0");
        assert_eq!(&output[168..], &input[168..]);
        for (i, (row, output_mask)) in rows.into_iter().zip([0x0b, 0x0d, 0, 0, 0x0e]).enumerate() {
            let start = 8 + 32 * i;
            assert_eq!(words(&input[start..start + 32]), row);
            let mut expected = row;
            expected[6] = (row[6] & 0xff) | (output_mask << 8);
            assert_eq!(words(&output[start..start + 32]), expected);
        }
    }

    #[test]
    fn compute_psv_contains_thread_and_wave_sizes_with_empty_tables() {
        let bytes = psv(&Psv {
            threads: [8, 4, 2],
            wave_size: Some(32),
            ..empty_psv(5)
        });
        assert_eq!(
            words(&bytes),
            [48, 0, 0, 0, 0, 32, 32, 5, 0, 0, 8, 4, 2, 0, 4, 0, 0]
        );
    }

    #[test]
    fn pixel_sample_frequency_depends_on_sample_interpolation_or_semantic() {
        for (interp, kind, expected) in [(5, 0, 0), (6, 0, 1), (7, 0, 1), (0, 12, 1)] {
            let inputs = [SigElement {
                interp,
                kind,
                ..element("UV", Some(0), 0, 1)
            }];
            let bytes = psv(&Psv {
                flag: true,
                inputs: &inputs,
                ..empty_psv(0)
            });
            assert_eq!(&bytes[4..8], &[1, expected, 0, 0]);
            assert_eq!(words(&bytes[20..28]), [0, u32::MAX]);
        }
    }

    #[test]
    fn vertex_psv_encodes_bindings_tables_and_cross_word_dependencies() {
        let inputs = [SigElement {
            index: 3,
            interp: 2,
            ..element("COLOR", Some(0), 1, 2)
        }];
        let outputs = [SigElement {
            index: 3,
            ..element("OR", Some(8), 1, 1)
        }];
        let bytes = psv(&Psv {
            inputs: &inputs,
            outputs: &outputs,
            bindings: &[
                Binding {
                    kind: 1,
                    space: 6,
                    register: 9,
                    dxil_kind: 14,
                },
                Binding {
                    kind: 2,
                    space: 7,
                    register: 2,
                    dxil_kind: 13,
                },
            ],
            ..empty_psv(1)
        });
        assert_eq!(&bytes[28..40], &[1, 0, 0, 0, 1, 1, 0, 1, 9, 0, 0, 0]);
        assert_eq!(
            words(&bytes[52..112]),
            [2, 24, 1, 6, 9, 9, 14, 0, 2, 7, 2, 2, 13, 0, 8]
        );
        assert_eq!(&bytes[112..120], b"\0COLOR\0\0");
        assert_eq!(words(&bytes[120..132]), [1, 3, 16]);
        assert_eq!(
            words(&bytes[132..164]),
            [1, 0, 0x0052_0001, 0x0203, 4, 0, 0x0051_0801, 3]
        );
        assert_eq!(
            words(&bytes[164..]),
            [u32::MAX, 3, u32::MAX, 3, u32::MAX, 3, 0, 0]
        );
    }

    #[test]
    fn mesh_psv_encodes_shared_memory_and_primitive_outputs() {
        let outputs = [SigElement {
            kind: 3,
            ..element("SV_Position", Some(0), 0, 4)
        }];
        let prims = [SigElement {
            kind: 30,
            comp: 1,
            index: 2,
            ..element("SV_CullPrimitive", Some(1), 3, 1)
        }];
        let bytes = psv(&Psv {
            outputs: &outputs,
            prims: &prims,
            mesh: Some((
                Mesh {
                    max_vertices: 64,
                    max_primitives: 127,
                    topology: 2,
                },
                1024,
            )),
            threads: [32, 1, 1],
            ..empty_psv(13)
        });
        assert_eq!(words(&bytes[..20]), [48, 1024, 0, 0, 0x007f_0040]);
        assert_eq!(&bytes[28..40], &[13, 0, 2, 2, 0, 1, 1, 0, 1, 0, 0, 0]);
        assert_eq!(
            words(&bytes[40..]),
            [
                32,
                1,
                1,
                0,
                4,
                0,
                2,
                0,
                2,
                16,
                0,
                0,
                0x0344_0001,
                3,
                0,
                1,
                0x1e71_0101,
                1
            ]
        );
    }
}
