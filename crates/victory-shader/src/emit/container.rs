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

/// Read the mesh shader's declared interpolation instead of deriving it from the value type.
/// The vertex outputs follow the input records in PSV0. OSG1 can list the same elements
/// in a different order, so apply the modes only after matching the two descriptions.
pub(super) fn mesh_output_modes(psv: &[u8], osg: &[u8], outputs: &[SigElement]) -> Option<Vec<u8>> {
    use crate::dxbc::u32_at;

    fn word(bytes: &[u8]) -> Option<usize> {
        Some(u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?) as usize)
    }

    fn table(bytes: &[u8], stride: usize) -> Option<(&[u8], &[u8])> {
        bytes
            .get(4..)?
            .split_at_checked(word(bytes)?.checked_mul(stride)?)
    }

    let (info, mut tables) = table(psv, 1)?;
    if info.len() < 36 || info[24] != 13 {
        return None;
    }
    let [inputs, vertices, primitives] = [info[28], info[29], info[30]].map(usize::from);
    if vertices == 0 {
        return None;
    }
    let resources = word(tables)?;
    tables = tables.get(4..)?;
    if resources != 0 {
        let stride = word(tables)?;
        if stride < 16 {
            return None;
        }
        tables = tables.get(4..)?.get(resources.checked_mul(stride)?..)?;
    }
    let (names, tables) = table(tables, 1)?;
    if names.len() % 4 != 0 {
        return None;
    }
    let (indices, tables) = table(tables, 4)?;
    let stride = word(tables)?;
    if stride < 16 {
        return None;
    }
    let records = tables
        .get(4..)?
        .get(..(inputs + vertices + primitives).checked_mul(stride)?)?;
    let vertices = &records[inputs * stride..(inputs + vertices) * stride];
    let osg_rows = osg.get(u32_at(osg, 4).ok()? as usize..)?;
    let mut modes: Vec<_> = outputs.iter().map(|e| e.interp).collect();
    for (i, output) in outputs.iter().enumerate() {
        let Some(row) = output.row.filter(|_| output.kind == 0 && output.comp == 9) else {
            continue;
        };
        let raw = osg_rows.get(i * 32..i * 32 + 32)?;
        let mask = raw[24];
        if mask == 0 || mask & 0xf0 != 0 || mask != output.mask() {
            return None;
        }
        let raw_comp = u32_at(raw, 16).ok()?;
        let mut matched = None;
        for record in vertices.chunks_exact(stride) {
            let cols = record[10];
            if record[11] != 0
                || u32::from(record[12]) != raw_comp
                || cols & 0x40 == 0
                || u32::from(record[9]) != row
                || cols & 0x3f != output.cols | (output.col << 4)
            {
                continue;
            }
            // A multirow element has one index for each row. This path only
            // uses one-row elements with an exact OSG1 counterpart.
            if record[8] != 1 || cols & 0x80 != 0 || record[14] & 0x30 != 0 {
                return None;
            }
            let index = word(&record[4..])?.checked_mul(4)?;
            let name = names.get(word(record)?..)?;
            let name = name.get(..name.iter().position(|b| *b == 0)?)?;
            if u32_at(indices.get(index..)?, 0).ok()? != output.index
                || name != output.name.as_bytes()
            {
                continue;
            }
            let mode = record[13];
            if !(1..=7).contains(&mode) || matched.replace(mode).is_some() {
                return None;
            }
        }
        modes[i] = matched?;
    }
    // The host requires one defined mode for every member of a packed row.
    for (i, output) in outputs.iter().enumerate() {
        if modes[i] != output.interp
            && outputs
                .iter()
                .zip(&modes)
                .any(|(other, &mode)| other.row == output.row && mode != 0 && mode != modes[i])
        {
            return None;
        }
    }
    Some(modes)
}

/// `ISG1`/`OSG1`: 32-byte rows, then the names.
pub(crate) fn signature(elements: &[SigElement], input: bool) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend((elements.len() as u32).to_le_bytes());
    out.extend(8u32.to_le_bytes());
    // Validator 1.6 writes one name per element in element order, repeats
    // included, except that system values share theirs (it keys names by
    // the semantic's string, which is one static string per system value);
    // the rows sorted by register (unallocated last), then by name offset:
    // of two elements packed into one register, the one first in element
    // order comes first whatever its columns (a sort by column made the
    // validator refuse 72 of the title's vertex, mesh and pixel programs:
    // "does not match expected for module").
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

/// One interpolation mode per register. The console's compiler packs only elements of one
/// mode into a register, and the validator refuses two ("has interpolation mode that differs
/// from another element packed into the same row"), but the modes here are partly defaults: a
/// float is linear unless a pixel program's reads say otherwise, so a float the console
/// declared `nointerpolation` beside an integer came out linear. Where a register's modes
/// differ, the elements whose mode is fixed give it to the others if they agree on one: an
/// integer or the front face (constant only) and, with `read` (pixel inputs), an element the
/// program reads (the mode its reads asked for). The others' mode decides nothing: an
/// output's interpolation is the pixel input's, and an input no read asks for is never
/// interpolated. A register whose modes agree stays as it was.
/// The title's 70 programs that packed two modes into a register (23 vertex and mesh, 47
/// pixel) each had a float of the console's constant mode (its `PSV0`) beside an integer, the
/// front face, or an input read at its vertices, that float written (outputs) or not read
/// (inputs).
pub(crate) fn one_mode_per_register(elements: &mut [SigElement], read: bool) {
    let fixed = |e: &SigElement| e.comp != 9 || e.kind == 13 || (read && e.usage != 0);
    let mut rows: Vec<u32> = elements.iter().filter_map(|e| e.row).collect();
    rows.sort_unstable();
    rows.dedup();
    for row in rows {
        // (an undefined mode, 0, takes the others' in the validator)
        let of = |e: &SigElement| e.row == Some(row) && e.interp != 0;
        let mut modes: Vec<u8> = elements
            .iter()
            .filter(|e| of(e))
            .map(|e| e.interp)
            .collect();
        modes.sort_unstable();
        modes.dedup();
        if modes.len() < 2 {
            continue;
        }
        let mut given: Vec<u8> = elements
            .iter()
            .filter(|e| of(e) && fixed(e))
            .map(|e| e.interp)
            .collect();
        given.sort_unstable();
        given.dedup();
        let [mode] = given[..] else { continue };
        for e in elements.iter_mut().filter(|e| of(e) && !fixed(e)) {
            e.interp = mode;
        }
    }
}

/// The most a mesh shader's outputs may take, in bytes (D3D12).
pub(crate) const MESH_OUTPUT_LIMIT: u64 = 32 * 1024;

/// A mesh shader's output size as the DXIL validator measures it against
/// [`MESH_OUTPUT_LIMIT`]: 4 bytes per component of every vertex element
/// and primitive element, times the maxima rounded up to multiples of 32.
pub(crate) fn mesh_output_bytes(
    vertex: &[SigElement],
    primitive: &[SigElement],
    max_vertices: u32,
    max_primitives: u32,
) -> u64 {
    let scalars = |els: &[SigElement]| els.iter().map(|e| u64::from(e.cols)).sum::<u64>();
    let align = |n: u32| u64::from(n).next_multiple_of(32);
    4 * (scalars(vertex) * align(max_vertices) + scalars(primitive) * align(max_primitives))
}

/// A resource bound by register, as `PSV0` lists it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Binding {
    /// PSV resource type: 1 sampler, 2 CBV, 4 raw SRV, 7 raw UAV.
    pub kind: u32,
    pub space: u32,
    pub register: u32,
    /// DXIL resource kind (13 cbuffer, 14 sampler).
    pub dxil_kind: u32,
}

pub(crate) struct Psv<'a> {
    /// PSV shader kind: 0 pixel, 1 vertex, 2 geometry, 5 compute.
    pub stage: u8,
    /// Vertex: writes `SV_Position`. Pixel: writes depth.
    pub flag: bool,
    pub inputs: &'a [SigElement],
    pub outputs: &'a [SigElement],
    /// Mesh: per-primitive outputs.
    pub prims: &'a [SigElement],
    /// Mesh: its state and the group-shared bytes it declares.
    pub mesh: Option<(super::shader::Mesh, u32)>,
    /// Geometry (triangles in, triangle strips out on stream 0): its
    /// maximum vertex count. `flag`: writes `SV_Position`.
    pub geometry: Option<u32>,
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
    if let Some(max_vertices) = p.geometry {
        // Input primitive (3: triangles), output topology (5: triangle
        // strips), stream mask, position written; the vertex maximum.
        info[0..4].copy_from_slice(&3u32.to_le_bytes());
        info[4..8].copy_from_slice(&5u32.to_le_bytes());
        info[8..12].copy_from_slice(&1u32.to_le_bytes());
        info[12] = u8::from(p.flag);
        info[26..28].copy_from_slice(&(max_vertices as u16).to_le_bytes());
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
    let (blocks, rest) = data.as_chunks::<64>();
    let mut state = [0x6745_2301u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
    for block in blocks {
        md5_block(&mut state, block);
    }

    let len = data.len() as u32;
    let mut tail = [0; 64];
    if rest.len() < 56 {
        tail[4..4 + rest.len()].copy_from_slice(rest);
        tail[4 + rest.len()] = 0x80;
    } else {
        tail[..rest.len()].copy_from_slice(rest);
        tail[rest.len()] = 0x80;
        md5_block(&mut state, &tail);
        tail = [0; 64];
    }
    tail[..4].copy_from_slice(&(len << 3).to_le_bytes());
    tail[60..].copy_from_slice(&(len << 1 | 1).to_le_bytes());
    md5_block(&mut state, &tail);
    let mut out = [0; 16];
    for (o, s) in out.as_chunks_mut::<4>().0.iter_mut().zip(state) {
        o.copy_from_slice(&s.to_le_bytes());
    }
    out
}

fn md5_block(state: &mut [u32; 4], block: &[u8; 64]) {
    const K: [u32; 64] = [
        0xd76a_a478, 0xe8c7_b756, 0x2420_70db, 0xc1bd_ceee,
        0xf57c_0faf, 0x4787_c62a, 0xa830_4613, 0xfd46_9501,
        0x6980_98d8, 0x8b44_f7af, 0xffff_5bb1, 0x895c_d7be,
        0x6b90_1122, 0xfd98_7193, 0xa679_438e, 0x49b4_0821,
        0xf61e_2562, 0xc040_b340, 0x265e_5a51, 0xe9b6_c7aa,
        0xd62f_105d, 0x0244_1453, 0xd8a1_e681, 0xe7d3_fbc8,
        0x21e1_cde6, 0xc337_07d6, 0xf4d5_0d87, 0x455a_14ed,
        0xa9e3_e905, 0xfcef_a3f8, 0x676f_02d9, 0x8d2a_4c8a,
        0xfffa_3942, 0x8771_f681, 0x6d9d_6122, 0xfde5_380c,
        0xa4be_ea44, 0x4bde_cfa9, 0xf6bb_4b60, 0xbebf_bc70,
        0x289b_7ec6, 0xeaa1_27fa, 0xd4ef_3085, 0x0488_1d05,
        0xd9d4_d039, 0xe6db_99e5, 0x1fa2_7cf8, 0xc4ac_5665,
        0xf429_2244, 0x432a_ff97, 0xab94_23a7, 0xfc93_a039,
        0x655b_59c3, 0x8f0c_cc92, 0xffef_f47d, 0x8584_5dd1,
        0x6fa8_7e4f, 0xfe2c_e6e0, 0xa301_4314, 0x4e08_11a1,
        0xf753_7e82, 0xbd3a_f235, 0x2ad7_d2bb, 0xeb86_d391,
    ];
    // Four steps return the working registers to their original positions.
    // Each round specializes to its own mix, word schedule and rotations.
    #[inline]
    fn round(
        [mut a, mut b, mut c, mut d]: [u32; 4],
        rotations: [u32; 4],
        mix: impl Fn(u32, u32, u32) -> u32,
        addend: impl Fn(usize) -> u32,
    ) -> [u32; 4] {
        let step = |a: u32, b, c, d, i, rot| {
            let sum = a.wrapping_add(mix(b, c, d)).wrapping_add(addend(i));
            b.wrapping_add(sum.rotate_left(rot))
        };
        for i in [0, 4, 8, 12] {
            a = step(a, b, c, d, i, rotations[0]);
            d = step(d, a, b, c, i + 1, rotations[1]);
            c = step(c, d, a, b, i + 2, rotations[2]);
            b = step(b, c, d, a, i + 3, rotations[3]);
        }
        [a, b, c, d]
    }

    // Word indices repeat modulo 16, so each round can start its schedule at zero.
    let words = block.as_chunks::<4>().0;
    let add = |i: usize, word: usize| K[i].wrapping_add(u32::from_le_bytes(words[word]));
    let work = round(
        *state,
        [7, 12, 17, 22],
        |b, c, d| (b & c) | (!b & d),
        |i| add(i, i),
    );
    let work = round(
        work,
        [5, 9, 14, 20],
        |b, c, d| (d & b) | (!d & c),
        |i| add(16 + i, (5 * i + 1) % 16),
    );
    let work = round(
        work,
        [4, 11, 16, 23],
        |b, c, d| b ^ c ^ d,
        |i| add(32 + i, (3 * i + 5) % 16),
    );
    let work = round(
        work,
        [6, 10, 15, 21],
        |b, c, d| c ^ (b | !d),
        |i| add(48 + i, (7 * i) % 16),
    );
    for (s, v) in state.iter_mut().zip(work) {
        *s = s.wrapping_add(v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two elements packed into one register are written in element order, not by column
    /// (the title's packed pedestrian outputs: TEXCOORD22 in columns 1-2 of the register
    /// whose column 0 is COLOR0, listed after it).
    #[test]
    fn rows_of_one_register_keep_the_element_order() {
        let element = |name: &str, index, row, col, cols| SigElement {
            name: name.into(),
            index,
            kind: 0,
            comp: 9,
            interp: 2,
            row: Some(row),
            col,
            cols,
            usage: 0,
            half: false,
        };
        let part = signature(
            &[
                element("TEXCOORD", 19, 13, 0, 3),
                element("TEXCOORD", 22, 14, 1, 2),
                element("COLOR", 0, 14, 0, 1),
                element("TEXCOORD", 21, 15, 0, 3),
            ],
            false,
        );
        let u32_at = |at: usize| u32::from_le_bytes(part[at..at + 4].try_into().unwrap());
        // (register, semantic index, mask) of each 32-byte row after the 8-byte header
        let rows: Vec<(u32, u32, u8)> = (0..4)
            .map(|k| 8 + 32 * k)
            .map(|at| (u32_at(at + 20), u32_at(at + 8), part[at + 24]))
            .collect();
        assert_eq!(rows, [(13, 19, 0b111), (14, 22, 0b110), (14, 0, 0b1), (15, 21, 0b111)]);
    }

    /// The registers of the title's programs that packed two modes: a float beside an integer
    /// or the front face, an unread input beside one read at its vertices. Registers whose modes
    /// agree, and those whose fixed elements disagree, keep their modes.
    #[test]
    fn one_register_takes_the_mode_of_its_fixed_elements() {
        let element = |name: &str, row, col, cols, comp, interp, usage| SigElement {
            name: name.into(),
            index: 0,
            kind: if name == "SV_IsFrontFace" { 13 } else { 0 },
            comp,
            interp,
            row: Some(row),
            col,
            cols,
            usage,
            half: false,
        };
        let modes = |els: &[SigElement]| els.iter().map(|e| e.interp).collect::<Vec<_>>();
        // VS_Transform_VfxVBuffer_AlphaClip's outputs: TEXCOORD2 (float2) beside TEXCOORD14
        // (uint2) in register 7; register 6 holds two linear floats.
        let mut outputs = vec![
            element("TEXCOORD", 6, 0, 3, 9, 2, 0b111),
            element("TEXCOORD", 6, 3, 1, 9, 2, 0b1),
            element("TEXCOORD", 7, 0, 2, 9, 2, 0b11),
            element("TEXCOORD", 7, 2, 2, 5, 1, 0b11),
        ];
        one_mode_per_register(&mut outputs, false);
        assert_eq!(modes(&outputs), [2, 2, 1, 1]);
        // A pixel program's inputs: an unread float beside the front face; another beside an
        // input read at its vertices (constant), where the float would be linear.
        let mut inputs = vec![
            element("TEXCOORD", 9, 0, 1, 9, 2, 0),
            element("SV_IsFrontFace", 9, 1, 1, 5, 1, 0b1),
            element("NORMAL", 5, 0, 3, 9, 2, 0),
            element("TEXCOORD", 5, 3, 1, 9, 1, 0b1),
        ];
        one_mode_per_register(&mut inputs, true);
        assert_eq!(modes(&inputs), [1, 1, 1, 1]);
        // Two floats a pixel program reads in two modes: no element gives the register one.
        let mut both = vec![
            element("TEXCOORD", 3, 0, 2, 9, 2, 0b11),
            element("TEXCOORD", 3, 2, 2, 9, 3, 0b11),
        ];
        one_mode_per_register(&mut both, true);
        assert_eq!(modes(&both), [2, 3]);
        // As outputs nothing is read: the floats keep their defaults too.
        one_mode_per_register(&mut both, false);
        assert_eq!(modes(&both), [2, 3]);
    }

    #[test]
    fn mesh_vertex_modes_follow_psv_semantics() {
        let element = |index, col, cols, comp, interp| SigElement {
            name: "TEXCOORD".into(),
            index,
            kind: 0,
            comp,
            interp,
            row: Some(9),
            col,
            cols,
            usage: (1 << cols) - 1,
            half: false,
        };
        let integer = element(8, 0, 3, 5, 1);
        let flat_float = element(11, 3, 1, 9, 1);
        let guest_outputs = [flat_float.clone(), integer.clone()];
        let guest_psv = psv(&Psv {
            stage: 13,
            flag: false,
            inputs: &[],
            outputs: &guest_outputs,
            prims: &[],
            mesh: None,
            geometry: None,
            bindings: &[Binding {
                kind: 2,
                space: 0,
                register: 0,
                dxil_kind: 13,
            }],
            threads: [32, 1, 1],
            wave_size: None,
        });
        let mut outputs = [
            integer,
            SigElement {
                interp: 2,
                ..flat_float
            },
        ];
        let osg = signature(&outputs, false);
        assert_eq!(
            mesh_output_modes(&guest_psv, &osg, &outputs),
            Some(vec![1, 1])
        );
        assert_eq!(
            mesh_output_modes(&guest_psv[..guest_psv.len() - 1], &osg, &outputs),
            None
        );

        outputs[1].index = 12;
        assert_eq!(mesh_output_modes(&guest_psv, &osg, &outputs), None);
        outputs[1].index = 11;
        outputs[0].interp = 2;
        assert_eq!(mesh_output_modes(&guest_psv, &osg, &outputs), None);
    }

    /// Run with `cargo test --release -p victory-shader container_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark"]
    fn container_throughput() {
        use crate::dxbc::{Container, FourCC};
        use crate::dxil::{DxilProgram, build_standard};
        use std::{hint::black_box, time::Instant};

        fn median_ns(iterations: u32, mut op: impl FnMut()) -> f64 {
            let mut samples = [0.; 9];
            for sample in &mut samples {
                let start = Instant::now();
                for _ in 0..iterations {
                    op();
                }
                *sample = start.elapsed().as_secs_f64() * 1e9 / f64::from(iterations);
            }
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        }

        let mut bitcode: Vec<_> = (0..8192).map(|i| i as u8).collect();
        bitcode[..4].copy_from_slice(b"BC\xc0\xde");
        let program = build_standard(0, (6, 6), 0x106, &bitcode);
        let parts: &[(&[u8; 4], &[u8])] = &[
            (b"SFI0", &[0; 8]),
            (b"ISG1", &[0; 112]),
            (b"OSG1", &[0; 48]),
            (b"PSV0", &[0; 176]),
            (b"DXIL", &program),
        ];
        let container = build(parts);
        let parse = median_ns(500_000, || {
            let c = Container::parse(black_box(&container)).unwrap();
            let p = c.part(FourCC::DXIL).unwrap();
            black_box(DxilProgram::parse(p.data).unwrap());
        });
        let emit = median_ns(4096, || {
            black_box(build(black_box(parts)));
        });
        eprintln!(
            "{} bytes, digest {}: parse + find DXIL {parse:.2} ns/op, build + hash {emit:.2} ns/op",
            container.len(),
            hex(container[4..20].try_into().unwrap())
        );
    }

    #[test]
    fn odd_parts_match_wire_bytes() {
        let bytes = build(&[(b"TEST", b"a"), (b"DATA", b"bb")]);
        let expected = b"DXBC\xc2\xdb\x46\xa8\xd7\x6c\x02\xda\x6f\xb8\x1b\x41\x3c\xe3\x22\x8e\
            \x01\0\0\0\x3b\0\0\0\x02\0\0\0\x28\0\0\0\x31\0\0\0\
            TEST\x01\0\0\0aDATA\x02\0\0\0bb";
        assert_eq!(&bytes, expected);
        let parsed = crate::dxbc::Container::parse(&bytes).unwrap();
        assert_eq!(parsed.parts[1].offset, 49);
        assert_eq!(parsed.parts[1].data, b"bb");
    }

    #[test]
    fn dxbc_hash_padding_boundaries() {
        // Digests from the previous implementation, for ascending input bytes.
        let bytes: Vec<u8> = (0..8192).map(|i| i as u8).collect();
        for (len, expected) in [
            (0, "140d60f6b775e2ba4e4abed401b2e9a1"),
            (55, "842e55534ba93daa93e94bd5af9b0c03"),
            (56, "00f9cc964ff2ec81959d4a3f092ce63f"),
            (63, "4400d428a41d0e75dc98b936693f7bfb"),
            (64, "f42eb06ca921c878e435e3b8d4f92b13"),
            (65, "9ce1a374e3f259448f5777113a2edb57"),
            (119, "ba3a67ecd4641ce4ed08f291896e1bc6"),
            (120, "882fce4f4c95194392e7adefd53fe343"),
            (128, "7f0468e3c7a668781a3d62e31d49bf72"),
            (8192, "31987841a6cceb14d13ec19712590074"),
        ] {
            assert_eq!(hex(hash(&bytes[..len])), expected, "length {len}");
        }
    }

    fn hex(digest: [u8; 16]) -> String {
        digest.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn md5_core_matches_the_standard() {
        for (input, expected) in [
            ("abc", "900150983cd24fb0d6963f7d28e17f72"),
            (
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
                "d174ab98d277d9f5a5611c2c9f419d9f",
            ),
        ] {
            let mut padded = input.as_bytes().to_vec();
            padded.push(0x80);
            padded.resize((padded.len() + 8).next_multiple_of(64) - 8, 0);
            padded.extend((input.len() as u64 * 8).to_le_bytes());
            let mut state = [0x6745_2301u32, 0xefcd_ab89, 0x98ba_dcfe, 0x1032_5476];
            for block in padded.as_chunks::<64>().0 {
                md5_block(&mut state, block);
            }
            let mut bytes = [0; 16];
            for (dst, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(state) {
                *dst = word.to_le_bytes();
            }
            assert_eq!(hex(bytes), expected);
        }
    }
}
