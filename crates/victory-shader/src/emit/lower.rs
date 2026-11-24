//! IR to DXIL, one machine instruction at a time (decision 0003).
//!
//! Every register dword is an `i32` (SCC an `i1`); float operations bitcast
//! their operands. Lane masks exist only as the per-lane booleans of
//! structurisation; branches and guards are built from them. A value the
//! emitter cannot express is `Missing` with a reason, and only fails the
//! program when something with an effect (an export, a branch) needs it.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::bitcode::{Attr, Bin, Cast, Term, V, pred};
use super::container::SigElement;
use super::shader::{Shader, flags};
use super::{TableView, UserData};
use crate::ShaderError;
use crate::gfx10::{Dpp, Fields, NULL, Operand};
use crate::io::{Barycentric, Copied, Export, Input, Interp, Io, Ring, Target};
use crate::ir::{EXEC, Exit, Inst, Program, Reg, Value};
use crate::native::{HwStage, NativeProgram, ResourceType, RootType, StageIo, Usage, UsedResource};
use crate::resource::{Address, CODE, RelOffset, Resources, Role, Site, Source};
use crate::structure::{Atom, Bool, BoolSet, LaneExit, Structured};

type Why = &'static str;

/// How an interpolation evaluates an input channel.
#[derive(Debug, Clone, Copy)]
enum Eval {
    /// Loaded with this DXIL interpolation mode (1: constant, P0).
    Mode(u8),
    /// From the vertex values of a constant element: P0 when `p0`, plus
    /// i (P1 - P0), plus j (P2 - P0), weights as floats.
    Weights {
        p0: bool,
        i: Option<V>,
        j: Option<V>,
    },
}

/// A heap handle's block, load, first dword, sampler heap and properties.
type HandleKey = (usize, Site, u8, bool, (u32, u32));

/// Element stride of a table-bound structured buffer, when the host descriptor
/// carries `StructureByteStride`. Flat byte-offset accesses must annotate the
/// handle as structured and split the byte address into `(index, offset)` so
/// drivers that check offsets against the element stride do not drop elements
/// past the first. Root descriptors have no host stride and stay raw.
fn structured_stride(used: Option<&UsedResource>) -> Option<u32> {
    let u = used?;
    (u.root == RootType::Table && u.stride >= 4 && u.stride.is_multiple_of(4)).then_some(u.stride)
}
#[derive(Debug, Clone, Copy)]
enum Val {
    I32(V),
    Bit(V),
    Missing(Why),
}

pub(crate) struct Lowered {
    pub dxil: Vec<u8>,
    /// Bit n: immediate sampler n is used for comparisons.
    pub cmp_samplers: u64,
    pub root_cbvs: Vec<u8>,
    pub root_srvs: Vec<u8>,
    pub stride_table: bool,
    pub user_data: Vec<UserData>,
    pub samplers: Vec<[u32; 4]>,
    /// Heap handles made with a non-uniform index.
    #[cfg(test)]
    pub nonuniform_handles: u32,
    /// Each heap handle's resource properties, in creation order.
    #[cfg(test)]
    pub heap_props: Vec<(u32, u32)>,
    /// Whether the program uses wave operations.
    #[cfg(test)]
    pub wave_ops: bool,
    pub table_views: Vec<TableView>,
    /// Test builds: typed loads assembled from a dword-aligned pair.
    #[cfg(test)]
    pub unaligned_typed_loads: u32,
}

pub(crate) struct Parts<'a> {
    pub native: &'a NativeProgram,
    pub p: &'a Program,
    pub st: &'a Structured,
    pub io: &'a Io,
    pub res: &'a Resources,
    pub inputs: Vec<SigElement>,
    pub outputs: Vec<SigElement>,
    pub prims: Vec<SigElement>,
    /// Mesh: what the group id arrives multiplied by.
    pub mesh_stride: u32,
}

pub(crate) fn lower(parts: &Parts) -> Result<Lowered, ShaderError> {
    Lower::new(parts).run()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReadDep {
    Value(Value),
    Edge {
        from: usize,
        to: usize,
        value: Value,
    },
}

struct ValueDeps {
    block: usize,
    guard: Option<Bool>,
    invert_guard: bool,
    ignore_why: Option<Why>,
    deps: Vec<ReadDep>,
}

struct OldDep {
    value: Option<Value>,
    direct_why: Option<Why>,
    invert_guard: bool,
    ignore_why: Option<Why>,
}

struct SinkRead {
    block: usize,
    guard: Option<Bool>,
    offset: usize,
    deps: Vec<ReadDep>,
}

const TWO_PI: f32 = std::f32::consts::TAU;
/// `s_sendmsg` id of a mesh or NGG stage's output allocation request.
const GS_ALLOC_REQ: u16 = 9;
/// `s_sendmsg` ids of a legacy GS: GS_OP_* in bits 4-5 (1 cut, 2 emit, 3
/// both), the stream in bits 8-9; then the end of the primitive's work.
const MSG_GS: u16 = 2;
const MSG_GS_DONE: u16 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Numeric {
    Unorm,
    Snorm,
    Uscaled,
    Sscaled,
    Uint,
    Sint,
    Float,
}

/// A GFX10 buffer format (the unified format of typed buffer loads): the
/// channel widths from the lowest bits up, and how the channels convert.
/// Formats are numbered in families (`8`, `16`, `8_8`, ...: AMD names
/// list channels from the highest bits), each with its numeric formats in
/// the order unorm, snorm, uscaled, sscaled, uint, sint, float.
fn buffer_format(format: u8) -> Option<(&'static [u32], Numeric)> {
    use Numeric::*;
    const NORM: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint];
    const ALL: &[Numeric] = &[Unorm, Snorm, Uscaled, Sscaled, Uint, Sint, Float];
    const WIDE: &[Numeric] = &[Uint, Sint, Float];
    // Numbers and order from LLVM's `UfmtGFX10` (`SIDefines.h`); the GFX10
    // buffer formats have no 5_9_9_9. Channel order from Mesa
    // (`ac_formats.c`): R11G11B10_FLOAT is 10_11_11, and 10_10_10_2 has
    // its 2-bit channel in the lowest bits.
    const FAMILIES: [(&[u32], &[Numeric]); 14] = [
        (&[8], NORM),
        (&[16], ALL),
        (&[8, 8], NORM),
        (&[32], WIDE),
        (&[16, 16], ALL),
        (&[11, 11, 10], ALL),     // 10_11_11
        (&[10, 11, 11], ALL),     // 11_11_10
        (&[2, 10, 10, 10], NORM), // 10_10_10_2
        (&[10, 10, 10, 2], NORM), // 2_10_10_10
        (&[8, 8, 8, 8], NORM),
        (&[32, 32], WIDE),
        (&[16, 16, 16, 16], ALL),
        (&[32, 32, 32], WIDE),
        (&[32, 32, 32, 32], WIDE),
    ];
    let mut n = usize::from(format.checked_sub(1)?);
    for (bits, numerics) in FAMILIES {
        if n < numerics.len() {
            return Some((bits, numerics[n]));
        }
        n -= numerics.len();
    }
    None
}

/// How far a float channel of `width` bits (16, or the unsigned 11- and
/// 10-bit floats of the packed formats) shifts up to be a half:
/// the small floats are a half's exponent (5 bits, biased 15) and the top
/// of its mantissa, without the sign, so `legacyF16ToF32` widens them
/// exactly, denormals, infinities and NaNs included.
fn half_shift(width: u32) -> u32 {
    debug_assert!(matches!(width, 10 | 11 | 16), "{width}-bit float");
    if width == 16 { 0 } else { 15 - width }
}

/// The four words of an immediate sampler.
fn sampler_words(words: &[u32]) -> Result<[u32; 4], Why> {
    words
        .get(..4)
        .and_then(|w| w.try_into().ok())
        .ok_or("a sampler constant")
}

struct Lower<'a> {
    s: Shader,
    x: &'a Parts<'a>,
    lane_fail: HashMap<Value, Why>,
    vals: HashMap<Value, Val>,
    floats: HashMap<Value, V>,
    /// DXIL block per IR block (unreachable: none).
    block: Vec<Option<usize>>,
    cur: usize,
    bools: HashMap<(usize, Bool), V>,
    bool_deps: HashMap<(usize, Bool), Vec<ReadDep>>,
    lane_bits: HashMap<Value, V>,
    /// IR phi location by value.
    phi_at: HashMap<Value, (usize, usize)>,
    mask_phis: HashMap<Value, V>,
    pending_masks: Vec<(V, usize, usize)>,
    data_phis: Vec<(V, usize, usize)>,
    taint_candidates: HashSet<Value>,
    cur_reads: Vec<ReadDep>,
    cur_old_deps: Vec<OldDep>,
    cur_edge: Option<(usize, usize)>,
    deps: HashMap<Value, Vec<ValueDeps>>,
    deps_order: Vec<Value>,
    direct_missing: Vec<(Value, Why, usize, Option<Bool>, bool)>,
    sinks: Vec<SinkRead>,
    /// Relative-offset DXIL phis for address phis, keyed by `(phi_value, base_value)`.
    addr_phis: HashMap<(Value, Value), V>,
    /// Address phis waiting for incoming values in `fill_phis`: `(phi, ir_block, phi_idx, base)`.
    pending_addr_phis: Vec<(V, usize, usize, Value)>,
    /// Defining instruction `(block, inst_index)` for each non-phi, non-input IR value.
    def_site: HashMap<Value, Site>,
    params: HashMap<Value, (usize, u32)>,
    user_data: Option<V>,
    user_reads: BTreeMap<u32, UserData>,
    table_views: BTreeMap<(u8, u32), (u32, u32)>,
    samplers: Vec<[u32; 4]>,
    descriptors: HashMap<(Site, bool), &'a Source>,
    scalar: HashMap<Site, Address>,
    /// Tracked descriptor addresses stored to or modified atomically.
    written_uavs: HashSet<(usize, u32, Option<Value>)>,
    /// Tracked descriptor addresses accessed with `glc` and never sampled.
    glc_uavs: HashSet<(usize, u32, Option<Value>)>,
    exports: HashMap<Site, &'a Export>,
    interps: HashMap<Site, &'a Interp>,
    handles: HashMap<HandleKey, V>,
    handle_deps: HashMap<HandleKey, Vec<ReadDep>>,
    /// Runtime stride table entries read in a block, by the descriptor's
    /// load: the raw slot, the stride and the bias.
    entries: HashMap<(usize, Site, u8), [V; 3]>,
    /// Blocks splitting the edge from one IR block to another.
    edges: HashMap<(usize, usize), usize>,
    /// Per IR block: a `done` export is behind every lane entering it.
    exported: Vec<bool>,
    /// Reverse-postorder index per IR block.
    rpo: Vec<usize>,
    /// DXIL block each IR block ends in (guarded stores split blocks).
    tail: Vec<Option<usize>>,
    /// The program writes UAVs: barriers fence them too.
    writes_uavs: bool,
    /// src0 of the current instruction as DPP fetched it.
    src0: Option<V>,
    /// Compute code that reads other lanes: thread ids follow the host
    /// lanes, so lane n is console lane n.
    console_lanes: bool,
    /// Mesh: the thread index in the group (made in the entry block).
    flat: Option<V>,
    /// Mesh: IR blocks whose `s_barrier` is left out.
    no_barrier: HashSet<usize>,
    /// Vertex and mesh outputs holding 16-bit values: row, whether two per
    /// dword, and the row packed after it.
    half_outputs: HashMap<u8, (bool, Option<u8>)>,
    /// How lane masks become data.
    mask_data: MaskData,
    /// The instruction being lowered may read lane masks as data.
    mask_ok: bool,
    /// Ballots of masks read as data, by low dword: (low, high).
    ballots: HashMap<Value, (V, V)>,
    ballot_deps: HashMap<Value, Vec<ReadDep>>,
    /// Masks whose ballot is being built: a mask whose lane bit needs its
    /// own value as a number (an opaque mask keyed by a dword it shares
    /// with another) would otherwise recurse until the stack overflows.
    balloting: HashSet<Value>,
    /// Block writing each value.
    def_block: HashMap<Value, usize>,
    /// IR block being lowered.
    cur_ir: usize,
    /// Whole-wave sections by block and start index.
    sections: HashMap<usize, HashMap<usize, super::wave::Section>>,
    /// Values written in a section that code outside it reads.
    section_out: HashSet<Value>,
    /// The program's constant table (the code from the first byte read
    /// through `s_getpc_b64` on): global, first byte, dwords.
    code_table: Option<(V, u32, u32)>,
    /// Loop headers' iteration counters (see [`LOOP_LIMIT`]): the phi by
    /// IR block, and its incoming (value, DXIL block) pairs.
    counters: HashMap<usize, (V, Vec<(V, usize)>)>,
    /// The block a loop stopped by [`LOOP_LIMIT`] leaves the program from.
    abort: Option<usize>,
    /// The one block that returns: every end of the program branches to it,
    /// so the function has a single exit as DXC emits it (dxil-spirv, under
    /// Proton, cannot structure a loop left by both a break and a return).
    ret: Option<usize>,
    /// ES and GS: ring accesses by site.
    rings: HashMap<Site, Ring>,
    /// GS: the vertices it emits (the GSVS ring), then the count emitted:
    /// a per-thread array.
    gsvs: Option<V>,
    /// Cube image sites (`dim` 3) whose descriptor is a `TextureCubeArray`.
    cube_arrays: HashSet<Site>,
}

/// Iterations after which a loop is left (per entry into it), unless
/// [`super::loop_limit`] says otherwise. The console runs a loop as long as
/// its data says; data the host has not mirrored yet (a null view reads
/// zeros) can make that four billion iterations, which hangs the host GPU
/// until the driver resets it. A lane that reaches the limit leaves the
/// program as killed (a pixel discards).
// shortcut: a genuine loop of more iterations than this is cut short too.
pub(super) const LOOP_LIMIT: u32 = 1 << 20;

/// Lane masks read as numbers (`s_bcnt1`, `s_ff1`, `v_mbcnt`, scalar
/// compares, bit tests) are ballots of the lanes' booleans, taken where the
/// mask is written: structurisation's `active` masks hold only lanes that
/// reach that block, so the ballot misses none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaskData {
    /// Compute: host lane n is console lane n (the program declares the
    /// console's wave size), so ballots are the console's masks bit for bit
    /// and any use is exact.
    Console,
    /// Pixel and vertex programs: ballots over the host's wave, whatever its
    /// size. Only uses that mean the same on any wave are allowed: counts,
    /// the first lane, the count below a lane, equality (the HLSL wave
    /// operations the console compiler expands). Lane numbers stay host
    /// lanes throughout (`v_mbcnt`, `v_readlane` at a computed lane).
    Host,
    /// Mesh programs: their lanes and waves (LDS per console wave) are the
    /// console's but no host wave size can be declared.
    None,
}

/// What a value holding a lane mask read as a number stops with.
const DESCRIPTOR_OTHER: Why = "a descriptor's address or format read as data";
const LANE_MASK: Why = "a lane mask read as data";

/// Instructions that read a lane mask (not EXEC implicitly, not a lane's
/// bit) as a number.
fn reads_mask_as_data(st: &Structured, i: &Inst) -> bool {
    let is_mask = |v: Value| st.masks.contains_key(&v) || st.highs.contains_key(&v);
    let ops: Vec<&Operand> = match &i.fields {
        // A mask operation (its result a mask) is no data use.
        Fields::Sop { .. }
            if i.writes
                .iter()
                .any(|&(r, w)| matches!(r, Reg::S(_)) && is_mask(w)) =>
        {
            return false;
        }
        Fields::Sop { src, .. } => src.iter().collect(),
        Fields::Valu(v)
            if !matches!(i.name(), "v_cndmask_b32") && !i.name().contains("_co_ci_") =>
        {
            v.src.iter().collect()
        }
        _ => return false,
    };
    ops.into_iter().any(|op| match *op {
        Operand::Scalar { code, .. } if code != NULL => i.read(Reg::S(code)).is_some_and(is_mask),
        _ => false,
    })
}

/// Where half `h` (dword h / 2, the high half when odd) of a parameter
/// packing 16-bit rows `a` and `b` two per dword comes from: a row and
/// channel of `elements`, none for padding. Channel c of `a` is half c;
/// `b` starts at the dword after `a`'s last channel.
fn packed_half(elements: &[SigElement], a: u8, b: Option<u8>, h: u8) -> Option<(u8, u8)> {
    let extent = |r: u8| {
        elements
            .iter()
            .filter(|e| e.row == Some(u32::from(r)))
            .map(|e| e.col + e.cols)
            .max()
    };
    let na = extent(a)?;
    let a_halves = na.div_ceil(2) * 2;
    if h < a_halves {
        return (h < na).then_some((a, h));
    }
    let b = b?;
    let c = h - a_halves;
    (c < extent(b)?).then_some((b, c))
}

/// Vertex or mesh outputs of 16-bit values (see [`Lower::half_outputs`]):
/// a packed row's partner is the next 16-bit row of the signature that no
/// output entry names, taken in row order.
fn half_outputs(io: &StageIo, outputs: &[SigElement]) -> HashMap<u8, (bool, Option<u8>)> {
    let entries = match io {
        StageIo::Vs { outputs, .. } => outputs,
        StageIo::Ms(ms) => &ms.outputs,
        _ => return HashMap::new(),
    };
    let per_vertex = || entries.iter().filter(|o| o.param & 0x80 == 0);
    let mut free: Vec<u8> = outputs
        .iter()
        .filter(|e| e.half && e.kind == 0)
        .filter_map(|e| u8::try_from(e.row?).ok())
        .filter(|&r| !per_vertex().any(|o| o.row == r))
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut halves: Vec<_> = per_vertex().filter(|o| o.fp16).collect();
    halves.sort_by_key(|o| o.row);
    halves
        .into_iter()
        .map(|o| {
            let b = o
                .pack2
                .then(|| free.iter().position(|&r| r > o.row))
                .flatten();
            (o.row, (o.pack2, b.map(|i| free.remove(i))))
        })
        .collect()
}

/// Pixel inputs packing two 16-bit rows per dword whose second row the
/// metadata leaves out (`row2` = `row`): as for outputs
/// ([`half_outputs`]), the next 16-bit row of the signature that no input
/// entry names, taken in row order (rows the metadata does give are taken
/// first).
fn half_input_partners(io: &StageIo, inputs: &[SigElement]) -> HashMap<u8, u8> {
    let StageIo::Ps {
        inputs: entries, ..
    } = io
    else {
        return HashMap::new();
    };
    let packed = || entries.iter().filter(|i| i.fp16 && i.pack2);
    let mut free: Vec<u8> = inputs
        .iter()
        .filter(|e| e.half && e.kind == 0)
        .filter_map(|e| u8::try_from(e.row?).ok())
        .filter(|&r| !entries.iter().any(|i| i.row == r))
        .filter(|&r| !packed().any(|i| i.row2 != i.row && i.row2 == r))
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut missing: Vec<_> = packed().filter(|i| i.row2 == i.row).collect();
    missing.sort_by_key(|i| i.row);
    let mut partners = HashMap::new();
    for i in missing {
        if let Some(k) = free.iter().position(|&r| r > i.row) {
            partners.insert(i.row, free.remove(k));
        }
    }
    partners
}

fn unsupported(offset: usize, what: Why) -> ShaderError {
    ShaderError::Unsupported { offset, what }
}

/// Deterministic preference when the same table slot is annotated with
/// multiple DXIL `ResourceProperties`: UAV annotations win over non-UAV ones.
fn prefer_props(cur: (u32, u32), new: (u32, u32)) -> (u32, u32) {
    let key = |p: (u32, u32)| (p.0 & 0x1000 != 0, p);
    if key(new) > key(cur) { new } else { cur }
}

/// Cube image sites (`dim` 3) that view a `TextureCubeArray` rather than a
/// `TextureCube`. GFX10 MIMG has no array bit for cubes: the compiler encodes
/// the face operand as `8 * slice + face`, and only `TextureCubeArray` reads
/// channel 2 (the face count) of `image_get_resinfo`. Any cube site sharing a
/// descriptor with an array site takes the array kind as well.
fn cube_arrays(p: &Program, res: &Resources) -> HashSet<Site> {
    let mut defs = HashMap::new();
    let mut phis = HashMap::new();
    let mut live = HashSet::new();
    for b in &p.blocks {
        for phi in &b.phis {
            phis.insert(phi.value, phi.args.as_slice());
        }
        for inst in &b.insts {
            for &(_, r) in &inst.reads {
                live.insert(r);
            }
            for &(_, w) in &inst.writes {
                defs.insert(w, inst);
            }
        }
    }
    let mut work: Vec<Value> = live.iter().copied().collect();
    while let Some(v) = work.pop() {
        if let Some(args) = phis.get(&v) {
            for &a in *args {
                if phis.contains_key(&a) && live.insert(a) {
                    work.push(a);
                }
            }
        }
    }

    struct Ctx<'a> {
        defs: HashMap<Value, &'a Inst>,
        phis: HashMap<Value, &'a [Value]>,
    }
    impl Ctx<'_> {
        fn reg_val(inst: &Inst, op: &Operand) -> Option<Value> {
            match *op {
                Operand::Vector { reg, .. } => inst.read(Reg::V(reg)),
                Operand::Scalar { code, .. } if code != NULL => inst.read(Reg::S(code)),
                _ => None,
            }
        }

        fn op_bits(&self, inst: &Inst, op: &Operand, depth: u8) -> Option<u32> {
            match *op {
                Operand::Literal(x) => Some(x),
                Operand::Const(c) => crate::resource::inline(c),
                Operand::Scalar { code: NULL, .. } => Some(0),
                _ => self.val_bits(Self::reg_val(inst, op)?, depth),
            }
        }

        fn val_bits(&self, v: Value, depth: u8) -> Option<u32> {
            if depth >= 16 {
                return None;
            }
            if let Some(args) = self.phis.get(&v) {
                let mut it = args
                    .iter()
                    .copied()
                    .filter(|&a| a != v)
                    .map(|a| self.val_bits(a, depth + 1));
                let first = it.next()??;
                return it.all(|x| x == Some(first)).then_some(first);
            }
            let inst = *self.defs.get(&v)?;
            match (&inst.fields, inst.name()) {
                (Fields::Sop { imm: Some(imm), .. }, "s_movk_i32") => {
                    Some(i32::from(*imm as i16) as u32)
                }
                (Fields::Sop { src, .. }, "s_mov_b32") => {
                    self.op_bits(inst, src.first()?, depth + 1)
                }
                (Fields::Valu(vf), "v_mov_b32")
                    if vf.sdwa.is_none() && vf.dpp.is_none() && !vf.clamp && vf.omod == 0 =>
                {
                    let mut x = self.op_bits(inst, vf.src.first()?, depth + 1)?;
                    if vf.abs & 1 != 0 {
                        x &= 0x7FFF_FFFF;
                    }
                    if vf.neg & 1 != 0 {
                        x ^= 0x8000_0000;
                    }
                    Some(x)
                }
                _ => None,
            }
        }

        fn fsrc_const(&self, inst: &Inst, k: usize) -> Option<f32> {
            let Fields::Valu(vf) = &inst.fields else {
                return None;
            };
            if vf.sdwa.is_some() || vf.dpp.is_some() {
                return None;
            }
            let mut x = self.op_bits(inst, vf.src.get(k)?, 0)?;
            if vf.abs >> k & 1 != 0 {
                x &= 0x7FFF_FFFF;
            }
            if vf.neg >> k & 1 != 0 {
                x ^= 0x8000_0000;
            }
            Some(f32::from_bits(x))
        }

        fn hsrc_const(&self, inst: &Inst, k: usize) -> Option<u32> {
            let Fields::Valu(vf) = &inst.fields else {
                return None;
            };
            if vf.sdwa.is_some() || vf.dpp.is_some() {
                return None;
            }
            let op = vf.src.get(k)?;
            let mut h = match *op {
                Operand::Const(c @ 240..=248) => [
                    0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
                ][usize::from(c - 240)],
                _ => {
                    let x = self.op_bits(inst, op, 0)?;
                    let sh = if vf.opsel >> k & 1 != 0 { 16 } else { 0 };
                    (x >> sh) & 0xFFFF
                }
            };
            if vf.abs >> k & 1 != 0 {
                h &= 0x7FFF;
            }
            if vf.neg >> k & 1 != 0 {
                h ^= 0x8000;
            }
            Some(h)
        }

        fn is_array_face(
            &self,
            v: Value,
            half: Option<bool>,
            seen: &mut HashSet<(Value, Option<bool>)>,
        ) -> bool {
            if !seen.insert((v, half)) {
                return false;
            }
            if let Some(w) = self.val_bits(v, 0) {
                return match half {
                    None => {
                        let f = f32::from_bits(w);
                        f.is_finite() && f >= 8.0
                    }
                    Some(hi) => {
                        let h = (w >> if hi { 16 } else { 0 }) & 0xFFFF;
                        (0x4800..=0x7BFF).contains(&h)
                    }
                };
            }
            if let Some(args) = self.phis.get(&v) {
                return args.iter().any(|&a| self.is_array_face(a, half, seen));
            }
            let Some(inst) = self.defs.get(&v) else {
                return false;
            };
            let Fields::Valu(vf) = &inst.fields else {
                return false;
            };
            if vf.sdwa.is_some() || vf.dpp.is_some() {
                return false;
            }
            let name = inst.name();
            let src_face = |k: usize, h: Option<bool>, seen: &mut HashSet<_>| {
                vf.src
                    .get(k)
                    .and_then(|o| Self::reg_val(inst, o))
                    .is_some_and(|x| self.is_array_face(x, h, seen))
            };
            if name == "v_mov_b32" && !vf.clamp && vf.omod == 0 && vf.abs == 0 && vf.neg == 0 {
                return src_face(0, half, seen);
            }
            if name == "v_cndmask_b32" && vf.abs == 0 && vf.neg == 0 {
                return src_face(0, half, seen) || src_face(1, half, seen);
            }
            if let Some(hi) = half {
                let dst_hi = vf.opsel >> 3 & 1 != 0;
                return match name {
                    "v_cvt_f16_f32" if dst_hi == hi => src_face(0, None, seen),
                    "v_cvt_pkrtz_f16_f32" => src_face(usize::from(hi), None, seen),
                    "v_pack_b32_f16" if vf.abs == 0 && vf.neg == 0 => {
                        let k = usize::from(hi);
                        src_face(k, Some(vf.opsel >> k & 1 != 0), seen)
                    }
                    "v_fma_f16" | "v_mad_f16" | "v_fmac_f16" | "v_mac_f16" | "v_fmamk_f16"
                    | "v_fmaak_f16" | "v_mul_f16"
                        if dst_hi == hi && !vf.clamp =>
                    {
                        [0, 1]
                            .into_iter()
                            .any(|k| self.hsrc_const(inst, k) == Some(0x4800))
                    }
                    "v_add_f16" if dst_hi == hi && !vf.clamp => [0, 1].into_iter().any(|k| {
                        self.hsrc_const(inst, k)
                            .is_some_and(|h| (0x4800..=0x7BFF).contains(&h))
                            || src_face(k, Some(vf.opsel >> k & 1 != 0), seen)
                    }),
                    _ => false,
                };
            }
            let scale = [1.0f32, 2.0, 4.0, 0.5][usize::from(vf.omod & 3)];
            match name {
                "v_cvt_f32_f16" if !vf.clamp && vf.omod == 0 && vf.abs == 0 && vf.neg == 0 => {
                    src_face(0, Some(vf.opsel & 1 != 0), seen)
                }
                "v_mul_f32" | "v_mul_legacy_f32" if !vf.clamp => [0, 1]
                    .into_iter()
                    .any(|k| self.fsrc_const(inst, k).is_some_and(|x| x * scale == 8.0)),
                "v_add_f32" if !vf.clamp && vf.omod == 0 => [0, 1].into_iter().any(|k| {
                    self.fsrc_const(inst, k)
                        .is_some_and(|f| f.is_finite() && f >= 8.0)
                        || (vf.abs >> k & 1 == 0 && vf.neg >> k & 1 == 0 && src_face(k, None, seen))
                }),
                "v_fma_f32" | "v_fmaak_f32" | "v_fmamk_f32" | "v_fmac_f32" | "v_mad_f32"
                | "v_madak_f32" | "v_madmk_f32" | "v_mac_f32" | "v_mad_legacy_f32"
                | "v_mac_legacy_f32"
                    if !vf.clamp =>
                {
                    let addend = if matches!(name, "v_fmac_f32" | "v_mac_f32" | "v_mac_legacy_f32")
                    {
                        vf.dst.as_ref().and_then(|d| Self::reg_val(inst, d))
                    } else if vf.abs & 4 == 0 && vf.neg & 4 == 0 {
                        vf.src.get(2).and_then(|o| Self::reg_val(inst, o))
                    } else {
                        None
                    };
                    [0, 1]
                        .into_iter()
                        .any(|k| self.fsrc_const(inst, k).is_some_and(|x| x * scale == 8.0))
                        || (vf.omod == 0
                            && addend.is_some_and(|a| self.is_array_face(a, None, seen)))
                }
                "v_fma_mix_f32" if !vf.clamp && vf.omod == 0 => [0, 1].into_iter().any(|k| {
                    let Some(op) = vf.src.get(k) else {
                        return false;
                    };
                    let Some(w) = self.op_bits(inst, op, 0) else {
                        return false;
                    };
                    let neg = vf.neg >> k & 1 != 0;
                    let abs = vf.neg_hi >> k & 1 != 0;
                    if vf.opsel_hi >> k & 1 != 0 {
                        let sh = if vf.opsel >> k & 1 != 0 { 16 } else { 0 };
                        let mut h = (w >> sh) & 0xFFFF;
                        if abs {
                            h &= 0x7FFF;
                        }
                        if neg {
                            h ^= 0x8000;
                        }
                        h == 0x4800
                    } else {
                        let mut x = w;
                        if abs {
                            x &= 0x7FFF_FFFF;
                        }
                        if neg {
                            x ^= 0x8000_0000;
                        }
                        f32::from_bits(x) == 8.0
                    }
                }),
                _ => false,
            }
        }
    }

    let ctx = Ctx { defs, phis };
    let scalar: HashMap<Site, Address> = res.scalar.iter().copied().collect();
    let descs: HashMap<Site, &Source> = res
        .descriptors
        .iter()
        .filter(|d| d.role == Role::Resource)
        .map(|d| (d.site, &d.source))
        .collect();
    let mut sites: Vec<(Site, Option<&Source>, bool)> = Vec::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            let Fields::Image {
                data,
                addr,
                dmask,
                dim: 3,
                a16,
                ..
            } = &inst.fields
            else {
                continue;
            };
            let site = (b, i);
            let name = inst.name();
            let mut direct = false;
            if name.starts_with("image_sample") || name.starts_with("image_gather") {
                let (mut whole, mut derivs) = (0usize, 0usize);
                if let Some(rest) = name
                    .strip_prefix("image_sample")
                    .or_else(|| name.strip_prefix("image_gather4"))
                {
                    for token in rest.split('_') {
                        match token {
                            "o" | "b" | "c" => whole += 1,
                            "d" | "cd" => derivs = 2,
                            _ => {}
                        }
                    }
                }
                let mut regs = Vec::new();
                for a in addr {
                    if let Operand::Vector { reg, dwords } = *a {
                        regs.extend((0..dwords).map(|k| reg + k));
                    }
                }
                let (idx, half) = if *a16 {
                    // `image` indexes `regs` by `(first + grads + k) / 2`.
                    let pos = whole + 2 * derivs + 2;
                    (pos / 2, Some(pos % 2 == 1))
                } else {
                    (whole + 2 * derivs + 2, None)
                };
                if let Some(&r) = regs.get(idx)
                    && let Some(fv) = inst.read(Reg::V(r))
                {
                    direct = ctx.is_array_face(fv, half, &mut HashSet::new());
                }
            } else if name == "image_get_resinfo" {
                let m = if *dmask == 0 { 1 } else { *dmask };
                if m & 4 != 0
                    && let Operand::Vector { reg, .. } = *data
                {
                    let k = (m & 3).count_ones() as u8;
                    direct = inst
                        .write(Reg::V(reg + k))
                        .is_some_and(|v| live.contains(&v));
                }
            }
            sites.push((site, descs.get(&site).copied(), direct));
        }
    }

    let leaf_addr = |s: &Source| match s {
        Source::Loaded { load, first } => scalar.get(load).map(|a| {
            (
                false,
                Address {
                    offset: a.offset.wrapping_add(4 * u32::from(*first)),
                    ..*a
                },
            )
        }),
        Source::Fetched { load, first } => scalar.get(load).map(|a| {
            (
                true,
                Address {
                    offset: a.offset.wrapping_add(4 * u32::from(*first)),
                    ..*a
                },
            )
        }),
        _ => None,
    };
    let same = |a: &Source, b: &Source| {
        a.leaves().into_iter().any(|la| {
            b.leaves()
                .into_iter()
                .any(|lb| la == lb || leaf_addr(la).is_some() && leaf_addr(la) == leaf_addr(lb))
        })
    };
    let mut out = HashSet::new();
    for &(site, src, direct) in &sites {
        if direct
            || src.is_some_and(|s| {
                sites
                    .iter()
                    .any(|&(_, os, od)| od && os.is_some_and(|o| same(s, o)))
            })
        {
            out.insert(site);
        }
    }
    out
}

const UAV_PROP: u32 = 0x1000;
const GLOBALLY_COHERENT_PROP: u32 = 0x4000;

fn uav_bits(uav: bool, glc: bool) -> u32 {
    if !uav {
        0
    } else if glc {
        UAV_PROP | GLOBALLY_COHERENT_PROP
    } else {
        UAV_PROP
    }
}

/// The image atomics made by [`Lower::atomic_loop`], in place of a DXIL
/// atomic operation code.
const ATOMIC_LOOP: u32 = u32::MAX - 1;

impl<'a> Lower<'a> {
    fn new(x: &'a Parts<'a>) -> Self {
        let p = x.p;
        let mut phi_at = HashMap::new();
        for (b, block) in p.blocks.iter().enumerate() {
            for (i, phi) in block.phis.iter().enumerate() {
                phi_at.insert(phi.value, (b, i));
            }
        }
        let mut s = Shader::new(x.native.stage == HwStage::Ps);
        s.inputs = x.inputs.clone();
        s.outputs = x.outputs.clone();
        let scalar: HashMap<Site, Address> = x.res.scalar.iter().copied().collect();
        let desc_addr = |load: Site, first: u8| {
            let a = scalar.get(&load)?;
            Some((
                a.param,
                a.offset.wrapping_add(4 * u32::from(first)),
                a.dynamic,
            ))
        };
        let mut written_uavs = HashSet::new();
        let mut glc_uavs = HashSet::new();
        let mut sampled = HashSet::new();
        for d in x
            .res
            .descriptors
            .iter()
            .filter(|d| d.role == Role::Resource)
        {
            let inst = &p.blocks[d.site.0].insts[d.site.1];
            let name = inst.name();
            let is_sample = matches!(
                &inst.fields,
                Fields::Image {
                    sampler: Some(_),
                    ..
                }
            );
            let is_write = name.contains("store") || name.contains("atomic");
            let is_glc = match &inst.fields {
                Fields::Buffer {
                    cache,
                    format: None,
                    ..
                } => cache.glc,
                Fields::Image {
                    cache,
                    sampler: None,
                    ..
                } => cache.glc && (is_write || name == "image_load"),
                Fields::Smem { cache, .. } => cache.glc && name.starts_with("s_buffer_load_"),
                _ => false,
            };
            for leaf in d.source.leaves() {
                if let &(Source::Loaded { load, first } | Source::Fetched { load, first }) = leaf
                    && let Some(addr) = desc_addr(load, first)
                {
                    if is_sample {
                        sampled.insert(addr);
                    }
                    if is_write {
                        written_uavs.insert(addr);
                    } else if is_glc {
                        glc_uavs.insert(addr);
                    }
                }
            }
        }
        glc_uavs.retain(|a| !sampled.contains(a));
        let insts = || p.blocks.iter().flat_map(|b| &b.insts);
        let writes_uavs = insts().any(|i| {
            matches!(i.fields, Fields::Image { .. } | Fields::Buffer { .. })
                && (i.name().contains("store") || i.name().contains("atomic"))
        }) || (x.native.stage == HwStage::Cs
            && (!written_uavs.is_empty() || !glc_uavs.is_empty()));
        let mut console_lanes = false;
        if let StageIo::Cs(cs) = x.native.io {
            s.threads = Some(cs.threads);
            s.lds_dwords = cs.lds_bytes() / 4;
            // Code reading other lanes needs the console's wave size and
            // lane order; so do lane masks read as numbers, and LDS shared
            // within one wave, where the console compiler leaves out barriers.
            console_lanes = insts().any(|i| {
                let n = i.name();
                n.starts_with("ds_swizzle")
                    || n.contains("permute")
                    || n.starts_with("v_mbcnt")
                    || matches!(n, "v_readlane_b32" | "v_writelane_b32")
                    || matches!(&i.fields, Fields::Valu(v) if v.dpp.is_some())
                    || reads_mask_as_data(x.st, i)
            }) || {
                // Masks held as numbers or constant lane sets: the lane
                // tests its own bit by its console lane.
                let whens = x.st.exits.iter().filter_map(|e| match *e {
                    LaneExit::Branch { when, .. } => Some(when),
                    _ => None,
                });
                let mut all: Vec<Bool> =
                    x.st.guards
                        .iter()
                        .flatten()
                        .flatten()
                        .copied()
                        .chain(whens)
                        .collect();
                all.sort_unstable();
                all.dedup();
                all.into_iter().any(|g| {
                    x.st.bools
                        .support(g)
                        .iter()
                        .any(|a| matches!(a, Atom::Opaque(_) | Atom::Constant(_)))
                }) || insts().any(|i| {
                    matches!(
                        i.name(),
                        "v_cndmask_b32"
                            | "v_add_co_ci_u32"
                            | "v_sub_co_ci_u32"
                            | "v_subrev_co_ci_u32"
                    ) && match &i.fields {
                        Fields::Valu(v) => match v.src.get(2) {
                            Some(&Operand::Scalar { code, .. }) => i
                                .read(Reg::S(code))
                                .is_none_or(|m| !x.st.masks.contains_key(&m)),
                            _ => false,
                        },
                        _ => false,
                    }
                })
            };
            // So does an address read from the first lane.
            console_lanes |= x.res.scalar.iter().any(|(_, a)| a.first_lane.is_some());
            let wave = if x.native.wave64 { 64 } else { 32 };
            let one_wave = cs.threads.iter().product::<u32>() <= wave;
            if console_lanes || one_wave && s.lds_dwords > 0 {
                s.wave_size = Some(wave);
            }
        }
        let mut no_barrier = HashSet::new();
        // ES and GS halves meet at a barrier on the console (the rings in
        // LDS); the host passes vertices between stages itself.
        if matches!(x.native.stage, HwStage::Es | HwStage::Gs) {
            no_barrier.extend(0..p.blocks.len());
        }
        if let StageIo::Gs(gs) = &x.native.io {
            // shortcut: triangles in, triangle strips out (what the stage
            // data's other fields say is not decoded).
            s.gs = Some(super::shader::Geometry {
                max_vertices: gs.max_vertices,
                input: 3,
                topology: 5,
            });
        }
        if let StageIo::Ms(ms) = &x.native.io {
            s.prims = x.prims.clone();
            s.threads = Some([ms.threads, 1, 1]);
            s.lds_dwords = ms.lds_bytes.div_ceil(4);
            s.mesh = Some(super::shader::Mesh {
                max_vertices: ms.max_vertices,
                max_primitives: ms.max_primitives,
                topology: ms.topology,
            });
            // Barriers order the lanes' LDS accesses; the console's
            // allocation request needs them too, but the host does not. One
            // lane sends that request, and what it dominates (the path for
            // counts out of range) runs in that lane alone, so its barrier
            // would hang the host.
            let order = crate::ir::reverse_postorder(&p.blocks);
            let mut index = vec![usize::MAX; p.blocks.len()];
            for (n, &b) in order.iter().enumerate() {
                index[b] = n;
            }
            let idom = crate::ir::idoms(&p.blocks, &order, &index);
            let alloc: Vec<usize> = (0..p.blocks.len())
                .filter(|&b| {
                    p.blocks[b].insts.iter().any(|i| {
                        i.name() == "s_sendmsg"
                            && matches!(i.fields, Fields::Sopp { imm } if imm & 0xF == GS_ALLOC_REQ)
                    })
                })
                .collect();
            for &b in &order {
                let mut d = b;
                let dominated = loop {
                    if alloc.contains(&d) {
                        break true;
                    }
                    if idom[d] == d || idom[d] == usize::MAX {
                        break false;
                    }
                    d = idom[d];
                };
                if ms.lds_bytes == 0 || dominated {
                    no_barrier.insert(b);
                }
            }
        }
        Lower {
            s,
            x,
            lane_fail: HashMap::new(),
            vals: HashMap::new(),
            floats: HashMap::new(),
            block: vec![None; p.blocks.len()],
            cur: 0,
            bools: HashMap::new(),
            bool_deps: HashMap::new(),
            lane_bits: HashMap::new(),
            phi_at,
            mask_phis: HashMap::new(),
            pending_masks: Vec::new(),
            data_phis: Vec::new(),
            taint_candidates: HashSet::new(),
            cur_reads: Vec::new(),
            cur_old_deps: Vec::new(),
            cur_edge: None,
            deps: HashMap::new(),
            deps_order: Vec::new(),
            direct_missing: Vec::new(),
            sinks: Vec::new(),
            addr_phis: HashMap::new(),
            pending_addr_phis: Vec::new(),
            def_site: p
                .blocks
                .iter()
                .enumerate()
                .flat_map(|(b, block)| {
                    block
                        .insts
                        .iter()
                        .enumerate()
                        .flat_map(move |(i, inst)| inst.writes.iter().map(move |w| (w.1, (b, i))))
                })
                .collect(),
            params: x.res.params.iter().map(|&(v, p, d)| (v, (p, d))).collect(),
            user_data: None,
            user_reads: BTreeMap::new(),
            table_views: BTreeMap::new(),
            samplers: Vec::new(),
            descriptors: x
                .res
                .descriptors
                .iter()
                .map(|d| ((d.site, d.role == Role::Sampler), &d.source))
                .collect(),
            scalar,
            written_uavs,
            glc_uavs,
            exports: x.io.exports.iter().map(|e| (e.site, e)).collect(),
            interps: x.io.interps.iter().map(|(s, i)| (*s, i)).collect(),
            handles: HashMap::new(),
            handle_deps: HashMap::new(),
            entries: HashMap::new(),
            edges: HashMap::new(),
            exported: vec![false; p.blocks.len()],
            rpo: vec![usize::MAX; p.blocks.len()],
            tail: vec![None; p.blocks.len()],
            writes_uavs,
            src0: None,
            console_lanes,
            flat: None,
            no_barrier,
            half_outputs: half_outputs(&x.native.io, &x.outputs),
            mask_data: match x.native.stage {
                HwStage::Cs => MaskData::Console,
                HwStage::Ps | HwStage::Vs => MaskData::Host,
                _ => MaskData::None,
            },
            mask_ok: false,
            ballots: HashMap::new(),
            ballot_deps: HashMap::new(),
            balloting: HashSet::new(),
            def_block: p
                .blocks
                .iter()
                .enumerate()
                .flat_map(|(b, block)| {
                    block
                        .insts
                        .iter()
                        .flat_map(move |i| i.writes.iter().map(move |w| (w.1, b)))
                        .chain(block.phis.iter().map(move |phi| (phi.value, b)))
                })
                .collect(),
            cur_ir: 0,
            sections: HashMap::new(),
            section_out: HashSet::new(),
            code_table: None,
            counters: HashMap::new(),
            abort: None,
            ret: None,
            rings: x.io.rings.iter().copied().collect(),
            gsvs: None,
            cube_arrays: cube_arrays(p, x.res),
        }
    }

    /// The dword of the program's code at byte `byte` (a constant table
    /// the compiler appends to the code), as an internal constant array.
    // shortcut: an index outside the table reads its last dword (the
    // console would read whatever memory follows the code).
    fn code_dword(&mut self, byte: V) -> Result<V, Why> {
        let (g, base, len) = match self.code_table {
            Some(t) => t,
            None => {
                let first = self
                    .x
                    .res
                    .scalar
                    .iter()
                    .map(|s| s.1)
                    .chain(self.x.res.descriptors.iter().flat_map(|d| {
                        d.source.leaves().into_iter().filter_map(|l| match l {
                            Source::Built { address, .. } => Some(*address),
                            _ => None,
                        })
                    }))
                    .filter(|a| a.param == CODE)
                    .map(|a| a.offset & !3)
                    .min()
                    .ok_or("a code table")?;
                let code = &self.x.native.code;
                let words: Vec<u32> = code
                    .get(first as usize..)
                    .ok_or("a code table past the code")?
                    .chunks(4)
                    .map(|c| {
                        let mut w = [0u8; 4];
                        w[..c.len()].copy_from_slice(c);
                        u32::from_le_bytes(w)
                    })
                    .collect();
                if words.is_empty() {
                    return Err("a code table past the code");
                }
                let elems: Vec<V> = words.iter().map(|&w| self.s.m.i32(w)).collect();
                let arr = self.s.m.array(words.len() as u32, self.s.t.i32);
                let init = self.s.m.aggregate(arr, &elems);
                let g = self.s.m.constant_global("code", init);
                let t = (g, first, words.len() as u32);
                self.code_table = Some(t);
                t
            }
        };
        let c = self.cur;
        let k = self.s.m.i32(base);
        let rel = self.s.m.bin(c, Bin::Sub, byte, k);
        let two = self.s.m.i32(2);
        let index = self.s.m.bin(c, Bin::LShr, rel, two);
        let last = self.s.m.i32(len - 1);
        let over = self.s.m.cmp(c, pred::IUGT, index, last);
        let index = self.s.m.select(c, over, last, index);
        let at = self.s.m.element(c, g, index);
        Ok(self.s.m.load(c, at))
    }

    /// A global memory load (`global_load_*`): from the program's constant
    /// table, or from a root SRV at the address the title sets.
    fn flat(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Flat { dst, .. } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let rest = name
            .strip_prefix("global_load_")
            .ok_or("a global memory write")?;
        let (n, sub) = match rest {
            "dword" => (1, None),
            "dwordx2" => (2, None),
            "dwordx3" => (3, None),
            "dwordx4" => (4, None),
            "ubyte" => (1, Some((8, false))),
            "sbyte" => (1, Some((8, true))),
            "ushort" => (1, Some((16, false))),
            "sshort" => (1, Some((16, true))),
            _ => return Err(name),
        };
        let a = *self
            .scalar
            .get(&(b, i))
            .ok_or("an untracked global address")?;
        let guard = self.guard(b, i)?;
        let Some(Operand::Vector { reg, .. }) = *dst else {
            return Err("a global load result");
        };
        // (First: a first-lane term may end the block.)
        let off = self.address_offset(a)?;
        let c = self.cur;
        let h = if a.param == CODE {
            None
        } else {
            let src = Source::Built {
                address: a,
                bits: 0,
                words: [0, 0],
            };
            Some(self.handle((b, i), &src, false, (11, 0))?)
        };
        let dword = |l: &mut Self, byte: V| -> Result<V, Why> {
            match h {
                None => l.code_dword(byte),
                Some(h) => {
                    let c = l.cur;
                    let (u, mask, al) = (l.s.m.undef(l.s.t.i32), l.s.m.i8(1), l.s.m.i32(4));
                    let r = l.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        l.s.t.res_ret,
                        139,
                        &[h, byte, u, mask, al],
                    );
                    let x = l.s.m.extract(c, r, 0);
                    Ok(l.as_int(x))
                }
            }
        };
        for k in 0..n {
            let x = match sub {
                None => {
                    let four = self.s.m.i32(4 * k);
                    let at = self.s.m.bin(c, Bin::Add, off, four);
                    dword(self, at)?
                }
                Some((width, signed)) => {
                    // shortcut: the field is taken not to cross a dword.
                    let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
                    let at = self.s.m.bin(c, Bin::And, off, not3);
                    let w = dword(self, at)?;
                    let byte = self.s.m.bin(c, Bin::And, off, three);
                    let sh = self.s.m.bin(c, Bin::Shl, byte, three);
                    let x = self.s.m.bin(c, Bin::LShr, w, sh);
                    let left = self.s.m.i32(32 - width);
                    let x = self.s.m.bin(c, Bin::Shl, x, left);
                    self.s
                        .m
                        .bin(c, if signed { Bin::AShr } else { Bin::LShr }, x, left)
                }
            };
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// Finds the whole-wave sections and the values they leave.
    fn find_sections(&mut self) {
        let p = self.x.p;
        let mut inside: HashMap<Value, (usize, usize, usize)> = HashMap::new();
        for b in 0..p.blocks.len() {
            let found = super::wave::sections(p, b);
            for (&start, sec) in &found {
                for i in start + 1..sec.end {
                    for &(_, w) in &p.blocks[b].insts[i].writes {
                        inside.insert(w, (b, start, sec.end));
                    }
                }
            }
            if !found.is_empty() {
                self.sections.insert(b, found);
            }
        }
        if inside.is_empty() {
            return;
        }
        for (b, block) in p.blocks.iter().enumerate() {
            for phi in &block.phis {
                self.section_out
                    .extend(phi.args.iter().filter(|a| inside.contains_key(a)));
            }
            for (i, inst) in block.insts.iter().enumerate() {
                for &(_, r) in &inst.reads {
                    if let Some(&(sb, start, end)) = inside.get(&r)
                        && !(sb == b && i > start && i < end)
                    {
                        self.section_out.insert(r);
                    }
                }
            }
        }
    }

    /// An instruction inside a whole-wave section: the values that leave
    /// the section are made from what the simulation found them to be.
    fn section_inst(&mut self, b: usize, i: usize, start: usize) {
        const UNKNOWN: Why = "a whole-wave section step the simulation does not follow";
        let inst = &self.x.p.blocks[b].insts[i];
        for &(_, w) in &inst.writes {
            self.cur_reads.clear();
            self.cur_edge = None;
            let val = if self.section_out.contains(&w) {
                let shape = self.sections[&b][&start].shapes.get(&w).copied();
                match shape.ok_or(UNKNOWN).and_then(|sh| self.shape(sh)) {
                    Ok(x) => {
                        let reads = std::mem::take(&mut self.cur_reads);
                        if !reads.is_empty() {
                            self.record_write_deps(w, b, None, false, None, reads);
                        }
                        Val::I32(x)
                    }
                    Err(why) => Val::Missing(why),
                }
            } else {
                Val::Missing(UNKNOWN)
            };
            self.vals.insert(w, val);
        }
    }

    /// A value a whole-wave section leaves, as DXIL wave operations over the
    /// host's active lanes: the console's result with console lanes; on
    /// another wave size, what the HLSL wave operation the console compiler
    /// expanded gives on that wave.
    fn shape(&mut self, shape: super::wave::Shape) -> Result<V, Why> {
        use super::wave::{Op, Shape};
        let (op, src, mask, ident, prefix) = match shape {
            Shape::K(k) => return Ok(self.s.m.i32(k)),
            Shape::Value(v) => return self.i32_of(v),
            Shape::Active(op, src, mask, ident) => (op, src, mask, ident, None),
            Shape::Masked(src, mask, ident) => {
                let c = self.cur;
                let &m = self
                    .x
                    .st
                    .masks
                    .get(&mask)
                    .ok_or("a wave reduction's untracked mask")?;
                let on = self.boolean(m, c)?;
                let x = self.i32_of(src)?;
                let id = self.s.m.i32(ident);
                return Ok(self.s.m.select(c, on, x, id));
            }
            Shape::Prefix(op, src, mask, ident, inclusive) => {
                (op, src, mask, ident, Some(inclusive))
            }
        };
        if self.mask_data == MaskData::None {
            return Err("a wave reduction in a mesh shader");
        }
        let c = self.cur;
        let &m = self
            .x
            .st
            .masks
            .get(&mask)
            .ok_or("a wave reduction's untracked mask")?;
        let on = self.boolean(m, c)?;
        let x = self.i32_of(src)?;
        let id = self.s.m.i32(ident);
        let x = self.s.m.select(c, on, x, id);
        self.s.flags |= flags::WAVE_OPS;
        let float = matches!(op, Op::AddF | Op::MinF | Op::MaxF);
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let arg = if float { self.as_float(x) } else { x };
        let ty = if float { f32t } else { i32t };
        let sfx = if float { "f32" } else { "i32" };
        let r = match (op, prefix) {
            (Op::AddU | Op::AddF, Some(inclusive)) => {
                let (k0, k1) = (self.s.m.i8(0), self.s.m.i8(1));
                let r = self.s.op(
                    c,
                    &format!("wavePrefixOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    121,
                    &[arg, k0, k1],
                );
                if inclusive {
                    self.s.m.bin(c, Bin::Add, r, arg)
                } else {
                    r
                }
            }
            (_, Some(_)) => return Err("a wave prefix of an operation other than a sum"),
            (Op::Or | Op::And | Op::Xor, None) => {
                let k = self.s.m.i8(match op {
                    Op::And => 0,
                    Op::Or => 1,
                    _ => 2,
                });
                self.s
                    .op(c, "waveActiveBit.i32", Attr::NoUnwind, i32t, 120, &[arg, k])
            }
            (_, None) => {
                let (code, unsigned) = match op {
                    Op::AddU => (0, 1),
                    Op::AddF => (0, 0),
                    Op::MinU => (2, 1),
                    Op::MaxU => (3, 1),
                    Op::MinI | Op::MinF => (2, 0),
                    _ => (3, 0),
                };
                let (k, u) = (self.s.m.i8(code), self.s.m.i8(unsigned));
                self.s.op(
                    c,
                    &format!("waveActiveOp.{sfx}"),
                    Attr::NoUnwind,
                    ty,
                    119,
                    &[arg, k, u],
                )
            }
        };
        Ok(if float { self.as_int(r) } else { r })
    }

    fn run(mut self) -> Result<Lowered, ShaderError> {
        let p = self.x.p;
        let order = crate::ir::reverse_postorder(&p.blocks);
        for (n, &b) in order.iter().enumerate() {
            self.rpo[b] = n;
        }
        for (n, &b) in order.iter().enumerate() {
            self.block[b] = Some(if n == 0 {
                self.s.entry
            } else {
                self.s.m.new_block()
            });
        }
        // Loop headers count iterations (`LOOP_LIMIT`).
        for &b in &order {
            let header = p.blocks[b]
                .preds
                .iter()
                .any(|&q| self.block[q].is_some() && self.rpo[b] <= self.rpo[q]);
            if header {
                let blk = self.block[b].expect("reachable");
                let phi = self.s.m.phi(blk, self.s.t.i32);
                self.counters.insert(b, (phi, Vec::new()));
            }
        }
        self.cur = self.s.entry;
        self.find_sections();
        self.inputs()?;
        if let Some(mesh) = self.s.mesh {
            self.mesh_prologue(mesh).map_err(|w| unsupported(0, w))?;
            self.block[order[0]] = Some(self.cur);
        }
        for &b in &order {
            self.cur = self.block[b].expect("reachable");
            self.cur_ir = b;
            let preds: Vec<usize> = p.blocks[b]
                .preds
                .iter()
                .copied()
                .filter(|&q| self.block[q].is_some())
                .collect();
            self.exported[b] = !preds.is_empty() && preds.iter().all(|&q| self.exported_out(q));
            for (i, phi) in p.blocks[b].phis.iter().enumerate() {
                let val = match phi.reg {
                    Reg::S(c) if c == EXEC || c == EXEC + 1 => Val::Missing("a lane mask"),
                    Reg::Scc => {
                        let v = self.s.m.phi(self.cur, self.s.t.i1);
                        self.data_phis.push((v, b, i));
                        self.taint_candidates.insert(phi.value);
                        Val::Bit(v)
                    }
                    _ => {
                        let v = self.s.m.phi(self.cur, self.s.t.i32);
                        self.data_phis.push((v, b, i));
                        self.taint_candidates.insert(phi.value);
                        Val::I32(v)
                    }
                };
                self.vals.insert(phi.value, val);
            }
            let mut i = 0;
            while i < p.blocks[b].insts.len() {
                self.inst(b, i)?;
                if let Some(end) = self.sections.get(&b).and_then(|s| s.get(&i)).map(|s| s.end) {
                    for j in i + 1..end {
                        self.section_inst(b, j, i);
                    }
                    i = end;
                    continue;
                }
                i += 1;
            }
            self.mask_ok = self.mask_data == MaskData::Console;
            self.tail[b] = Some(self.cur);
            self.exit(b)?;
        }
        self.fill_phis()?;
        let user_data = self.user_reads.values().copied().collect();
        let table_views = self
            .table_views
            .into_iter()
            .map(|((slot, dword), props)| TableView { slot, dword, props })
            .collect();
        self.s.samplers = self.samplers.len() as u32;
        let root_cbvs = self.s.root_cbvs.iter().map(|r| r.0).collect();
        let root_srvs = self.s.root_srvs.iter().map(|r| r.0).collect();
        let stride_table = self.s.stride_table.is_some();
        let cmp_samplers = self.s.cmp_samplers;
        Ok(Lowered {
            #[cfg(test)]
            unaligned_typed_loads: self.s.unaligned_typed_loads,
            cmp_samplers,
            root_cbvs,
            root_srvs,
            stride_table,
            #[cfg(test)]
            nonuniform_handles: self.s.nonuniform_handles,
            #[cfg(test)]
            heap_props: std::mem::take(&mut self.s.heap_props),
            #[cfg(test)]
            wave_ops: self.s.flags & flags::WAVE_OPS != 0,
            dxil: self.s.finish(),
            user_data,
            samplers: self.samplers,
            table_views,
        })
    }

    fn exported_out(&self, b: usize) -> bool {
        self.exported[b] || self.has_done_export(b)
    }

    fn has_done_export(&self, b: usize) -> bool {
        (0..self.x.p.blocks[b].insts.len())
            .any(|i| self.exports.get(&(b, i)).is_some_and(|e| e.done))
    }

    /// Values of the program's input registers.
    fn inputs(&mut self) -> Result<(), ShaderError> {
        let io_inputs: HashMap<Value, Input> = self.x.io.inputs.iter().copied().collect();
        for &(reg, v) in &self.x.p.inputs {
            let val = match reg {
                Reg::S(_) if self.params.contains_key(&v) => continue, // loaded on first use
                // Most programs only interpolate with them: loaded on first
                // use as a value (`get`).
                Reg::V(_) if matches!(io_inputs.get(&v), Some(Input::Barycentric { .. })) => {
                    continue;
                }
                Reg::S(n) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Above the SGPRs the hardware initializes, a register
                    // holds what the last wave left, so no program depends
                    // on it: a read the paths only seem to allow (a branch
                    // past the write, taken only with one past the read)
                    // reads zero, as VGPRs do.
                    None if n <= 105 && self.x.io.initialized_sgprs.is_some_and(|k| n >= k) => {
                        Val::I32(self.s.m.i32(0))
                    }
                    None => Val::Missing("a hardware SGPR input"),
                },
                Reg::Scc => Val::Missing("SCC on entry"),
                Reg::V(_) => match io_inputs.get(&v) {
                    Some(&input) => self.input(input),
                    // Registers no one set read as zero (the validator
                    // refuses reads of undef).
                    None => Val::I32(self.s.m.i32(0)),
                },
            };
            self.vals.insert(v, val);
        }
        Ok(())
    }

    fn input(&mut self, input: Input) -> Val {
        let (kind, col) = match input {
            Input::VertexId => (1, 0),
            Input::InstanceId => (2, 0),
            Input::Position(c) => (3, c),
            Input::Attribute { row, channel } => {
                return match self.find_input(row, channel) {
                    Some((id, col)) => self.load_input(id, col),
                    None => Val::Missing("a vertex input the signature lacks"),
                };
            }
            Input::Barycentric { kind, index } => return self.barycentric(kind, index),
            Input::PrimitiveId => return Val::Missing("the primitive id"),
            // The console code tests it against 0 (`v_cmp_ne_u32`), as the
            // host's value can be.
            Input::FrontFace => (13, 0),
            Input::ThreadIdInGroup(c) => return self.thread_id(c),
            Input::GroupId(c) => return self.compute_id("groupId.i32", 94, c),
            Input::GroupInfo => return Val::Missing("the thread-group info SGPR"),
            Input::MeshGroup => {
                let Val::I32(g) = self.compute_id("groupId.i32", 94, 0) else {
                    unreachable!()
                };
                let n = self.s.m.i32(self.x.mesh_stride);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Mul, g, n));
            }
            Input::MergedWaveInfo => {
                // shortcut: only the wave index (bits 24-27) is set; the
                // console's other bits (lane counts) are read by no program.
                let f = self.mesh_thread();
                let shift = self.s.m.i32(self.wave().trailing_zeros());
                let w = self.s.m.bin(self.s.entry, Bin::LShr, f, shift);
                let k24 = self.s.m.i32(24);
                return Val::I32(self.s.m.bin(self.s.entry, Bin::Shl, w, k24));
            }
            _ => return Val::Missing("a pixel system input"),
        };
        let free = self
            .s
            .inputs
            .iter()
            .filter_map(|e| e.row)
            .map(|r| r + 1)
            .max()
            .unwrap_or(0);
        let id = match self.s.inputs.iter().position(|e| e.kind == kind) {
            Some(id) => {
                if kind != 3 {
                    self.s.inputs[id].row.get_or_insert(free);
                }
                id
            }
            None if kind >= 3 => return Val::Missing("a pixel system input the signature lacks"),
            None => {
                // The console signature leaves out vertex and instance ids; PC
                // gives them a row after the others.
                let row = self
                    .s
                    .inputs
                    .iter()
                    .filter_map(|e| e.row)
                    .map(|r| r + 1)
                    .max()
                    .unwrap_or(0);
                self.s.inputs.push(SigElement {
                    name: ["", "SV_VertexID", "SV_InstanceID"][kind as usize].into(),
                    index: 0,
                    kind,
                    comp: 5,
                    interp: 0,
                    row: Some(row),
                    col: 0,
                    cols: 1,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        self.load_input(id, col)
    }

    /// A component of a compute thread or group id, read in the entry block.
    fn compute_id(&mut self, name: &str, opcode: u32, c: u8) -> Val {
        let (entry, c) = (self.s.entry, self.s.m.i32(u32::from(c)));
        Val::I32(
            self.s
                .op(entry, name, Attr::ReadNone, self.s.t.i32, opcode, &[c]),
        )
    }

    fn find_input(&self, row: u8, channel: u8) -> Option<(usize, u8)> {
        self.s.inputs.iter().enumerate().find_map(|(id, e)| {
            let r = e.row?;
            // `then`, not `then_some`: the channel is only past `col` when
            // the condition holds.
            (r == u32::from(row) && channel >= e.col && channel < e.col + e.cols)
                .then(|| (id, channel - e.col))
        })
    }

    fn load_input(&mut self, id: usize, col: u8) -> Val {
        if self.s.inputs[id].half {
            // shortcut: 16-bit vertex inputs (the fetch routine's) are not
            // unpacked; pixel inputs are (`interp`).
            return Val::Missing("a 16-bit input");
        }
        self.load_any_input(id, col)
    }

    /// An input element's channel as declared (16-bit ones as 32-bit floats).
    fn load_any_input(&mut self, id: usize, col: u8) -> Val {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let (int, bit) = (e.comp != 9, e.comp == 1);
        let ty = if bit {
            self.s.t.i1
        } else if int {
            self.s.t.i32
        } else {
            self.s.t.f32
        };
        let suffix = if bit {
            ".i1"
        } else if int {
            ".i32"
        } else {
            ".f32"
        };
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.undef(self.s.t.i32),
        ];
        let v = self.s.op(
            entry,
            &format!("loadInput{suffix}"),
            Attr::ReadNone,
            ty,
            4,
            &args,
        );
        if bit {
            let z = self.s.m.cast(entry, Cast::ZExt, v, self.s.t.i32);
            Val::I32(z)
        } else if int {
            Val::I32(v)
        } else {
            Val::I32(self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.i32))
        }
    }

    fn record_read(&mut self, v: Value) {
        if !self.taint_candidates.contains(&v) {
            return;
        }
        let dep = match self.cur_edge {
            Some((from, to)) => ReadDep::Edge { from, to, value: v },
            None => ReadDep::Value(v),
        };
        if !self.cur_reads.contains(&dep) {
            self.cur_reads.push(dep);
        }
    }

    fn record_write_deps(
        &mut self,
        w: Value,
        block: usize,
        guard: Option<Bool>,
        invert_guard: bool,
        ignore_why: Option<Why>,
        deps: Vec<ReadDep>,
    ) {
        self.taint_candidates.insert(w);
        let list = self.deps.entry(w).or_insert_with(|| {
            self.deps_order.push(w);
            Vec::new()
        });
        list.push(ValueDeps {
            block,
            guard,
            invert_guard,
            ignore_why,
            deps,
        });
    }

    /// The DXIL form of an IR value.
    fn get(&mut self, v: Value) -> Result<Val, Why> {
        self.get_inner(v, true)
    }

    fn get_inner(&mut self, v: Value, track: bool) -> Result<Val, Why> {
        let is_mask = self.x.st.masks.contains_key(&v) || self.x.st.highs.contains_key(&v);
        match self.vals.get(&v) {
            Some(&Val::Missing(_)) | None if self.mask_ok && is_mask => {
                return self.mask_dword(v).map(Val::I32);
            }
            Some(&Val::Missing(w)) => return Err(w),
            Some(&val) => {
                if track {
                    self.record_read(v);
                }
                return Ok(val);
            }
            None => {}
        }
        if let Some(&(param, dword)) = self.params.get(&v) {
            let val = self.user(param, dword);
            self.vals.insert(v, val);
            return Ok(val);
        }
        if let Some(loaded) = self.lazy_barycentric(v) {
            loaded?;
            return Ok(self.vals[&v]);
        }
        Err("a value defined after its use")
    }

    /// Loads `v` if it is a barycentric input VGPR not loaded yet (see
    /// `inputs`): `None` if it is not one.
    fn lazy_barycentric(&mut self, v: Value) -> Option<Result<(), Why>> {
        let (kind, index) = self.x.io.inputs.iter().find_map(|&(w, i)| match i {
            Input::Barycentric { kind, index } if w == v => Some((kind, index)),
            _ => None,
        })?;
        let val = self.barycentric(kind, index);
        self.vals.insert(v, val);
        Some(match val {
            Val::Missing(w) => Err(w),
            _ => Ok(()),
        })
    }

    /// Barycentric `index` (0 i, 1 j) of `kind` as a value: a channel of
    /// `SV_Barycentrics` declared with the same interpolation mode.
    ///
    /// GFX10 interpolates P0 + i (P1 - P0) + j (P2 - P0), so i and j are
    /// the weights of vertices 1 and 2. The host's weights (x, y, z) are
    /// those of vertices 0, 1 and 2, vertex 0 being the provoking vertex,
    /// which is the console's P0 (`v_interp_mov` P0 is lowered to a flat
    /// input): i is y and j is z.
    /// shortcut: triangles only; on lines and points the console's i and j
    /// are not checked against the host's weights.
    fn barycentric(&mut self, kind: Barycentric, index: u8) -> Val {
        let Some(mode) = interp_mode(kind) else {
            return Val::Missing("pull-model barycentrics");
        };
        if index > 1 {
            return Val::Missing("pull-model barycentrics");
        }
        // One element per interpolation mode (the console's own, if its
        // signature declares one). DXIL allows two, one perspective and
        // one linear: a second perspective (or linear) mode is refused,
        // since its weights differ from the first's where the modes'
        // sample points do (centroid weights stay inside the triangle,
        // center ones need not), so reusing the declared element would
        // be a silent wrong value.
        let bary = |e: &SigElement| e.kind == 28;
        let clash = self.s.inputs.iter().any(|e| {
            bary(e) && e.usage != 0 && e.interp != mode && linear(e.interp) == linear(mode)
        });
        if clash {
            return Val::Missing(if linear(mode) {
                "barycentrics of two linear modes (SV_Barycentrics allows one)"
            } else {
                "barycentrics of two perspective modes (SV_Barycentrics allows one)"
            });
        }
        let id = match self
            .s
            .inputs
            .iter()
            .position(|e| bary(e) && (e.interp == mode || e.usage == 0))
        {
            Some(id) => id,
            None => {
                let index = self.s.inputs.iter().filter(|e| bary(e)).count() as u32;
                if index >= 2 {
                    return Val::Missing("barycentrics of a third interpolation mode");
                }
                // A system value the validator wants unpacked (packing
                // location -1), like other non-packed system values.
                self.s.inputs.push(SigElement {
                    name: "SV_Barycentrics".into(),
                    index,
                    kind: 28,
                    comp: 9,
                    interp: mode,
                    row: None,
                    col: 0,
                    cols: 3,
                    usage: 0,
                    half: false,
                });
                self.s.inputs.len() - 1
            }
        };
        let e = &mut self.s.inputs[id];
        if e.col != 0 || e.cols < 3 {
            return Val::Missing("a console barycentrics element without i and j");
        }
        e.row = None;
        e.interp = mode;
        self.s.flags |= flags::BARYCENTRICS;
        self.load_any_input(id, index + 1)
    }

    /// Reads an overwritten register's previous value, recording its lane
    /// dependency (`invert_guard` for `!guard`, active lanes otherwise) so
    /// direct missing values and tainted phis still propagate where kept.
    fn old_i32_of(&mut self, v: Option<Value>, invert_guard: bool, ignore_why: Option<Why>) -> V {
        let Some(v) = v else {
            return self.s.m.i32(0);
        };
        match self.get_inner(v, false) {
            Ok(Val::I32(x)) => {
                if self.taint_candidates.contains(&v) {
                    self.cur_old_deps.push(OldDep {
                        value: Some(v),
                        direct_why: None,
                        invert_guard,
                        ignore_why,
                    });
                }
                x
            }
            Ok(Val::Bit(x)) => {
                if self.taint_candidates.contains(&v) {
                    self.cur_old_deps.push(OldDep {
                        value: Some(v),
                        direct_why: None,
                        invert_guard,
                        ignore_why,
                    });
                }
                self.s.m.cast(self.cur, Cast::ZExt, x, self.s.t.i32)
            }
            Ok(Val::Missing(why)) | Err(why) => {
                if ignore_why != Some(why) {
                    self.cur_old_deps.push(OldDep {
                        value: None,
                        direct_why: Some(why),
                        invert_guard,
                        ignore_why,
                    });
                }
                self.s.m.i32(0)
            }
        }
    }

    fn i32_of(&mut self, v: Value) -> Result<V, Why> {
        match self.get(v)? {
            Val::I32(x) => Ok(x),
            Val::Bit(x) => Ok(self.s.m.cast(self.cur, Cast::ZExt, x, self.s.t.i32)),
            Val::Missing(w) => Err(w),
        }
    }

    /// A user-data dword: a root parameter's value.
    fn user(&mut self, param: usize, dword: u32) -> Val {
        let rp = self.x.native.layout[param];
        match rp.root {
            RootType::ResourceHeap | RootType::SamplerHeap => return Val::I32(self.s.m.i32(0)),
            RootType::Constants | RootType::Table => {}
            _ => return Val::Missing("a root descriptor"),
        }
        let index = u32::from(rp.user_data) + dword;
        self.user_reads.insert(
            index,
            UserData {
                index,
                slot: rp.slot,
                root: rp.root,
                dword,
            },
        );
        self.s.user_data = self.s.user_data.max(index + 1);
        let h = match self.user_data {
            Some(h) => h,
            None => {
                let h = self.s.user_data_handle();
                self.user_data = Some(h);
                h
            }
        };
        let entry = self.s.entry;
        let row = self.s.m.i32(index / 4);
        let r = self.s.op(
            entry,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, row],
        );
        Val::I32(self.s.m.extract(entry, r, index % 4))
    }

    fn extend_cur_reads(&mut self, extra: &[ReadDep]) {
        for &dep in extra {
            if !self.cur_reads.contains(&dep) {
                self.cur_reads.push(dep);
            }
        }
    }

    /// A dword of a lane mask as a number: the ballot of the lanes' bits,
    /// taken in the block that writes the mask (see [`MaskData`]).
    fn mask_dword(&mut self, v: Value) -> Result<V, Why> {
        let (lo, high) = match self.x.st.highs.get(&v) {
            Some(&lo) => (lo, true),
            None => (v, false),
        };
        let &m = self.x.st.masks.get(&lo).ok_or(LANE_MASK)?;
        if !self.x.st.active.contains(&lo) {
            return Err("a lane mask holding lanes elsewhere read as data");
        }
        let (x, y) = match self.ballots.get(&lo) {
            Some(&xy) => {
                if let Some(deps) = self.ballot_deps.get(&lo) {
                    let deps = deps.clone();
                    self.extend_cur_reads(&deps);
                }
                xy
            }
            None => {
                let at = match self.def_block.get(&lo) {
                    // Program inputs (EXEC on entry).
                    None => self.s.entry,
                    Some(&b) if b == self.cur_ir => self.cur,
                    Some(&b) => self.tail[b].ok_or("a lane mask written after its use")?,
                };
                if !self.balloting.insert(lo) {
                    return Err("a lane mask whose lanes depend on its own value");
                }
                let saved = std::mem::take(&mut self.cur_reads);
                let bit = self.boolean(m, at);
                self.balloting.remove(&lo);
                let sub = std::mem::replace(&mut self.cur_reads, saved);
                let bit = bit?;
                self.extend_cur_reads(&sub);
                if !sub.is_empty() {
                    self.ballot_deps.insert(lo, sub);
                }
                let i32t = self.s.t.i32;
                let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
                let r = self
                    .s
                    .op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[bit]);
                self.s.flags |= flags::WAVE_OPS;
                let xy = (self.s.m.extract(at, r, 0), self.s.m.extract(at, r, 1));
                self.ballots.insert(lo, xy);
                xy
            }
        };
        Ok(if high { y } else { x })
    }

    // ---- operands ----

    fn operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Ok(self.s.m.i32(0)),
            Operand::Scalar { code, .. } => {
                let v = inst.read(Reg::S(code)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Vector { reg, .. } => {
                let v = inst.read(Reg::V(reg)).ok_or("an unread operand")?;
                self.i32_of(v)
            }
            Operand::Const(c) => crate::resource::inline(c)
                .map(|x| self.s.m.i32(x))
                .ok_or("a special operand (vccz, execz, scc, lds_direct)"),
            Operand::Literal(x) => Ok(self.s.m.i32(x)),
            Operand::Attr { .. } => Err("an attribute operand"),
        }
    }

    /// GFX10 puts an inline f16 constant in the low half of the operand.
    fn f16_operand(&mut self, inst: &Inst, op: &Operand) -> Result<V, Why> {
        match *op {
            Operand::Const(c @ 240..=248) => {
                let h = [
                    0x3800, 0xB800, 0x3C00, 0xBC00, 0x4000, 0xC000, 0x4400, 0xC400, 0x3118,
                ][usize::from(c - 240)];
                Ok(self.s.m.i32(h))
            }
            _ => self.operand(inst, op),
        }
    }

    fn as_float(&mut self, x: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, x, self.s.t.f32)
    }

    fn as_int(&mut self, f: V) -> V {
        self.s.m.cast(self.cur, Cast::Bitcast, f, self.s.t.i32)
    }

    /// The SDWA operand selection of source `k`: a byte or word of the
    /// dword, zero- or sign-extended (none: the whole dword).
    fn sdwa_src(inst: &Inst, k: usize) -> Option<(u8, bool)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        let sel = *s.src_sel.get(k)?;
        let sext = s.sext >> k & 1 != 0;
        (sel < 6).then_some((sel, sext))
    }

    /// Bits `sel` (0-3 a byte, 4-5 a word) of `x`, extended.
    fn select_bits(&mut self, x: V, sel: u8, sext: bool) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8, 8 * sel)
        } else {
            (16, 16 * (sel - 4))
        };
        let left = self.s.m.i32(u32::from(32 - width - at));
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - u32::from(width));
        let op = if sext { Bin::AShr } else { Bin::LShr };
        self.s.m.bin(c, op, t, right)
    }

    /// A float source with its abs/neg modifiers.
    fn fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let selected = Self::sdwa_src(inst, k);
        let mut f = match op {
            _ if selected.is_some() => {
                let (sel, sext) = selected.expect("checked");
                let x = self.operand(inst, &op)?;
                let x = self.select_bits(x, sel, sext);
                self.as_float(x)
            }
            _ if k == 0 && self.src0.is_some() => {
                let x = self.src0.expect("checked");
                self.as_float(x)
            }
            Operand::Vector { reg, .. } | Operand::Scalar { code: reg, .. }
                if !matches!(op, Operand::Scalar { code: NULL, .. }) =>
            {
                let r = if matches!(op, Operand::Vector { .. }) {
                    Reg::V(reg)
                } else {
                    Reg::S(reg)
                };
                let val = inst.read(r).ok_or("an unread operand")?;
                self.float_of(val)?
            }
            _ => {
                let x = self.operand(inst, &op)?;
                self.as_float(x)
            }
        };
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    fn src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 {
            return Err("integer operand modifiers");
        }
        match self.src0 {
            Some(x) if k == 0 => Ok(x),
            _ => {
                let x = self.operand(inst, &op)?;
                Ok(match Self::sdwa_src(inst, k) {
                    Some((sel, sext)) => self.select_bits(x, sel, sext),
                    None => x,
                })
            }
        }
    }

    fn unary(&mut self, op: u32, f: V) -> V {
        self.s.op(
            self.cur,
            "unary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[f],
        )
    }

    fn binary(&mut self, op: u32, a: V, b: V) -> V {
        self.s.op(
            self.cur,
            "binary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            op,
            &[a, b],
        )
    }

    /// The hardware's maximum (`max`) or minimum of two floats: as DXIL's
    /// FMax/FMin, but -0 orders below +0 (RDNA2 ISA, `v_max_f32`), where
    /// the compare and select DXIL specifies gives either zero.
    fn max_min(&mut self, max: bool, a: V, b: V) -> V {
        let c = self.cur;
        let r = self.binary(if max { 35 } else { 36 }, a, b);
        let (ia, ib) = (self.as_int(a), self.as_int(b));
        let either = self.s.m.bin(c, Bin::Or, ia, ib);
        let magnitude = self.s.m.i32(0x7FFF_FFFF);
        let either = self.s.m.bin(c, Bin::And, either, magnitude);
        let zero = self.s.m.i32(0);
        let zeros = self.s.m.cmp(c, pred::EQ, either, zero);
        // Of two zeros, +0 unless both are -0 (max), -0 unless neither (min).
        let op = if max { Bin::And } else { Bin::Or };
        let sign = self.s.m.bin(c, op, ia, ib);
        let signed = self.as_float(sign);
        self.s.m.select(c, zeros, signed, r)
    }

    fn fmad(&mut self, a: V, b: V, c: V) -> V {
        self.s.op(
            self.cur,
            "tertiary.f32",
            Attr::ReadNone,
            self.s.t.f32,
            46,
            &[a, b, c],
        )
    }

    /// `f` as an f16 in the low bits, rounded toward zero: the nearest-even
    /// conversion, one step back toward zero where it rounded away (which
    /// also turns an overflow to infinity into the largest finite half).
    fn f16_toward_zero(&mut self, f: V) -> V {
        let c = self.cur;
        let i32t = self.s.t.i32;
        let h = self
            .s
            .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
        let back = self
            .s
            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h]);
        let (fa, ba) = (self.unary(6, f), self.unary(6, back));
        // Ordered: never for NaN.
        let away = self.s.m.cmp(c, pred::OGT, ba, fa);
        let one = self.s.m.i32(1);
        let stepped = self.s.m.bin(c, Bin::Sub, h, one);
        self.s.m.select(c, away, stepped, h)
    }

    /// `(a >> off[4:0]) & ((1 << width) - 1)`, sign-extended from the field's
    /// top bit when `signed`; widths from 32 on take every bit. A signed
    /// field that runs past bit 31 extends from bit 31, as the hardware
    /// does: the shift is arithmetic then.
    fn bfe(&mut self, a: V, off: V, width: V, signed: bool) -> V {
        let c = self.cur;
        let (k31, k32, zero, one, ones) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(0),
            self.s.m.i32(1),
            self.s.m.i32(u32::MAX),
        );
        let off = self.s.m.bin(c, Bin::And, off, k31);
        let shift = if signed { Bin::AShr } else { Bin::LShr };
        let shifted = self.s.m.bin(c, shift, a, off);
        let full = self.s.m.cmp(c, pred::IUGE, width, k32);
        let w = self.s.m.bin(c, Bin::And, width, k31);
        let mask = self.s.m.bin(c, Bin::Shl, one, w);
        let mask = self.s.m.bin(c, Bin::Sub, mask, one);
        let mask = self.s.m.select(c, full, ones, mask);
        let r = self.s.m.bin(c, Bin::And, shifted, mask);
        if !signed {
            return r;
        }
        // Sign-extend a field narrower than 32 bits (none for width 0).
        let left = self.s.m.bin(c, Bin::Sub, k32, w);
        let left = self.s.m.bin(c, Bin::And, left, k31);
        let t = self.s.m.bin(c, Bin::Shl, r, left);
        let t = self.s.m.bin(c, Bin::AShr, t, left);
        let is0 = self.s.m.cmp(c, pred::EQ, w, zero);
        let keep = self.s.m.bin(c, Bin::Or, full, is0);
        self.s.m.select(c, keep, r, t)
    }

    /// The median of three. With a NaN among them the hardware gives the
    /// minimum of the three instead (the host's minimum skips NaNs, so that
    /// is the smallest number among them).
    fn med3(&mut self, a: V, b: V, third: V) -> V {
        let c = self.cur;
        let lo = self.binary(36, a, b);
        let hi = self.binary(35, a, b);
        let t = self.binary(36, hi, third);
        let median = self.binary(35, lo, t);
        let least = self.binary(36, lo, third);
        let ab = self.s.m.cmp(c, pred::UNO, a, b);
        let nan3 = self.s.m.cmp(c, pred::UNO, third, third);
        let nan = self.s.m.bin(c, Bin::Or, ab, nan3);
        self.s.m.select(c, nan, least, median)
    }

    /// `v_div_fixup_f32` / `v_div_fixup_f16`: finalizes a division sequence's
    /// quotient `quot` (`src0`) for `num / denom` (`src2 / src1`), resolving
    /// IEEE-754 special cases (`NaN`, `0/0`, `inf/inf`, `x/0`, `x/inf`,
    /// `0/x`, `inf/x`, and `quot == NaN` overflow) and applying `sign(denom)
    /// ^ sign(num)` in pure integer bit operations.
    fn div_fixup(&mut self, quot: V, denom: V, num: V) -> V {
        let c = self.cur;
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let q = self.as_int(quot);
        let d = self.as_int(denom);
        let n = self.as_int(num);

        let sign_mask = k(self, 0x8000_0000);
        let abs_mask = k(self, 0x7FFF_FFFF);
        let inf_bits = k(self, 0x7F80_0000);
        let qnan_bits = k(self, 0x7FC0_0000);
        let quiet_bit = k(self, 0x0040_0000);
        let zero = k(self, 0);

        let sign_xor = self.s.m.bin(c, Bin::Xor, d, n);
        let sign_out = self.s.m.bin(c, Bin::And, sign_xor, sign_mask);

        let q_mag = self.s.m.bin(c, Bin::And, q, abs_mask);
        let d_mag = self.s.m.bin(c, Bin::And, d, abs_mask);
        let n_mag = self.s.m.bin(c, Bin::And, n, abs_mask);

        let q_nan = self.s.m.cmp(c, pred::IUGT, q_mag, inf_bits);
        let d_nan = self.s.m.cmp(c, pred::IUGT, d_mag, inf_bits);
        let n_nan = self.s.m.cmp(c, pred::IUGT, n_mag, inf_bits);

        let d_inf = self.s.m.cmp(c, pred::EQ, d_mag, inf_bits);
        let n_inf = self.s.m.cmp(c, pred::EQ, n_mag, inf_bits);
        let d_zero = self.s.m.cmp(c, pred::EQ, d_mag, zero);
        let n_zero = self.s.m.cmp(c, pred::EQ, n_mag, zero);

        let signed_inf = self.s.m.bin(c, Bin::Or, inf_bits, sign_out);
        let signed_zero = sign_out;
        let signed_q = self.s.m.bin(c, Bin::Or, q_mag, sign_out);

        // Priority (lowest to highest, built bottom-up via select):
        // 6. Normal/inf quotient: copysign(quot, sign_out); NaN quotient -> ±inf.
        let r = self.s.m.select(c, q_nan, signed_inf, signed_q);
        // 5. denom == ±inf or num == ±0 -> ±0.0 with sign_out.
        let make_zero = self.s.m.bin(c, Bin::Or, d_inf, n_zero);
        let r = self.s.m.select(c, make_zero, signed_zero, r);
        // 4. num == ±inf or denom == ±0 -> ±inf with sign_out.
        let make_inf = self.s.m.bin(c, Bin::Or, n_inf, d_zero);
        let r = self.s.m.select(c, make_inf, signed_inf, r);
        // 3. (0 / 0) or (inf / inf) -> canonical qNaN.
        let zero_over_zero = self.s.m.bin(c, Bin::And, d_zero, n_zero);
        let inf_over_inf = self.s.m.bin(c, Bin::And, d_inf, n_inf);
        let indet = self.s.m.bin(c, Bin::Or, zero_over_zero, inf_over_inf);
        let r = self.s.m.select(c, indet, qnan_bits, r);
        // 2. num is NaN -> quiet(num).
        let n_quiet = self.s.m.bin(c, Bin::Or, n, quiet_bit);
        let r = self.s.m.select(c, n_nan, n_quiet, r);
        // 1. denom is NaN -> quiet(denom).
        let d_quiet = self.s.m.bin(c, Bin::Or, d, quiet_bit);
        let r = self.s.m.select(c, d_nan, d_quiet, r);

        self.as_float(r)
    }

    /// `a - floor(a)`, which on the hardware never reaches 1.0: where it
    /// would round to it (a tiny negative `a`), the result is `below_one`,
    /// the largest value under 1.0 in the operation's width. Hosts differ
    /// here (some give 1.0). Comparing against a `threshold` distinct from
    /// `below_one` prevents drivers (observed on NVIDIA) from folding
    /// `r > k ? k : r` into `minNum` (`FMNMX`) and swallowing NaN.
    fn fract(&mut self, a: V, threshold: f32, below_one: f32) -> V {
        let c = self.cur;
        let r = self.unary(22, a);
        let limit = self.s.m.f32(threshold.to_bits());
        let k = self.s.m.f32(below_one.to_bits());
        let over = self.s.m.cmp(c, pred::OGE, r, limit);
        self.s.m.select(c, over, k, r)
    }

    /// Sine, or cosine, of an angle in turns, as the hardware takes it. The
    /// whole turns are taken off first (`x - round_even(x)`, exact, within
    /// half a turn), so a large angle keeps its fraction; scaling to radians
    /// first rounds it away. A zero passes unreduced (the sine of -0 is -0).
    /// The sine of a half turn and the cosine of a quarter turn are exactly
    /// +0 on the hardware, where the host's function of the rounded angle
    /// gives about 1e-7. Infinities and NaN give NaN.
    /// shortcut: below 2^-12 turns the hardware scales by a slightly
    /// different constant than 2π (a last-place difference).
    fn trig_turns(&mut self, x: V, cosine: bool) -> V {
        let c = self.cur;
        let zero = self.s.m.f32(0);
        let whole = self.unary(26, x);
        let part = self.s.m.bin(c, Bin::Sub, x, whole);
        let is_zero = self.s.m.cmp(c, pred::OEQ, x, zero);
        let part = self.s.m.select(c, is_zero, x, part);
        let k = self.s.m.f32(TWO_PI.to_bits());
        let angle = self.s.m.bin(c, Bin::Mul, part, k);
        let r = self.unary(if cosine { 12 } else { 13 }, angle);
        let at = if cosine { 0.25f32 } else { 0.5 };
        let at = self.s.m.f32(at.to_bits());
        let size = self.unary(6, part);
        let exact = self.s.m.cmp(c, pred::OEQ, size, at);
        self.s.m.select(c, exact, zero, r)
    }

    /// The high dword of a 32 × 32-bit product.
    fn mul_hi(&mut self, a: V, b: V, signed: bool) -> V {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let ext = if signed { Cast::SExt } else { Cast::ZExt };
        let a = self.s.m.cast(c, ext, a, i64t);
        let b = self.s.m.cast(c, ext, b, i64t);
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let k32 = self.s.m.int_const(i64t, 32);
        let hi = self.s.m.bin(c, Bin::LShr, m, k32);
        self.s.flags |= flags::INT64;
        self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32)
    }

    /// `0 × anything = 0`, as in D3D9.
    fn legacy_mul(&mut self, a: V, b: V) -> V {
        let c = self.cur;
        let m = self.s.m.bin(c, Bin::Mul, a, b);
        let zero = self.s.m.f32(0);
        let az = self.s.m.cmp(c, pred::OEQ, a, zero);
        let bz = self.s.m.cmp(c, pred::OEQ, b, zero);
        let z = self.s.m.bin(c, Bin::Or, az, bz);
        self.s.m.select(c, z, zero, m)
    }

    /// Decomposes an f32 into `(mantissa, exponent)` per RDNA2 `v_frexp_*`:
    /// - finite non-zero (normal or subnormal): `mantissa` in `(-1, -0.5] U [0.5, 1)`
    ///   and `exponent` such that `a == mantissa * 2^exponent`;
    /// - `±0`, `±inf`, `NaN`: `mantissa = a` (preserving sign/special value), `exponent = 0`.
    fn frexp_f32(&mut self, a: V) -> (V, V) {
        let c = self.cur;
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let bits = self.as_int(a);
        let sign_mask = k(self, 0x8000_0000);
        let mag_mask = k(self, 0x7FFF_FFFF);
        let mant_mask = k(self, 0x007F_FFFF);
        let half_exp = k(self, 0x3F00_0000);
        let (k0, k23, k255, k126) = (k(self, 0), k(self, 23), k(self, 255), k(self, 126));

        let sign = self.s.m.bin(c, Bin::And, bits, sign_mask);
        let mag = self.s.m.bin(c, Bin::And, bits, mag_mask);
        let exp_raw = self.s.m.bin(c, Bin::LShr, mag, k23);
        let mant_raw = self.s.m.bin(c, Bin::And, mag, mant_mask);

        // Normal path: biased exponent 1..=254 -> exponent = exp_raw - 126, mantissa exp = 126.
        let norm_mant = self.s.m.bin(c, Bin::Or, sign, half_exp);
        let norm_mant = self.s.m.bin(c, Bin::Or, norm_mant, mant_raw);
        let norm_exp = self.s.m.bin(c, Bin::Sub, exp_raw, k126);

        // Subnormal path (exp_raw == 0, mant_raw != 0): shift top bit of mant_raw to bit 23.
        // FirstbitHi (op 33) counts leading zeros from bit 31: bit k -> lz = 31 - k in [9..=31].
        let lz = self
            .s
            .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 33, &[mant_raw]);
        let k8 = k(self, 8);
        let sh = self.s.m.bin(c, Bin::Sub, lz, k8);
        let k31 = k(self, 31);
        let sh_safe = self.s.m.bin(c, Bin::And, sh, k31);
        let sub_mant_bits = self.s.m.bin(c, Bin::Shl, mant_raw, sh_safe);
        let sub_mant_bits = self.s.m.bin(c, Bin::And, sub_mant_bits, mant_mask);
        let sub_mant = self.s.m.bin(c, Bin::Or, sign, half_exp);
        let sub_mant = self.s.m.bin(c, Bin::Or, sub_mant, sub_mant_bits);
        // Value = mant_raw * 2^-149 = (0.5..1) * 2^(-117 - lz).
        let k_neg117 = k(self, (-117i32) as u32);
        let sub_exp = self.s.m.bin(c, Bin::Sub, k_neg117, lz);

        let is_exp0 = self.s.m.cmp(c, pred::EQ, exp_raw, k0);
        let is_zero = self.s.m.cmp(c, pred::EQ, mag, k0);
        let is_inf_nan = self.s.m.cmp(c, pred::EQ, exp_raw, k255);
        let is_special = self.s.m.bin(c, Bin::Or, is_zero, is_inf_nan);

        let mant_fin = self.s.m.select(c, is_exp0, sub_mant, norm_mant);
        let exp_fin = self.s.m.select(c, is_exp0, sub_exp, norm_exp);

        let mant_bits = self.s.m.select(c, is_special, bits, mant_fin);
        let exp = self.s.m.select(c, is_special, k0, exp_fin);
        (self.as_float(mant_bits), exp)
    }

    // ---- lane booleans ----

    fn boolean(&mut self, b: Bool, at: usize) -> Result<V, Why> {
        if b == Bool::TRUE || b == Bool::FALSE {
            return Ok(self.s.m.bool(b == Bool::TRUE));
        }
        if let Some(&v) = self.bools.get(&(at, b)) {
            if let Some(deps) = self.bool_deps.get(&(at, b)) {
                let deps = deps.clone();
                self.extend_cur_reads(&deps);
            }
            return Ok(v);
        }
        let saved = std::mem::take(&mut self.cur_reads);
        let r = self.boolean_uncached(b, at);
        let sub = std::mem::replace(&mut self.cur_reads, saved);
        let v = r?;
        self.extend_cur_reads(&sub);
        if !sub.is_empty() {
            self.bool_deps.insert((at, b), sub);
        }
        self.bools.insert((at, b), v);
        Ok(v)
    }

    fn boolean_uncached(&mut self, b: Bool, at: usize) -> Result<V, Why> {
        let (atom, lo, hi) = self.x.st.bools.node(b).expect("not a constant");
        let a = self.atom(atom, at)?;
        let m = &mut self.s.m;
        let v = match (lo, hi) {
            (Bool::FALSE, Bool::TRUE) => a,
            (Bool::TRUE, Bool::FALSE) => {
                let t = m.bool(true);
                m.bin(at, Bin::Xor, a, t)
            }
            (Bool::FALSE, h) => {
                let h = self.boolean(h, at)?;
                self.s.m.bin(at, Bin::And, a, h)
            }
            (l, Bool::TRUE) => {
                let l = self.boolean(l, at)?;
                self.s.m.bin(at, Bin::Or, a, l)
            }
            (l, h) => {
                let l = self.boolean(l, at)?;
                let h = self.boolean(h, at)?;
                self.s.m.select(at, a, h, l)
            }
        };
        Ok(v)
    }

    fn atom(&mut self, atom: Atom, at: usize) -> Result<V, Why> {
        match atom {
            Atom::Lane(v) => {
                let r = self.lane_bits.get(&v).copied().ok_or_else(|| {
                    self.lane_fail
                        .get(&v)
                        .copied()
                        .unwrap_or("a lane mask from a carry")
                })?;
                self.record_read(v);
                Ok(r)
            }
            Atom::Uniform(v) => match self.get(v)? {
                Val::Bit(x) => Ok(x),
                Val::I32(x) => {
                    let z = self.s.m.i32(0);
                    Ok(self.s.m.cmp(at, pred::NE, x, z))
                }
                Val::Missing(w) => Err(w),
            },
            Atom::Phi(id) => {
                if let Some(&v) = self.mask_phis.get(&id) {
                    return Ok(v);
                }
                let &(b, i) = self
                    .phi_at
                    .get(&id)
                    .ok_or("a lane-mask join without a phi")?;
                let blk = self.block[b].ok_or("an unreachable join")?;
                let v = self.s.m.phi(blk, self.s.t.i1);
                self.mask_phis.insert(id, v);
                self.pending_masks.push((v, b, i));
                Ok(v)
            }
            // shortcut: "some lane of the quad is set" is taken as true: the
            // lanes it adds only feed derivatives, and exports keep their own
            // masks. Wrong if such a section had a side effect.
            Atom::Quad(_) => Ok(self.s.m.bool(true)),
            // With console lanes, a mask held as a number is the console's:
            // the lane tests its own bit.
            Atom::Opaque(v) if self.console_lanes => {
                let hi = match self.pair_high(v) {
                    Some(h) => Some(self.get(h)?),
                    None => None,
                };
                let lo = self.get(v)?;
                let word = |w: Val| match w {
                    Val::I32(x) => Ok(x),
                    _ => Err("an opaque lane mask"),
                };
                let lo = word(lo)?;
                let hi = hi.map(word).transpose()?;
                Ok(self.lane_bit(at, lo, hi))
            }
            Atom::Constant(k) if self.console_lanes => {
                let lo = self.s.m.i32(k as u32);
                let hi = self.s.m.i32((k >> 32) as u32);
                Ok(self.lane_bit(at, lo, Some(hi)))
            }
            Atom::Opaque(_) => Err("an opaque lane mask"),
            Atom::Constant(_) => Err("a constant lane mask"),
        }
    }

    /// This lane's bit from a mask operand (`v_cndmask_b32` / `*_ci_u32`):
    /// uses the tracked boolean when `structure` followed it, handles inline
    /// `0`/`-1` constants, or tests this console lane's bit of the scalar
    /// mask register (pair in wave64) when console lanes are active.
    fn mask_cond(
        &mut self,
        inst: &Inst,
        op: Option<&Operand>,
        at: usize,
        err: Why,
    ) -> Result<V, Why> {
        match op {
            Some(&Operand::Const(128)) => Ok(self.s.m.bool(false)),
            Some(&Operand::Const(193)) => Ok(self.s.m.bool(true)),
            Some(&Operand::Scalar { code, .. }) => {
                if let Some(m) = inst
                    .read(Reg::S(code))
                    .and_then(|v| self.x.st.masks.get(&v).copied())
                {
                    return self.boolean(m, at);
                }
                if !self.console_lanes {
                    return Err(err);
                }
                let lo = inst.read(Reg::S(code)).ok_or(err)?;
                let lo = self.i32_of(lo)?;
                let hi = if self.x.native.wave64 {
                    match inst.read(Reg::S(code + 1)) {
                        Some(v) => Some(self.i32_of(v)?),
                        None => Some(self.s.m.i32(0)),
                    }
                } else {
                    None
                };
                self.s.flags |= flags::WAVE_OPS;
                Ok(self.lane_bit(at, lo, hi))
            }
            _ => Err(err),
        }
    }

    /// This lane's bit of the mask `lo`, `hi` (wave64).
    fn lane_bit(&mut self, at: usize, lo: V, hi: Option<V>) -> V {
        let lane = self.s.op(
            at,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        );
        let (k31, k32, one, z) = (
            self.s.m.i32(31),
            self.s.m.i32(32),
            self.s.m.i32(1),
            self.s.m.i32(0),
        );
        let word = match hi {
            Some(hi) => {
                let low = self.s.m.cmp(at, pred::IULT, lane, k32);
                self.s.m.select(at, low, lo, hi)
            }
            None => lo,
        };
        let sh = self.s.m.bin(at, Bin::And, lane, k31);
        let t = self.s.m.bin(at, Bin::LShr, word, sh);
        let t = self.s.m.bin(at, Bin::And, t, one);
        self.s.m.cmp(at, pred::NE, t, z)
    }

    /// The value of the high dword of the wave64 register pair whose low
    /// dword holds `lo` (none in wave32).
    fn pair_high(&self, lo: Value) -> Option<Value> {
        if !self.x.native.wave64 {
            return None;
        }
        let p = self.x.p;
        let pick = |reg: Reg| match reg {
            Reg::S(c) => Some(Reg::S(c + 1)),
            _ => None,
        };
        if let Some(&(reg, _)) = p.inputs.iter().find(|i| i.1 == lo) {
            let h = pick(reg)?;
            return p.inputs.iter().find(|i| i.0 == h).map(|i| i.1);
        }
        if let Some(&(b, i)) = self.phi_at.get(&lo) {
            let h = pick(p.blocks[b].phis[i].reg)?;
            return p.blocks[b]
                .phis
                .iter()
                .find(|q| q.reg == h)
                .map(|q| q.value);
        }
        let b = *self.def_block.get(&lo)?;
        let inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.writes.iter().any(|w| w.1 == lo))?;
        let &(reg, _) = inst.writes.iter().find(|w| w.1 == lo)?;
        inst.write(pick(reg)?)
    }

    fn guard(&mut self, b: usize, i: usize) -> Result<Option<V>, Why> {
        match self.x.st.guards[b][i] {
            None => Ok(None),
            Some(g) => Ok(Some(self.boolean(g, self.cur)?)),
        }
    }

    // ---- results ----

    fn set(&mut self, v: Option<Value>, val: Val) {
        if let Some(v) = v {
            self.vals.insert(v, val);
        }
    }

    /// Writes VGPR `reg` (or the SGPR of a lane read), under `guard`.
    fn write(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let x = match (Self::sdwa_dst(inst), reg) {
            (Some((sel, unused)), Reg::V(_)) => self.place_bits(inst, reg, x, sel, unused),
            _ => x,
        };
        self.write_placed(inst, reg, x, guard);
    }

    /// [`Self::write`] of a dword already placed (SDWA applied).
    fn write_placed(&mut self, inst: &Inst, reg: Reg, x: V, guard: Option<V>) {
        let Some(new) = inst.write(reg) else { return };
        let x = match (guard, reg) {
            (Some(g), Reg::V(_)) => {
                let old = self.old_i32_of(inst.read(reg), true, None);
                self.s.m.select(self.cur, g, x, old)
            }
            _ => x,
        };
        self.vals.insert(new, Val::I32(x));
    }

    /// The two register offsets of `v_movrelsd_2_b32` and `v_swaprel_b32`:
    /// M0[9:0] for the source, M0[25:16] for the destination.
    fn m0_halves(&mut self, m0: V) -> (V, V) {
        let c = self.cur;
        let low = self.s.m.i32(0x3FF);
        let sixteen = self.s.m.i32(16);
        let si = self.s.m.bin(c, Bin::And, m0, low);
        let high = self.s.m.bin(c, Bin::LShr, m0, sixteen);
        let di = self.s.m.bin(c, Bin::And, high, low);
        (si, di)
    }

    /// `v[base + index]`, from the M0-relative range `inst` reads from
    /// `base` on ([`crate::ir`] names it up to the program's VGPR count);
    /// 0 past it.
    fn rel_read(&mut self, inst: &Inst, base: u8, index: V) -> Result<V, Why> {
        let c = self.cur;
        let mut r = self.s.m.i32(0);
        for k in 0..=255u8.saturating_sub(base) {
            let Some(x) = inst.read(Reg::V(base + k)) else {
                break;
            };
            let x = self.i32_of(x)?;
            let kk = self.s.m.i32(u32::from(k));
            let is = self.s.m.cmp(c, pred::EQ, index, kk);
            r = self.s.m.select(c, is, x, r);
        }
        Ok(r)
    }

    /// The value a VGPR an M0-relative write may land on held before it
    /// (0 when the program never set it).
    fn old_vgpr(&mut self, inst: &Inst, reg: u8) -> Result<V, Why> {
        match inst.read(Reg::V(reg)) {
            Some(o) => self.i32_of(o),
            None => Ok(self.s.m.i32(0)),
        }
    }

    /// The SDWA destination selection: a byte or word, and what the other
    /// bits get (0 zeros, 1 the sign above the field, 2 the old bits).
    fn sdwa_dst(inst: &Inst) -> Option<(u8, u8)> {
        let Fields::Valu(v) = &inst.fields else {
            return None;
        };
        let s = v.sdwa?;
        (s.dst_sel < 6).then_some((s.dst_sel, s.dst_unused))
    }

    /// The low byte or word of `x` placed at `sel` in VGPR `reg`.
    fn place_bits(&mut self, inst: &Inst, reg: Reg, x: V, sel: u8, unused: u8) -> V {
        let c = self.cur;
        let (width, at) = if sel < 4 {
            (8u32, 8 * u32::from(sel))
        } else {
            (16, 16 * u32::from(sel - 4))
        };
        let left = self.s.m.i32(32 - width);
        let t = self.s.m.bin(c, Bin::Shl, x, left);
        let right = self.s.m.i32(32 - width - at);
        if unused == 1 {
            // Bits below the field shift in as zeros.
            return self.s.m.bin(c, Bin::AShr, t, right);
        }
        let field = self.s.m.bin(c, Bin::LShr, t, right);
        if unused != 2 {
            return field;
        }
        let old = self.old_i32_of(inst.read(reg), false, None);
        let keep = self.s.m.i32(!((((1u64 << width) - 1) as u32) << at));
        let old = self.s.m.bin(c, Bin::And, old, keep);
        self.s.m.bin(c, Bin::Or, old, field)
    }

    fn write_f(&mut self, inst: &Inst, f: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let mut f = f;
        match v.omod {
            0 => {}
            k => {
                let factor = [1.0f32, 2.0, 4.0, 0.5][usize::from(k)];
                let c = self.s.m.f32(factor.to_bits());
                f = self.s.m.bin(self.cur, Bin::Mul, f, c);
            }
        }
        if v.clamp {
            f = self.unary(7, f);
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a float result outside a VGPR");
        };
        let x = self.as_int(f);
        self.write(inst, Reg::V(reg), x, guard);
        if guard.is_none()
            && Self::sdwa_dst(inst).is_none()
            && let Some(new) = inst.write(Reg::V(reg))
        {
            self.floats.insert(new, f);
        }
        Ok(())
    }

    fn write_i(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.clamp || v.omod != 0 {
            return Err("integer output modifiers");
        }
        match v.dst {
            Some(Operand::Vector { reg, .. }) => self.write(inst, Reg::V(reg), x, guard),
            Some(Operand::Scalar { code, .. }) => self.write(inst, Reg::S(code), x, None),
            _ => return Err("a result without a register"),
        }
        Ok(())
    }

    /// Starts code only the lanes where `guard` holds run (a store under a
    /// lane guard): returns where it started and the join block.
    fn begin_guarded(&mut self, guard: Option<V>) -> Option<(usize, usize)> {
        let g = guard?;
        let (then, join) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(self.cur, Term::CondBr(g, then, join));
        let head = self.cur;
        self.cur = then;
        Some((head, join))
    }

    /// Ends guarded code; each (value inside, value outside) becomes a phi.
    fn end_guarded(&mut self, at: Option<(usize, usize)>, values: &[(V, V)]) -> Vec<V> {
        let Some((head, join)) = at else {
            return values.iter().map(|v| v.0).collect();
        };
        let then = self.cur;
        self.s.m.terminate(then, Term::Br(join));
        self.cur = join;
        values
            .iter()
            .map(|&(inside, outside)| {
                let ty = self.s.m.type_of(inside);
                let phi = self.s.m.phi(join, ty);
                self.s.m.set_phi(phi, vec![(inside, then), (outside, head)]);
                phi
            })
            .collect()
    }

    /// An image atomic DXIL has no operation for (`inc`, `dec`, `fcmpswap`,
    /// `fmin`, `fmax`), on the dword at `coords` of the 32-bit integer view
    /// `h`: a compare-and-swap loop from a first guess of 0, until the
    /// exchange finds the value the new one was made from (a null view reads
    /// 0, so it leaves at once). `vals`: the data (fcmpswap: new, compare).
    /// Returns the old value.
    fn atomic_loop(&mut self, h: V, coords: &[V], op: &str, vals: &[V]) -> V {
        let i32t = self.s.t.i32;
        let entry = self.cur;
        let (head, done) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(entry, Term::Br(head));
        self.cur = head;
        let old = self.s.m.phi(head, i32t);
        let (data, zero, one) = (vals[0], self.s.m.i32(0), self.s.m.i32(1));
        let new = match op {
            // old >= data ? 0 : old + 1
            "inc" => {
                let up = self.s.m.bin(head, Bin::Add, old, one);
                let wrap = self.s.m.cmp(head, pred::IUGE, old, data);
                self.s.m.select(head, wrap, zero, up)
            }
            // old == 0 || old > data ? data : old - 1
            "dec" => {
                let down = self.s.m.bin(head, Bin::Sub, old, one);
                let empty = self.s.m.cmp(head, pred::EQ, old, zero);
                let over = self.s.m.cmp(head, pred::IUGT, old, data);
                let reload = self.s.m.bin(head, Bin::Or, empty, over);
                self.s.m.select(head, reload, data, down)
            }
            // A float compare: -0 matches +0, NaN matches nothing.
            "fcmpswap" => {
                let (o, c) = (self.as_float(old), self.as_float(vals[1]));
                let eq = self.s.m.cmp(head, pred::OEQ, o, c);
                self.s.m.select(head, eq, data, old)
            }
            // As v_min_f32 and v_max_f32.
            _ => {
                let (d, o) = (self.as_float(data), self.as_float(old));
                let m = self.binary(if op == "fmax" { 35 } else { 36 }, d, o);
                self.as_int(m)
            }
        };
        let mut args = vec![h];
        args.extend_from_slice(coords);
        args.extend([old, new]);
        let prev = self.s.op(
            head,
            "atomicCompareExchange.i32",
            Attr::NoUnwind,
            i32t,
            79,
            &args,
        );
        let found = self.s.m.cmp(head, pred::EQ, prev, old);
        self.s.m.terminate(head, Term::CondBr(found, done, head));
        self.s.m.set_phi(old, vec![(zero, entry), (prev, head)]);
        self.cur = done;
        prev
    }

    /// Code that depends on the lane layout: wave operations, and in compute
    /// the console's wave size (lanes are thread ids in group order there).
    /// Pixel lanes match only within a quad.
    fn cross_lane(&mut self, quad: bool) -> Result<(), Why> {
        if self.s.mesh.is_some() {
            // shortcut: no wave size for mesh shaders in SM 6.6, so host
            // lanes cannot follow the console's; emulate through group-shared
            // memory if mesh programs turn out to need these.
            return Err("a cross-lane operation in a mesh shader");
        }
        if self.s.threads.is_none() && !(quad && self.s.pixel) {
            return Err("a cross-lane operation outside compute");
        }
        debug_assert!(self.s.threads.is_none() || self.console_lanes);
        self.s.flags |= flags::WAVE_OPS;
        Ok(())
    }

    /// Component `c` of the thread id in the group. With console lanes, the
    /// thread is relabelled so that its host lane is its console lane: the
    /// flat id is the wave's first lane's, rounded down to the wave size,
    /// plus the lane index. (A bijection while each host wave holds a
    /// wave-aligned run of flat ids, as the console's waves do.)
    fn thread_id(&mut self, c: u8) -> Val {
        if !self.console_lanes {
            return self.compute_id("threadIdInGroup.i32", 95, c);
        }
        let [x, y, _] = self.s.threads.expect("compute");
        let e = self.s.entry;
        let i32t = self.s.t.i32;
        let flat = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            i32t,
            96,
            &[],
        );
        let first = self.s.op(
            e,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[flat],
        );
        let wave = self.s.wave_size.expect("set with console lanes");
        let mask = self.s.m.i32(!(wave - 1));
        let base = self.s.m.bin(e, Bin::And, first, mask);
        let lane = self.lane_index();
        let f = self.s.m.bin(e, Bin::Add, base, lane);
        self.s.flags |= flags::WAVE_OPS;
        let (xk, yk, xy) = (self.s.m.i32(x), self.s.m.i32(y), self.s.m.i32(x * y));
        Val::I32(match c {
            0 => self.s.m.bin(e, Bin::URem, f, xk),
            1 => {
                let t = self.s.m.bin(e, Bin::UDiv, f, xk);
                self.s.m.bin(e, Bin::URem, t, yk)
            }
            _ => self.s.m.bin(e, Bin::UDiv, f, xy),
        })
    }

    /// Mesh: the counts and the primitives no lane exports. The console's
    /// counts come from one lane's allocation request, which DXIL cannot
    /// express (one call, dominating every output write).
    // shortcut: the counts are the declared maxima and each lane first culls
    // its primitive (all indices 0, zero area), so primitives past the
    // program's count are culled and vertices past it unused; this costs
    // the host the unused vertex and primitive slots. Pass the program's own
    // counts if that shows in frame time.
    fn mesh_prologue(&mut self, mesh: super::shader::Mesh) -> Result<(), Why> {
        let threads = self.s.threads.expect("mesh")[0];
        if mesh.topology != 2 {
            return Err("a mesh shader not drawing triangles");
        }
        if threads > 128 || mesh.max_vertices > 256 || mesh.max_primitives > 256 {
            return Err("a mesh shader beyond the host's limits");
        }
        let e = self.s.entry;
        let void = self.s.t.void;
        let (v, p) = (
            self.s.m.i32(mesh.max_vertices),
            self.s.m.i32(mesh.max_primitives),
        );
        self.s
            .op(e, "setMeshOutputCounts", Attr::NoUnwind, void, 168, &[v, p]);
        let flat = self.mesh_thread();
        let guard = (threads > mesh.max_primitives).then(|| self.s.m.cmp(e, pred::IULT, flat, p));
        let at = self.begin_guarded(guard);
        let z = self.s.m.i32(0);
        self.s.op(
            self.cur,
            "emitIndices",
            Attr::NoUnwind,
            void,
            169,
            &[flat, z, z, z],
        );
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// Mesh: the thread's index in the group, which is its vertex and
    /// primitive index.
    fn mesh_thread(&mut self) -> V {
        if let Some(f) = self.flat {
            return f;
        }
        let e = self.s.entry;
        let f = self.s.op(
            e,
            "flattenedThreadIdInGroup.i32",
            Attr::ReadNone,
            self.s.t.i32,
            96,
            &[],
        );
        self.flat = Some(f);
        f
    }

    /// The console wave size.
    fn wave(&self) -> u32 {
        if self.x.native.wave64 { 64 } else { 32 }
    }

    fn lane_index(&mut self) -> V {
        if self.s.mesh.is_some() {
            // The console's lane: mesh code reads no other lanes, so host
            // waves need not match the console's.
            let f = self.mesh_thread();
            let m = self.s.m.i32(self.wave() - 1);
            return self.s.m.bin(self.s.entry, Bin::And, f, m);
        }
        self.s.op(
            self.cur,
            "waveGetLaneIndex",
            Attr::ReadOnly,
            self.s.t.i32,
            111,
            &[],
        )
    }

    /// `x` of lane `src` (in this lane's quad) through quad reads, whose
    /// lane must be a constant.
    fn quad_pick(&mut self, x: V, src: V) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let three = self.s.m.i32(3);
        let q = self.s.m.bin(c, Bin::And, src, three);
        let mut r = x;
        for n in 0..4 {
            let kn = self.s.m.i32(n);
            let read = self.s.op(
                c,
                "quadReadLaneAt.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                122,
                &[x, kn],
            );
            let is = self.s.m.cmp(c, pred::EQ, q, kn);
            r = self.s.m.select(c, is, read, r);
        }
        r
    }

    fn helper_reads_own(&mut self, own: V, first: V) -> V {
        if !self.s.pixel {
            return first;
        }
        let c = self.cur;
        let helper = self
            .s
            .op(c, "isHelperLane.i1", Attr::ReadOnly, self.s.t.i1, 221, &[]);
        self.s.m.select(c, helper, own, first)
    }

    /// The x then y gradients (three each, undefined past `coords`) of
    /// float coordinates `coords` across this lane's quad: lane 1 less lane
    /// 0, lane 2 less lane 0, as the hardware's coarse derivatives.
    fn quad_gradients(&mut self, coords: &[V]) -> Vec<V> {
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let fu = self.s.m.undef(self.s.t.f32);
        let lane = self.lane_index();
        let (not3, one, two) = (self.s.m.i32(!3), self.s.m.i32(1), self.s.m.i32(2));
        let base = self.s.m.bin(c, Bin::And, lane, not3);
        let right = self.s.m.bin(c, Bin::Or, base, one);
        let below = self.s.m.bin(c, Bin::Or, base, two);
        let mut g = vec![fu; 6];
        for (k, &x) in coords.iter().enumerate().take(3) {
            let xi = self.as_int(x);
            let at = |l: &mut Self, lane: V| {
                let v = l.read_lane(xi, lane);
                l.as_float(v)
            };
            let (x0, x1, x2) = (at(self, base), at(self, right), at(self, below));
            g[k] = self.s.m.bin(c, Bin::Sub, x1, x0);
            g[3 + k] = self.s.m.bin(c, Bin::Sub, x2, x0);
        }
        g
    }

    fn read_lane(&mut self, x: V, lane: V) -> V {
        self.s.op(
            self.cur,
            "waveReadLaneAt.i32",
            Attr::NoUnwind,
            self.s.t.i32,
            117,
            &[x, lane],
        )
    }

    /// The float value of a register value.
    fn float_of(&mut self, v: Value) -> Result<V, Why> {
        if let Some(&f) = self.floats.get(&v) {
            self.record_read(v);
            return Ok(f);
        }
        let x = self.i32_of(v)?;
        Ok(self.as_float(x))
    }

    /// The values of VGPRs `reg..reg + n` as `inst` reads them.
    fn vgprs(&mut self, inst: &Inst, op: &Operand, n: u8) -> Result<Vec<V>, Why> {
        let Operand::Vector { reg, .. } = *op else {
            return Err("a vector operand");
        };
        (0..n)
            .map(|k| {
                let v = inst.read(Reg::V(reg + k)).ok_or("an unread operand")?;
                self.i32_of(v)
            })
            .collect()
    }

    /// LDS (thread-group shared memory) and `ds_swizzle`.
    fn ds(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Ds {
            dst,
            addr,
            data,
            offset0,
            offset1,
            gds,
        } = &inst.fields
        else {
            unreachable!()
        };
        if *gds {
            return Err("GDS");
        }
        if matches!(self.x.native.stage, HwStage::Es | HwStage::Gs) {
            return self.ring(inst, b, i);
        }
        let name = inst.name();
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let offset = u32::from(*offset0) | u32::from(*offset1) << 8;
        let dst_reg = match dst {
            Some(Operand::Vector { reg, .. }) => Some(*reg),
            _ => None,
        };
        if name == "ds_swizzle_b32" {
            // Bit 15 set: each lane of a quad picks a lane of it; else lanes
            // of each group of 32 combine their index with and/or/xor masks.
            let quad = offset & 0x8000 != 0
                || offset & 0x1C == 0x1C && offset & 0x380 == 0 && offset >> 10 & 0x1F < 4;
            // Mesh: host quads are threads 4n..4n+3 of a 1D group, as the
            // console's, so quad-local swizzles read through quad operations.
            let mesh_quad = self.s.mesh.is_some() && quad;
            if !mesh_quad {
                self.cross_lane(quad)?;
            }
            // The source sits in the address field.
            let x = self.vgprs(inst, addr.as_ref().ok_or("a swizzle source")?, 1)?[0];
            let lane = self.lane_index();
            let k = |l: &mut Self, v: u32| l.s.m.i32(v);
            let src = if offset & 0x8000 != 0 {
                let (three, one) = (k(self, 3), k(self, 1));
                let sel = self.s.m.bin(c, Bin::And, lane, three);
                let sh = self.s.m.bin(c, Bin::Shl, sel, one);
                let pattern = k(self, offset & 0xFF);
                let pick = self.s.m.bin(c, Bin::LShr, pattern, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, three);
                let not3 = k(self, !3);
                let base = self.s.m.bin(c, Bin::And, lane, not3);
                self.s.m.bin(c, Bin::Or, base, pick)
            } else {
                let and = k(self, offset & 0x1F | !0x1F);
                let or = k(self, offset >> 5 & 0x1F);
                let xor = k(self, offset >> 10 & 0x1F);
                let t = self.s.m.bin(c, Bin::And, lane, and);
                let t = self.s.m.bin(c, Bin::Or, t, or);
                self.s.m.bin(c, Bin::Xor, t, xor)
            };
            let r = if mesh_quad {
                self.quad_pick(x, src)
            } else {
                self.read_lane(x, src)
            };
            let reg = dst_reg.ok_or("a swizzle result")?;
            self.write(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        if self.s.lds_dwords == 0 {
            return Err("LDS the program does not allocate");
        }
        let a = self.vgprs(inst, addr.as_ref().ok_or("an LDS address")?, 1)?[0];
        // shortcut: LDS addresses are taken as dword-aligned and in bounds
        // (out-of-bounds console accesses read 0 and drop writes).
        let dword = |l: &mut Self, bytes: u32| -> V {
            let c = l.cur;
            let o = l.s.m.i32(bytes);
            let at = l.s.m.bin(c, Bin::Add, a, o);
            let two = l.s.m.i32(2);
            let index = l.s.m.bin(c, Bin::LShr, at, two);
            let lds = l.s.lds();
            l.s.m.element(c, lds, index)
        };
        let width = |n: &str| match n.rsplit('_').next() {
            Some("b64") => 2,
            Some("b96") => 3,
            Some("b128") => 4,
            _ => 1,
        };
        // (byte offset, dwords) per element; read2/write2 scale their two
        // offsets by the element size (×64 for st64).
        let pair = |n: &str| -> Vec<(u32, u32)> {
            let w = width(n);
            let scale = 4 * w * if n.contains("st64") { 64 } else { 1 };
            vec![
                (u32::from(*offset0) * scale, w),
                (u32::from(*offset1) * scale, w),
            ]
        };
        // Bytes and halves: the dword holding them, and their bit position
        // in it. A half at byte 3 straddles into the next dword, as native
        // RDNA2 reads it (an unaligned half on hardware with unaligned LDS
        // access); bytes never do.
        // `_d16` reads fill a register's low half (`_d16_hi` its high half)
        // and keep the other; `_d16_hi` writes take the data's high half.
        let base = name.trim_end_matches("_hi").trim_end_matches("_d16");
        let sub = match base {
            "ds_read_u8" | "ds_read_i8" | "ds_write_b8" => Some(8u32),
            "ds_read_u16" | "ds_read_i16" | "ds_write_b16" => Some(16),
            _ => None,
        };
        let sub = sub.filter(|_| {
            base == name
                || name.ends_with("_d16") && base.starts_with("ds_read")
                || name.ends_with("_d16_hi")
        });
        if let Some(bits) = sub {
            let p = dword(self, offset);
            let o = self.s.m.i32(offset);
            let at = self.s.m.bin(c, Bin::Add, a, o);
            let three = self.s.m.i32(3);
            let byte = self.s.m.bin(c, Bin::And, at, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            let mask = self.s.m.i32((1u32 << bits) - 1);
            // The next dword, for a half at byte 3 (the last dword stands in
            // for one past the end).
            let next = if bits == 16 {
                let two = self.s.m.i32(2);
                let index = self.s.m.bin(c, Bin::LShr, at, two);
                let one = self.s.m.i32(1);
                let n = self.s.m.bin(c, Bin::Add, index, one);
                let last = self.s.m.i32(self.s.lds_dwords - 1);
                let past = self.s.m.cmp(c, pred::IUGT, n, last);
                let n = self.s.m.select(c, past, last, n);
                let lds = self.s.lds();
                Some(self.s.m.element(c, lds, n))
            } else {
                None
            };
            let i64t = self.s.m.int(64);
            if name.starts_with("ds_read") {
                let reg = dst_reg.ok_or("an LDS read result")?;
                let w = self.s.m.load(c, p);
                let x = match next {
                    Some(q) => {
                        // Both dwords as one 64-bit value, shifted down.
                        self.s.flags |= flags::INT64;
                        let hi = self.s.m.load(c, q);
                        let lo = self.s.m.cast(c, Cast::ZExt, w, i64t);
                        let hi = self.s.m.cast(c, Cast::ZExt, hi, i64t);
                        let k32 = self.s.m.int_const(i64t, 32);
                        let hi = self.s.m.bin(c, Bin::Shl, hi, k32);
                        let both = self.s.m.bin(c, Bin::Or, lo, hi);
                        let sh64 = self.s.m.cast(c, Cast::ZExt, sh, i64t);
                        let x = self.s.m.bin(c, Bin::LShr, both, sh64);
                        self.s.m.cast(c, Cast::Trunc, x, self.s.t.i32)
                    }
                    None => self.s.m.bin(c, Bin::LShr, w, sh),
                };
                let x = if base.contains("_i") {
                    let k = self.s.m.i32(32 - bits);
                    let x = self.s.m.bin(c, Bin::Shl, x, k);
                    self.s.m.bin(c, Bin::AShr, x, k)
                } else {
                    self.s.m.bin(c, Bin::And, x, mask)
                };
                let x = if base == name {
                    x
                } else {
                    let old = match inst.read(Reg::V(reg)) {
                        Some(v) => self.i32_of(v)?,
                        None => self.s.m.i32(0),
                    };
                    let half = self.s.m.i32(0xFFFF);
                    let x = self.s.m.bin(c, Bin::And, x, half);
                    let (x, keep) = if name.ends_with("_hi") {
                        let k16 = self.s.m.i32(16);
                        (self.s.m.bin(c, Bin::Shl, x, k16), half)
                    } else {
                        (x, self.s.m.i32(0xFFFF_0000))
                    };
                    let old = self.s.m.bin(c, Bin::And, old, keep);
                    self.s.m.bin(c, Bin::Or, old, x)
                };
                self.write(inst, Reg::V(reg), x, guard);
            } else {
                // Clear, then set: two atomics, so lanes writing other
                // bytes of the dword keep theirs.
                let mut x = self.vgprs(inst, data.first().ok_or("an LDS write")?, 1)?[0];
                if name != base {
                    // The data's high half.
                    let k16 = self.s.m.i32(16);
                    x = self.s.m.bin(c, Bin::LShr, x, k16);
                }
                let x = self.s.m.bin(c, Bin::And, x, mask);
                let ones = self.s.m.i32(u32::MAX);
                // (dword, bits to clear, bits to set); a half at byte 3 also
                // writes the next dword's low byte.
                let mut parts = Vec::new();
                match next {
                    Some(q) => {
                        self.s.flags |= flags::INT64;
                        let sh64 = self.s.m.cast(c, Cast::ZExt, sh, i64t);
                        let k32 = self.s.m.int_const(i64t, 32);
                        let x64 = self.s.m.cast(c, Cast::ZExt, x, i64t);
                        let x64 = self.s.m.bin(c, Bin::Shl, x64, sh64);
                        let m64 = self.s.m.int_const(i64t, 0xFFFF);
                        let m64 = self.s.m.bin(c, Bin::Shl, m64, sh64);
                        for (ptr, shift) in [(p, false), (q, true)] {
                            let (xv, mv) = if shift {
                                (
                                    self.s.m.bin(c, Bin::LShr, x64, k32),
                                    self.s.m.bin(c, Bin::LShr, m64, k32),
                                )
                            } else {
                                (x64, m64)
                            };
                            let xv = self.s.m.cast(c, Cast::Trunc, xv, self.s.t.i32);
                            let mv = self.s.m.cast(c, Cast::Trunc, mv, self.s.t.i32);
                            let keep = self.s.m.bin(c, Bin::Xor, mv, ones);
                            parts.push((ptr, keep, xv));
                        }
                    }
                    None => {
                        let x = self.s.m.bin(c, Bin::Shl, x, sh);
                        let m = self.s.m.bin(c, Bin::Shl, mask, sh);
                        let keep = self.s.m.bin(c, Bin::Xor, m, ones);
                        parts.push((p, keep, x));
                    }
                }
                use super::bitcode::Rmw;
                let at = self.begin_guarded(guard);
                for (ptr, keep, x) in parts {
                    self.s.m.atomic(self.cur, Rmw::And, ptr, keep);
                    self.s.m.atomic(self.cur, Rmw::Or, ptr, x);
                }
                self.end_guarded(at, &[]);
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_read") {
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                return Err("an LDS read of bytes or halves");
            };
            let reg = dst_reg.ok_or("an LDS read result")?;
            let mut n = 0;
            for (off, w) in elements {
                for k in 0..w {
                    let p = dword(self, off + 4 * k);
                    let x = self.s.m.load(self.cur, p);
                    self.write(inst, Reg::V(reg + n), x, guard);
                    n += 1;
                }
            }
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("ds_write") {
            let elements = if rest.starts_with('2') {
                pair(name)
            } else if matches!(rest, "_b32" | "_b64" | "_b96" | "_b128") {
                vec![(offset, width(name))]
            } else {
                return Err("an LDS write of bytes or halves");
            };
            let mut stores = Vec::new();
            for (e, (off, w)) in elements.into_iter().enumerate() {
                let vals = self.vgprs(inst, data.get(e).ok_or("an LDS write")?, w as u8)?;
                for (k, x) in vals.into_iter().enumerate() {
                    stores.push((off + 4 * k as u32, x));
                }
            }
            let at = self.begin_guarded(guard);
            for (off, x) in stores {
                let p = dword(self, off);
                self.s.m.store(self.cur, p, x);
            }
            self.end_guarded(at, &[]);
            return Ok(());
        }
        // Atomics: the old value returns with `_rtn`.
        let op = name
            .strip_prefix("ds_")
            .map(|n| n.replace("_rtn", ""))
            .ok_or("an LDS operation")?;
        use super::bitcode::Rmw;
        let rmw = match op.as_str() {
            "add_u32" => Some(Rmw::Add),
            "sub_u32" => Some(Rmw::Sub),
            "and_b32" => Some(Rmw::And),
            "or_b32" => Some(Rmw::Or),
            "xor_b32" => Some(Rmw::Xor),
            "min_i32" => Some(Rmw::Min),
            "max_i32" => Some(Rmw::Max),
            "min_u32" => Some(Rmw::UMin),
            "max_u32" => Some(Rmw::UMax),
            "wrxchg_b32" => Some(Rmw::Xchg),
            "cmpst_b32" | "max_f32" | "min_f32" => None,
            _ => return Err("an LDS operation"),
        };
        let v0 = self.vgprs(inst, data.first().ok_or("an LDS atomic")?, 1)?[0];
        let v1 = match data.get(1) {
            Some(d) => Some(self.vgprs(inst, d, 1)?[0]),
            None => None,
        };
        let outside = match (guard, dst_reg.and_then(|r| inst.read(Reg::V(r)))) {
            (Some(_), Some(v)) => self.old_i32_of(Some(v), true, None),
            _ => self.s.m.i32(0),
        };
        let p = dword(self, offset);
        if let Some(max) = match op.as_str() {
            "max_f32" => Some(true),
            "min_f32" => Some(false),
            _ => None,
        } {
            let at = self.begin_guarded(guard);
            let r = self.lds_float_min_max(p, v0, max);
            let r = self.end_guarded(at, &[(r, outside)])[0];
            if let Some(reg) = dst_reg {
                self.write(inst, Reg::V(reg), r, None);
            }
            return Ok(());
        }
        let at = self.begin_guarded(guard);
        let r = match rmw {
            Some(op) => self.s.m.atomic(self.cur, op, p, v0),
            // Compare with data0, store data1.
            None => self
                .s
                .m
                .cmpxchg(self.cur, p, v0, v1.ok_or("an LDS compare")?),
        };
        let r = self.end_guarded(at, &[(r, outside)])[0];
        if let Some(reg) = dst_reg {
            self.write(inst, Reg::V(reg), r, None);
        }
        Ok(())
    }

    /// `ds_max_f32` (`max`) or `ds_min_f32` on the LDS dword at `p` with
    /// the float bits `data`; the old bits return.
    ///
    /// DXIL has no float atomics on group-shared memory, so this is a
    /// compare-exchange loop around [`Lower::lds_float_pick`], which gives
    /// what RDNA2 writes (measured on gfx1036, reported on the pull
    /// request): an sNaN, data first, then memory, is written quieted;
    /// otherwise a NaN loses to a number (two quiet NaNs give the data);
    /// numbers compare as floats with -0 below +0.
    // shortcut: f32 denormals compare as kept; with the console's MODE
    // flushing them they would order as ±0 first (MODE is not modelled).
    fn lds_float_min_max(&mut self, p: V, data: V, max: bool) -> V {
        let pre = self.cur;
        let init = self.s.m.load(pre, p);
        let (lp, exit) = (self.s.m.new_block(), self.s.m.new_block());
        self.s.m.terminate(pre, Term::Br(lp));
        let i32t = self.s.t.i32;
        let old = self.s.m.phi(lp, i32t);
        self.cur = lp;
        let new = self.lds_float_pick(old, data, max);
        let prev = self.s.m.cmpxchg(lp, p, old, new);
        let done = self.s.m.cmp(lp, pred::EQ, prev, old);
        self.s.m.set_phi(old, vec![(init, pre), (prev, lp)]);
        self.s.m.terminate(lp, Term::CondBr(done, exit, lp));
        self.cur = exit;
        old
    }

    /// What `ds_max_f32` (`max`) or `ds_min_f32` writes over the dword
    /// `mem` with `data` (float bits; see [`Lower::lds_float_min_max`]).
    fn lds_float_pick(&mut self, mem: V, data: V, max: bool) -> V {
        let c = self.cur;
        let k = |l: &mut Self, v: u32| l.s.m.i32(v);
        let (abs_mask, inf, quiet, zero) = (
            k(self, 0x7FFF_FFFF),
            k(self, 0x7F80_0000),
            k(self, 0x40_0000),
            k(self, 0),
        );
        let nan = |l: &mut Self, x: V| {
            let m = l.s.m.bin(c, Bin::And, x, abs_mask);
            l.s.m.cmp(c, pred::IUGT, m, inf)
        };
        let snan = |l: &mut Self, x: V| {
            let n = nan(l, x);
            let q = l.s.m.bin(c, Bin::And, x, quiet);
            let sig = l.s.m.cmp(c, pred::EQ, q, zero);
            l.s.m.bin(c, Bin::And, n, sig)
        };
        // A total order of the numbers as signed integers, -0 below +0:
        // negative floats' magnitude bits reversed.
        let key = |l: &mut Self, x: V| {
            let k31 = l.s.m.i32(31);
            let sign = l.s.m.bin(c, Bin::AShr, x, k31);
            let flip = l.s.m.bin(c, Bin::And, sign, abs_mask);
            l.s.m.bin(c, Bin::Xor, x, flip)
        };
        let (dk, mk) = (key(self, data), key(self, mem));
        let take = self
            .s
            .m
            .cmp(c, if max { pred::SGT } else { pred::SLT }, dk, mk);
        let r = self.s.m.select(c, take, data, mem);
        let (dn, mn) = (nan(self, data), nan(self, mem));
        // One NaN: the number; two quiet NaNs: the data.
        let r = self.s.m.select(c, mn, data, r);
        let both = self.s.m.bin(c, Bin::And, dn, mn);
        let only_data = {
            let t = self.s.m.bool(true);
            let not_both = self.s.m.bin(c, Bin::Xor, both, t);
            self.s.m.bin(c, Bin::And, dn, not_both)
        };
        let r = self.s.m.select(c, only_data, mem, r);
        // Signalling NaNs win, quieted: the data's first.
        let qm = self.s.m.bin(c, Bin::Or, mem, quiet);
        let sm = snan(self, mem);
        let r = self.s.m.select(c, sm, qm, r);
        let qd = self.s.m.bin(c, Bin::Or, data, quiet);
        let sd = snan(self, data);
        self.s.m.select(c, sd, qd, r)
    }

    // ---- instructions ----
    // ---- instructions ----

    fn inst(&mut self, b: usize, i: usize) -> Result<(), ShaderError> {
        let inst: &'a Inst = &self.x.p.blocks[b].insts[i];
        let offset = inst.machine.offset;
        let name = inst.name();
        self.cur_reads.clear();
        self.cur_old_deps.clear();
        self.cur_edge = None;
        self.mask_ok = match self.mask_data {
            MaskData::Console => true,
            MaskData::Host => {
                name.starts_with("s_bcnt1_i32_")
                    || name.starts_with("s_ff1_i32_")
                    || name.starts_with("v_mbcnt_")
                    || matches!(
                        name,
                        "s_cmp_eq_u32" | "s_cmp_lg_u32" | "s_cmp_eq_u64" | "s_cmp_lg_u64"
                    )
            }
            MaskData::None => false,
        };
        let effect = matches!(inst.fields, Fields::Exp { .. })
            || matches!(inst.fields, Fields::Flat { .. }) && !name.starts_with("global_load_")
            || name.contains("store")
            || name.contains("atomic")
            || name == "s_barrier"
            || matches!(inst.fields, Fields::Ds { .. })
                && !(name.starts_with("ds_read")
                    || name.starts_with("ds_swizzle")
                    || name.contains("permute"));
        if let Fields::Sopp { imm } = inst.fields
            && name == "s_sendmsg"
            && self.s.gs.is_some()
        {
            return self.gs_message(imm).map_err(|w| unsupported(offset, w));
        }
        let r = match &inst.fields {
            Fields::Sop { .. } => self.sop(inst),
            Fields::Sopp { .. } if name == "s_barrier" && self.no_barrier.contains(&b) => Ok(()),
            Fields::Sopp { .. } if name == "s_barrier" => {
                // Thread-group sync with a shared-memory fence, and a UAV
                // fence when the program writes UAVs (the waits before an
                // s_barrier make those writes visible on the console).
                let mode = if self.writes_uavs { 1 | 2 | 8 } else { 1 | 8 };
                let mode = self.s.m.i32(mode);
                let void = self.s.t.void;
                self.s
                    .op(self.cur, "barrier", Attr::NoDuplicate, void, 80, &[mode]);
                Ok(())
            }
            Fields::Sopp { .. } => Ok(()),
            Fields::Smem { .. } => self.smem(inst, (b, i)),
            Fields::Valu(_) if name.starts_with("v_interp_") => self.interp(inst, (b, i)),
            Fields::Valu(_) => self.valu(inst, b, i),
            Fields::Vintrp { .. } => self.interp(inst, (b, i)),
            Fields::Image { .. } => self.image(inst, b, i),
            Fields::Buffer { .. } => self.buffer(inst, b, i),
            Fields::Exp { .. } => self.export(b, i),
            Fields::Ds { .. } => self.ds(inst, b, i),
            Fields::Flat { .. } if name.starts_with("global_") => self.flat(inst, b, i),
            Fields::Flat { .. } => Err("flat memory"),
        };
        let reads = std::mem::take(&mut self.cur_reads);
        let old_deps = std::mem::take(&mut self.cur_old_deps);
        match r {
            Ok(()) => {
                let guard = self.x.st.guards[b][i];
                if !reads.is_empty() {
                    if effect {
                        self.sinks.push(SinkRead {
                            block: b,
                            guard,
                            offset,
                            deps: reads,
                        });
                    } else {
                        let use_guard = matches!(
                            &inst.fields,
                            Fields::Valu(v)
                                if v.dpp.is_none()
                                    && !matches!(
                                        name,
                                        "v_readfirstlane_b32"
                                            | "v_readlane_b32"
                                            | "v_writelane_b32"
                                    )
                        )
                        .then_some(guard)
                        .flatten();
                        for &(_, w) in &inst.writes {
                            if matches!(self.vals.get(&w), Some(Val::I32(_) | Val::Bit(_)))
                                || self.lane_bits.contains_key(&w)
                            {
                                self.record_write_deps(w, b, use_guard, false, None, reads.clone());
                            }
                        }
                    }
                }
                for old in old_deps {
                    for &(_, w) in &inst.writes {
                        if !matches!(self.vals.get(&w), Some(Val::I32(_) | Val::Bit(_))) {
                            continue;
                        }
                        if let Some(why) = old.direct_why {
                            self.taint_candidates.insert(w);
                            self.direct_missing
                                .push((w, why, b, guard, old.invert_guard));
                        }
                        if let Some(u) = old.value {
                            self.record_write_deps(
                                w,
                                b,
                                guard,
                                old.invert_guard,
                                old.ignore_why,
                                vec![ReadDep::Value(u)],
                            );
                        }
                    }
                }
                Ok(())
            }
            Err(why) if effect => Err(unsupported(offset, why)),
            Err(why) => {
                for &(_, v) in &inst.writes {
                    self.vals.insert(v, Val::Missing(why));
                }
                Ok(())
            }
        }
    }

    fn is_mask(&self, v: Option<Value>) -> bool {
        v.is_some_and(|v| self.x.st.masks.contains_key(&v))
    }

    fn sop(&mut self, inst: &Inst) -> Result<(), Why> {
        let Fields::Sop { dst, src, imm } = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let scc = inst.write(Reg::Scc);
        let dst_code = match dst {
            Some(Operand::Scalar { code, .. }) => Some(*code),
            _ => None,
        };
        let lo = dst_code.and_then(|c| inst.write(Reg::S(c)));
        if inst.write(Reg::S(EXEC)).is_some() || self.is_mask(lo) {
            return Err(LANE_MASK);
        }
        if matches!(
            name,
            "s_setreg_imm32_b32" | "s_setreg_b32" | "s_swappc_b64" | "s_version"
        ) || name.starts_with("s_waitcnt")
        {
            // shortcut: the float MODE register is not modelled; s_swappc_b64 is the
            // vertex fetch call, whose results arrive as inputs.
            return Err("a register the emitter does not model");
        }
        let c = self.cur;
        let src32 = |l: &mut Self, k: usize| -> Result<V, Why> {
            // SOPK arithmetic reads its destination as the first source.
            if k == 0 && matches!(name, "s_addk_i32" | "s_mulk_i32") {
                let d = dst.as_ref().ok_or("a missing operand")?;
                return l.operand(inst, d);
            }
            let op = src.get(k).ok_or("a missing operand")?;
            l.operand(inst, op)
        };
        let sext16 = |l: &mut Self| l.s.m.i32(i32::from(imm.unwrap_or(0) as i16) as u32);
        // Dword `h` of 64-bit source `k` (inline constants sign-extend).
        let half = |l: &mut Self, k: usize, h: u8| -> Result<V, Why> {
            match src.get(k) {
                Some(&Operand::Scalar { code, .. }) if code != NULL => {
                    let v = inst.read(Reg::S(code + h)).ok_or("an unread operand")?;
                    l.i32_of(v)
                }
                Some(op) if h == 0 => l.operand(inst, op),
                Some(Operand::Const(k)) => {
                    let x = crate::resource::inline(*k).ok_or("a special operand")?;
                    Ok(l.s.m.i32(if (x as i32) < 0 { u32::MAX } else { 0 }))
                }
                _ => Ok(l.s.m.i32(0)),
            }
        };
        if let Some(op) = name.strip_prefix("s_bitcmp") {
            // SCC is whether bit src1 (mod 32, or mod 64 for the 64-bit form)
            // of src0 is clear (bitcmp0) or set (bitcmp1).
            let (bit, k31) = (src32(self, 1)?, self.s.m.i32(31));
            let idx = self.s.m.bin(c, Bin::And, bit, k31);
            let word = if op.ends_with("_b64") {
                let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                let k32 = self.s.m.i32(32);
                let upper = self.s.m.bin(c, Bin::And, bit, k32);
                let z = self.s.m.i32(0);
                let upper = self.s.m.cmp(c, pred::NE, upper, z);
                self.s.m.select(c, upper, b, a)
            } else {
                src32(self, 0)?
            };
            let shifted = self.s.m.bin(c, Bin::LShr, word, idx);
            let one = self.s.m.i32(1);
            let tested = self.s.m.bin(c, Bin::And, shifted, one);
            let want = self.s.m.i32(u32::from(op.starts_with('1')));
            let r = self.s.m.cmp(c, pred::EQ, tested, want);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").filter(|o| o.ends_with("_u64")) {
            // eq and lg only: both dwords.
            let mut r = None;
            for h in 0..2 {
                let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                let e = self.s.m.cmp(c, pred::EQ, a, b);
                r = Some(match r {
                    Some(x) => self.s.m.bin(c, Bin::And, x, e),
                    None => e,
                });
            }
            let mut r = r.expect("two dwords");
            if op.starts_with("lg") {
                let t = self.s.m.bool(true);
                r = self.s.m.bin(c, Bin::Xor, r, t);
            }
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        if let Some(op) = name.strip_prefix("s_cmp_").or(name.strip_prefix("s_cmpk_")) {
            let (a, b) = if name.starts_with("s_cmpk_") {
                let b = if op.ends_with("_i32") {
                    sext16(self)
                } else {
                    self.s.m.i32(u32::from(imm.unwrap_or(0)))
                };
                (src32(self, 0)?, b)
            } else {
                if op.ends_with("64") {
                    return Err("a 64-bit scalar compare");
                }
                (src32(self, 0)?, src32(self, 1)?)
            };
            let signed = op.ends_with("_i32");
            let p = match &op[..2] {
                "eq" => pred::EQ,
                "lg" => pred::NE,
                "gt" if signed => pred::SGT,
                "ge" if signed => pred::SGE,
                "lt" if signed => pred::SLT,
                "le" if signed => pred::SLE,
                "gt" => pred::IUGT,
                "ge" => pred::IUGE,
                "lt" => pred::IULT,
                "le" => pred::IULE,
                _ => return Err("a scalar compare"),
            };
            let r = self.s.m.cmp(c, p, a, b);
            self.set(scc, Val::Bit(r));
            return Ok(());
        }
        let wide = name.ends_with("_b64") || name.ends_with("_u64") || name.ends_with("_i64");
        if wide {
            let hi = dst_code.and_then(|c| inst.write(Reg::S(c + 1)));
            let nonzero = |l: &mut Self, a: V, b: V| {
                let o = l.s.m.bin(c, Bin::Or, a, b);
                let z = l.s.m.i32(0);
                l.s.m.cmp(c, pred::NE, o, z)
            };
            return match name {
                "s_bcnt1_i32_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let i32t = self.s.t.i32;
                    let na = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[a]);
                    let nb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 31, &[b]);
                    let r = self.s.m.bin(c, Bin::Add, na, nb);
                    self.set(lo, Val::I32(r));
                    let z = self.s.m.i32(0);
                    let s = self.s.m.cmp(c, pred::NE, r, z);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_ff1_i32_b64" => {
                    // The lowest set bit; -1 for none (as DXIL's first bit low).
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let i32t = self.s.t.i32;
                    let fa = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[a]);
                    let fb = self
                        .s
                        .op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[b]);
                    let (z, k32) = (self.s.m.i32(0), self.s.m.i32(32));
                    let fb32 = self.s.m.bin(c, Bin::Add, fb, k32);
                    let none = self.s.m.cmp(c, pred::EQ, b, z);
                    let fb = self.s.m.select(c, none, fb, fb32);
                    let low = self.s.m.cmp(c, pred::NE, a, z);
                    let r = self.s.m.select(c, low, fa, fb);
                    self.set(lo, Val::I32(r));
                    Ok(())
                }
                "s_and_b64" | "s_or_b64" | "s_xor_b64" | "s_andn2_b64" | "s_orn2_b64"
                | "s_nand_b64" | "s_nor_b64" | "s_xnor_b64" | "s_not_b64" => {
                    let ones = self.s.m.i32(u32::MAX);
                    let mut r = Vec::new();
                    for h in 0..2u8 {
                        let a = half(self, 0, h)?;
                        let x = if name == "s_not_b64" {
                            self.s.m.bin(c, Bin::Xor, a, ones)
                        } else {
                            let mut b = half(self, 1, h)?;
                            if name.contains("n2") {
                                b = self.s.m.bin(c, Bin::Xor, b, ones);
                            }
                            let op = match &name[2..5] {
                                "and" | "nan" => Bin::And,
                                "or_" | "orn" | "nor" => Bin::Or,
                                _ => Bin::Xor,
                            };
                            let x = self.s.m.bin(c, op, a, b);
                            if matches!(name, "s_nand_b64" | "s_nor_b64" | "s_xnor_b64") {
                                self.s.m.bin(c, Bin::Xor, x, ones)
                            } else {
                                x
                            }
                        };
                        r.push(x);
                    }
                    self.set(lo, Val::I32(r[0]));
                    self.set(hi, Val::I32(r[1]));
                    let s = nonzero(self, r[0], r[1]);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_lshl_b64" | "s_lshr_b64" | "s_ashr_i64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    let n = src32(self, 1)?;
                    let i64t = self.s.m.int(64);
                    self.s.flags |= flags::INT64;
                    let a = self.s.m.cast(c, Cast::ZExt, a, i64t);
                    let b = self.s.m.cast(c, Cast::ZExt, b, i64t);
                    let k32 = self.s.m.int_const(i64t, 32);
                    let b = self.s.m.bin(c, Bin::Shl, b, k32);
                    let x = self.s.m.bin(c, Bin::Or, a, b);
                    let k63 = self.s.m.i32(63);
                    let n = self.s.m.bin(c, Bin::And, n, k63);
                    let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                    let op = match name {
                        "s_lshl_b64" => Bin::Shl,
                        "s_lshr_b64" => Bin::LShr,
                        _ => Bin::AShr,
                    };
                    let r = self.s.m.bin(c, op, x, n);
                    let rl = self.s.m.cast(c, Cast::Trunc, r, self.s.t.i32);
                    let rh = self.s.m.bin(c, Bin::LShr, r, k32);
                    let rh = self.s.m.cast(c, Cast::Trunc, rh, self.s.t.i32);
                    self.set(lo, Val::I32(rl));
                    self.set(hi, Val::I32(rh));
                    let s = nonzero(self, rl, rh);
                    self.set(scc, Val::Bit(s));
                    Ok(())
                }
                "s_mov_b64" => {
                    let (a, b) = (half(self, 0, 0)?, half(self, 0, 1)?);
                    self.set(lo, Val::I32(a));
                    self.set(hi, Val::I32(b));
                    Ok(())
                }
                "s_cselect_b64" => {
                    let cond = self.scc_in(inst)?;
                    for (h, v) in [(0, lo), (1, hi)] {
                        let (a, b) = (half(self, 0, h)?, half(self, 1, h)?);
                        let r = self.s.m.select(c, cond, a, b);
                        self.set(v, Val::I32(r));
                    }
                    Ok(())
                }
                _ => Err(name),
            };
        }
        let m = &mut self.s.m;
        let _ = m;
        let (r, carry): (V, Option<V>) = match name {
            "s_mov_b32" => (src32(self, 0)?, None),
            "s_movk_i32" => (sext16(self), None),
            "s_cselect_b32" => {
                let cond = self.scc_in(inst)?;
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.s.m.select(c, cond, a, b), None)
            }
            "s_brev_b32" => {
                let a = src32(self, 0)?;
                let r = self
                    .s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a]);
                (r, None)
            }
            "s_not_b32" => {
                let a = src32(self, 0)?;
                let ones = self.s.m.i32(u32::MAX);
                (self.s.m.bin(c, Bin::Xor, a, ones), None)
            }
            "s_add_u32" | "s_add_i32" | "s_addk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_addk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                let r = self.s.m.bin(c, Bin::Add, a, b);
                let carry = if name == "s_add_u32" {
                    Some(self.s.m.cmp(c, pred::IULT, r, a))
                } else {
                    None
                };
                (r, carry)
            }
            "s_sub_u32" | "s_sub_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let r = self.s.m.bin(c, Bin::Sub, a, b);
                let carry = (name == "s_sub_u32").then(|| self.s.m.cmp(c, pred::IULT, a, b));
                (r, carry)
            }
            "s_addc_u32" | "s_subb_u32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let cin = self.scc_in(inst)?;
                let (r, carry) = self.add_sub_carry(c, name == "s_addc_u32", a, b, Some(cin));
                (r, Some(carry))
            }
            "s_mul_i32" | "s_mulk_i32" => {
                let a = src32(self, 0)?;
                let b = if name == "s_mulk_i32" {
                    sext16(self)
                } else {
                    src32(self, 1)?
                };
                (self.s.m.bin(c, Bin::Mul, a, b), None)
            }
            "s_mul_hi_u32" | "s_mul_hi_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                (self.mul_hi(a, b, name.ends_with("i32")), None)
            }
            "s_flbit_i32_b32" | "s_flbit_i32" => {
                // The first bit from the top (differing from the sign for
                // the signed form), -1 for none: DXIL's first-bit-high.
                let a = src32(self, 0)?;
                let op = if name == "s_flbit_i32_b32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_ff1_i32_b32" | "s_bcnt1_i32_b32" => {
                // First set bit (-1 for zero, as DXIL's), or the bit count.
                let a = src32(self, 0)?;
                let op = if name == "s_ff1_i32_b32" { 32 } else { 31 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                (r, None)
            }
            "s_bfe_u32" | "s_bfe_i32" => {
                // src1: offset in bits 0-4, width in bits 16-22.
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let (k16, k127) = (self.s.m.i32(16), self.s.m.i32(0x7F));
                let w = self.s.m.bin(c, Bin::LShr, b, k16);
                let w = self.s.m.bin(c, Bin::And, w, k127);
                (self.bfe(a, b, w, name == "s_bfe_i32"), None)
            }
            "s_and_b32" | "s_or_b32" | "s_xor_b32" | "s_andn2_b32" | "s_orn2_b32"
            | "s_nand_b32" | "s_nor_b32" | "s_xnor_b32" => {
                let (a, mut b) = (src32(self, 0)?, src32(self, 1)?);
                let ones = self.s.m.i32(u32::MAX);
                if name.contains("n2") {
                    b = self.s.m.bin(c, Bin::Xor, b, ones);
                }
                let op = match &name[2..5] {
                    "and" | "nan" => Bin::And,
                    "or_" | "orn" | "nor" => Bin::Or,
                    _ => Bin::Xor,
                };
                let mut r = self.s.m.bin(c, op, a, b);
                if matches!(name, "s_nand_b32" | "s_nor_b32" | "s_xnor_b32") {
                    r = self.s.m.bin(c, Bin::Xor, r, ones);
                }
                (r, None)
            }
            "s_lshl_b32" | "s_lshr_b32" | "s_ashr_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let op = match name {
                    "s_lshl_b32" => Bin::Shl,
                    "s_lshr_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                (self.s.m.bin(c, op, a, b), None)
            }
            "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k = self.s.m.i32(u32::from(name.as_bytes()[6] - b'0'));
                let t = self.s.m.bin(c, Bin::Shl, a, k);
                (self.s.m.bin(c, Bin::Add, t, b), None)
            }
            "s_min_u32" | "s_max_u32" | "s_min_i32" | "s_max_i32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let p = match name {
                    "s_min_u32" => pred::IULT,
                    "s_max_u32" => pred::IUGT,
                    "s_min_i32" => pred::SLT,
                    _ => pred::SGT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                (self.s.m.select(c, t, a, b), None)
            }
            "s_abs_i32" => {
                let a = src32(self, 0)?;
                let z = self.s.m.i32(0);
                let neg = self.s.m.cmp(c, pred::SLT, a, z);
                let n = self.s.m.bin(c, Bin::Sub, z, a);
                (self.s.m.select(c, neg, n, a), None)
            }
            "s_bitset0_b32" | "s_bitset1_b32" => {
                // Sets (clears) bit src[4:0] of the destination it reads.
                let old = self.operand(inst, dst.as_ref().ok_or("a missing operand")?)?;
                let a = src32(self, 0)?;
                let (k31, one) = (self.s.m.i32(31), self.s.m.i32(1));
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let bit = self.s.m.bin(c, Bin::Shl, one, a);
                if name == "s_bitset1_b32" {
                    (self.s.m.bin(c, Bin::Or, old, bit), None)
                } else {
                    let ones = self.s.m.i32(u32::MAX);
                    let nb = self.s.m.bin(c, Bin::Xor, bit, ones);
                    (self.s.m.bin(c, Bin::And, old, nb), None)
                }
            }
            "s_bfm_b32" => {
                let (a, b) = (src32(self, 0)?, src32(self, 1)?);
                let k31 = self.s.m.i32(31);
                let one = self.s.m.i32(1);
                let a = self.s.m.bin(c, Bin::And, a, k31);
                let b = self.s.m.bin(c, Bin::And, b, k31);
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                (self.s.m.bin(c, Bin::Shl, m, b), None)
            }
            _ => return Err(name),
        };
        self.set(lo, Val::I32(r));
        if let Some(scc) = scc {
            let val = match carry {
                Some(k) => Val::Bit(k),
                None if matches!(
                    name,
                    "s_mov_b32"
                        | "s_movk_i32"
                        | "s_cselect_b32"
                        | "s_brev_b32"
                        | "s_mul_i32"
                        | "s_mulk_i32"
                ) =>
                {
                    return Ok(());
                }
                None if name.starts_with("s_add") || name.starts_with("s_sub") => {
                    Val::Missing("the carry of a scalar operation")
                }
                None => {
                    let z = self.s.m.i32(0);
                    Val::Bit(self.s.m.cmp(c, pred::NE, r, z))
                }
            };
            self.vals.insert(scc, val);
        }
        Ok(())
    }

    /// `a + b (+ cin)` or `a - b (- cin)` with the carry (borrow) out, as the
    /// scalar and vector add-with-carry instructions set it.
    fn add_sub_carry(&mut self, c: usize, add: bool, a: V, b: V, cin: Option<V>) -> (V, V) {
        let op = if add { Bin::Add } else { Bin::Sub };
        let mut r = self.s.m.bin(c, op, a, b);
        // Without a carry in: a + b wraps below a; a - b borrows when a < b.
        let mut carry = if add {
            self.s.m.cmp(c, pred::IULT, r, a)
        } else {
            self.s.m.cmp(c, pred::IULT, a, b)
        };
        if let Some(cin) = cin {
            let one = self.s.m.cast(c, Cast::ZExt, cin, self.s.t.i32);
            let t = r;
            r = self.s.m.bin(c, op, t, one);
            // The carry in carries on when the partial result is all ones
            // (add) or zero (sub).
            let edge = self.s.m.i32(if add { u32::MAX } else { 0 });
            let at_edge = self.s.m.cmp(c, pred::EQ, t, edge);
            let more = self.s.m.bin(c, Bin::And, cin, at_edge);
            carry = self.s.m.bin(c, Bin::Or, carry, more);
        }
        (r, carry)
    }

    fn scc_in(&mut self, inst: &Inst) -> Result<V, Why> {
        let v = inst.read(Reg::Scc).ok_or("an unread SCC")?;
        match self.get(v)? {
            Val::Bit(x) => Ok(x),
            _ => Err("SCC without a value"),
        }
    }

    fn smem(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Fields::Smem {
            data,
            offset,
            soffset,
            cache,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if let Some(a) = self.scalar.get(&site).copied().filter(|a| a.param == CODE) {
            return self.code_smem(inst, a);
        }
        let Some(rest) = name.strip_prefix("s_buffer_load_dword") else {
            if let Some(a) = self.scalar.get(&site).copied()
                && self
                    .x
                    .native
                    .layout
                    .get(a.param)
                    .is_some_and(|p| p.root == RootType::Descriptor)
                && !self.loads_descriptor(site)
            {
                return self.root_view_smem(inst, a, site);
            }
            return self.descriptor_data(inst, site);
        };
        let count = match rest {
            "" => 1,
            "x2" => 2,
            "x4" => 4,
            "x8" => 8,
            "x16" => 16,
            _ => return Err("a scalar buffer load"),
        };
        let Some(Operand::Scalar { code, .. }) = *data else {
            return Err("a scalar buffer load");
        };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked descriptor")?;
        let kind = self.used_kind(src);
        let c = self.cur;
        let (base, extra) = self.built_offset(src)?;
        let offset = &((*offset as u32).wrapping_add(base) as i32);
        let mut addr = self.s.m.i32(*offset as u32);
        if let Some(x) = extra {
            addr = self.s.m.bin(c, Bin::Add, addr, x);
        }
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(inst, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        if matches!(src, Source::Built { address, .. } if address.param == CODE) {
            // A descriptor built around a table in the code.
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let x = self.code_dword(a)?;
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        let uav = self.is_uav(src);
        if uav || matches!(kind, Some(u) if u.kind == ResourceType::Srv) {
            let u = uav_bits(uav, cache.glc);
            let stride = structured_stride(kind);
            let props = match stride {
                Some(s) => (12 | u, s),
                None => (11 | u, 0),
            };
            let h = self.handle(site, src, false, props)?;
            let stride_v = stride.map(|s| self.s.m.i32(s));
            let (iu, mask, al) = (
                self.s.m.undef(self.s.t.i32),
                self.s.m.i8(1),
                self.s.m.i32(4),
            );
            for k in 0..count {
                let off = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, addr, off);
                let (idx, eoff) = match stride_v {
                    Some(sv) => (
                        self.s.m.bin(c, Bin::UDiv, a, sv),
                        self.s.m.bin(c, Bin::URem, a, sv),
                    ),
                    None => (a, iu),
                };
                let r = self.s.op(
                    c,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    139,
                    &[h, idx, eoff, mask, al],
                );
                let x = self.s.m.extract(c, r, 0);
                let x = self.as_int(x);
                self.write(inst, Reg::S(code + k as u8), x, None);
            }
            return Ok(());
        }
        let h = self.handle(site, src, false, (13, 0x1_0000))?;
        let constant = match addr {
            V::Const(_)
                if matches!(soffset, Operand::Scalar { code: NULL, .. }) && extra.is_none() =>
            {
                Some(*offset as u32)
            }
            _ => None,
        };
        let mut rows: HashMap<u32, V> = HashMap::new();
        for k in 0..count {
            let x = match constant {
                Some(base) => {
                    let byte = base + 4 * k;
                    let row = match rows.get(&(byte >> 4)) {
                        Some(&r) => r,
                        None => {
                            let ri = self.s.m.i32(byte >> 4);
                            let r = self.s.op(
                                c,
                                "cbufferLoadLegacy.i32",
                                Attr::ReadOnly,
                                self.s.t.cbuf_ret,
                                59,
                                &[h, ri],
                            );
                            rows.insert(byte >> 4, r);
                            r
                        }
                    };
                    self.s.m.extract(c, row, (byte >> 2) & 3)
                }
                None => {
                    let off = self.s.m.i32(4 * k);
                    let a = self.s.m.bin(c, Bin::Add, addr, off);
                    self.cbuffer_dword(h, a)
                }
            };
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// A descriptor load (`s_load_dword*` from a table or heap) whose dwords
    /// the code also reads as numbers: the console compiler computes sizes
    /// from descriptor bits (an image's width, height, depth, levels; a
    /// buffer's record count) instead of querying them. Those fields are
    /// rebuilt from the host view's dimensions in the GFX10 layout (image
    /// dword 1 bits 30-31 and dword 2 bits 0-13 width - 1, dword 2 bits
    /// 14-29 height - 1, dword 3 identity swizzle, last level at 16, type at
    /// 28, dword 4 depth or layers - 1; buffer dword 2 the record count).
    // shortcut: the base level reads 0 (the host view starts at its own
    // level 0, so sizes the code shifts by it come out right) and the other
    // dwords (addresses, formats, tiling) have no host value: reading them
    // stops the program.
    fn descriptor_data(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        const OTHER: Why = DESCRIPTOR_OTHER;
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar memory read as data");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar memory read as data"),
        };
        // How the program uses the descriptor tells the view's kind.
        // Prefer an image or vector-buffer use over `s_buffer_load_dword*`,
        // since those encode typed/structured properties directly; a
        // waterfall loop that only sizes a lane's entry falls back to a
        // load of the same entry that an access uses (see `resource::Waterfall`),
        // or any sibling load of the same heap entry (`same_entry_use`).
        let use_of = |of: Site| {
            let mut uses = self
                .x
                .res
                .descriptors
                .iter()
                .filter(|d| d.role == Role::Resource)
                .filter_map(|d| {
                    d.source.leaves().into_iter().find_map(|l| match *l {
                        Source::Loaded { load, first } if load == of => Some((d.site, first)),
                        _ => None,
                    })
                });
            let mut fallback = None;
            loop {
                let Some((at, first)) = uses.next() else {
                    break fallback;
                };
                match &self.x.p.blocks[at.0].insts[at.1].fields {
                    Fields::Image { .. } | Fields::Buffer { .. } => break Some((at, first)),
                    _ => {
                        fallback.get_or_insert((at, first));
                    }
                }
            }
        };
        let use_at = use_of(site)
            .or_else(|| {
                let w = &self.x.res.waterfall;
                let &(_, twin) = w.sizes.iter().find(|s| s.0 == site)?;
                use_of(twin)
            })
            .or_else(|| self.same_entry_use(site));
        let (src, uav, props, image) = if let Some((at, first)) = use_at {
            let src = Source::Loaded { load: site, first };
            let used = self.used_kind(&src);
            let ui: &Inst = &self.x.p.blocks[at.0].insts[at.1];
            let un = ui.name();
            let uav = un.contains("store") || un.contains("atomic") || self.is_uav(&src);
            let u = uav_bits(uav, false);
            let (props, image) = match &ui.fields {
                Fields::Image { dim, .. } => {
                    let kind = match dim {
                        0 => 1,
                        1 => 2,
                        2 => 4,
                        4 => 6,
                        5 => 7,
                        _ => return Err("an image dimension (cube or MSAA)"),
                    };
                    ((kind | u, 0x409), Some(kind))
                }
                Fields::Buffer { idxen, format, .. } => {
                    let stride = used.map_or(0, |u| u.stride);
                    let props = if un.starts_with("buffer_load_format")
                        || un.starts_with("buffer_store_format")
                    {
                        (10 | u, 0x409)
                    } else if *idxen && format.is_none() && stride >= 4 && stride.is_multiple_of(4)
                    {
                        (12 | u, stride)
                    } else {
                        (11 | u, 0)
                    };
                    if used.is_some_and(|u| u.kind == ResourceType::Cbv) {
                        return Err("a constant buffer's descriptor read as data");
                    }
                    (props, None)
                }
                Fields::Smem { .. } if un.starts_with("s_buffer_load_dword") => {
                    if !uav
                        && !matches!(
                            used.map(|u| u.kind),
                            Some(ResourceType::Srv | ResourceType::Uav)
                        )
                    {
                        return Err("a constant buffer's descriptor read as data");
                    }
                    ((11 | u, 0), None)
                }
                _ => return Err("a scalar memory read as data"),
            };
            (src, uav, props, image)
        } else {
            // Read as data and never used directly or via another load of the
            // same entry: a table range names its kind, and an 8-dword load in
            // a Table/ResourceHeap with no range naming it (runtime index) is
            // taken as a 2D image (`#1067`).
            let src = Source::Loaded {
                load: site,
                first: 0,
            };
            let used = self.used_kind(&src);
            if matches!(used.map(|u| u.kind), Some(ResourceType::Cbv)) {
                return Err("a constant buffer's descriptor read as data");
            }
            let in_views = self
                .scalar
                .get(&site)
                .and_then(|a| self.x.native.layout.get(a.param))
                .is_some_and(|rp| matches!(rp.root, RootType::Table | RootType::ResourceHeap));
            let (uav, dwords, stride) = match used {
                Some(u) if matches!(u.kind, ResourceType::Srv | ResourceType::Uav) => {
                    (u.kind == ResourceType::Uav, u.dwords, u.stride)
                }
                None if in_views && count == 8 => (false, 8, 0),
                _ => return Err("a descriptor read only as data"),
            };
            let u = if uav { 0x1000 } else { 0 };
            let (props, image) = if dwords == 4 {
                let props = if stride >= 4 && stride.is_multiple_of(4) {
                    (12 | u, stride)
                } else {
                    (11 | u, 0)
                };
                (props, None)
            } else {
                ((2 | u, 0x409), Some(2))
            };
            (src, uav, props, image)
        };
        let Source::Loaded { first, .. } = src else {
            unreachable!()
        };
        let h = self.handle(site, &src, false, props)?;
        let c = self.cur;
        let i32t = self.s.t.i32;
        let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
        let mip = if image.is_some() && !uav {
            self.s.m.i32(0)
        } else {
            self.s.m.undef(i32t)
        };
        let d = self
            .s
            .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
        let field = |l: &mut Self, k: u32| l.s.m.extract(c, d, k);
        let words: Vec<(u32, V)> = match image {
            Some(kind) => {
                let one = self.s.m.i32(1);
                let w = field(self, 0);
                let (height, depth) = match kind {
                    1 => (one, one),
                    2 => (field(self, 1), one),
                    6 => (one, field(self, 1)),
                    _ => (field(self, 1), field(self, 2)),
                };
                let levels = if uav { one } else { field(self, 3) };
                let less = |l: &mut Self, x: V| l.s.m.bin(c, Bin::Sub, x, one);
                let (w, height, depth, levels) = (
                    less(self, w),
                    less(self, height),
                    less(self, depth),
                    less(self, levels),
                );
                let k = |l: &mut Self, v: u32| l.s.m.i32(v);
                let (k30, k2, k14, k16, mask) = (
                    k(self, 30),
                    k(self, 2),
                    k(self, 14),
                    k(self, 16),
                    k(self, 0x3FFF),
                );
                let d1 = self.s.m.bin(c, Bin::Shl, w, k30);
                let hi = self.s.m.bin(c, Bin::LShr, w, k2);
                let hi = self.s.m.bin(c, Bin::And, hi, mask);
                let ht = self.s.m.bin(c, Bin::Shl, height, k14);
                let d2 = self.s.m.bin(c, Bin::Or, hi, ht);
                let ty = [0, 8, 9, 0, 10, 0, 12, 13][kind as usize];
                let base = k(self, 0xFAC | ty << 28);
                let last = self.s.m.bin(c, Bin::Shl, levels, k16);
                let d3 = self.s.m.bin(c, Bin::Or, base, last);
                vec![(1, d1), (2, d2), (3, d3), (4, depth)]
            }
            None => vec![(2, field(self, 0))],
        };
        for k in 0..count {
            let rel = k.wrapping_sub(u32::from(first));
            match words.iter().find(|w| w.0 == rel) {
                Some(&(_, x)) => self.write(inst, Reg::S(code + k as u8), x, None),
                None => {
                    if let Some(v) = inst.write(Reg::S(code + k as u8)) {
                        self.vals.insert(v, Val::Missing(OTHER));
                    }
                }
            }
        }
        Ok(())
    }

    /// The use site and first dword of a resource descriptor loaded from
    /// the same heap entry as the scalar load at `site`: the same parameter
    /// and offset, and the same computed term (or the first lane's value of
    /// the same VGPR, in another pass of a waterfall loop).
    fn same_entry_use(&self, site: Site) -> Option<(Site, u8)> {
        let a = self.scalar.get(&site)?;
        let lane = |v: Value| -> Value {
            self.x
                .p
                .blocks
                .iter()
                .flat_map(|b| &b.insts)
                .find(|i| i.writes.iter().any(|w| w.1 == v))
                .filter(|i| i.name() == "v_readfirstlane_b32")
                .and_then(|i| i.reads.iter().find(|r| matches!(r.0, Reg::V(_))))
                .map_or(v, |r| r.1)
        };
        let same = |b: &Address| {
            b.param == a.param
                && b.offset == a.offset
                && match (a.dynamic, b.dynamic) {
                    (None, None) => true,
                    (Some(x), Some(y)) => x == y || lane(x) == lane(y),
                    _ => false,
                }
        };
        self.x
            .res
            .descriptors
            .iter()
            .filter(|d| d.role == Role::Resource)
            .find_map(|d| {
                d.source.leaves().into_iter().find_map(|l| match *l {
                    Source::Loaded { load, first }
                        if load != site && self.scalar.get(&load).is_some_and(same) =>
                    {
                        Some((d.site, first))
                    }
                    _ => None,
                })
            })
    }

    /// Whether a descriptor the program uses comes from the scalar load at
    /// `site` (a descriptor in memory, not data).
    fn loads_descriptor(&self, site: Site) -> bool {
        self.x.res.descriptors.iter().any(|d| {
            d.source.leaves().into_iter().any(|l| {
                matches!(*l, Source::Loaded { load, .. } | Source::Fetched { load, .. }
                    if load == site)
            })
        })
    }

    /// `s_load_dword*` from a root view's address: that buffer's memory, as
    /// global loads from it read it (a root SRV raw, a root CBV as a
    /// constant buffer).
    fn root_view_smem(&mut self, inst: &Inst, a: Address, site: Site) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar memory read as data");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar memory read as data"),
        };
        let c = self.cur;
        let mut off = self.s.m.i32(a.offset);
        if let Some(d) = a.dynamic {
            let d = self.i32_of(d)?;
            off = self.s.m.bin(c, Bin::Add, off, d);
        }
        let src = Source::Built {
            address: a,
            bits: 0,
            words: [0, 0],
        };
        let raw = match self.used_kind(&src).map(|u| u.kind) {
            Some(ResourceType::Srv) => true,
            Some(ResourceType::Cbv) | None => false,
            _ => return Err("a root UAV read with scalar loads"),
        };
        let h = if raw {
            self.handle(site, &src, false, (11, 0))?
        } else {
            self.handle(site, &src, false, (13, 0x1_0000))?
        };
        for k in 0..count {
            let four = self.s.m.i32(4 * k);
            let at = self.s.m.bin(c, Bin::Add, off, four);
            let x = if raw {
                let (u, mask, al) = (
                    self.s.m.undef(self.s.t.i32),
                    self.s.m.i8(1),
                    self.s.m.i32(4),
                );
                let r = self.s.op(
                    c,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    139,
                    &[h, at, u, mask, al],
                );
                let x = self.s.m.extract(c, r, 0);
                self.as_int(x)
            } else {
                self.cbuffer_dword(h, at)
            };
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// `s_load_dword*` from the program's constant table.
    fn code_smem(&mut self, inst: &Inst, a: Address) -> Result<(), Why> {
        let Fields::Smem {
            data: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err("a scalar load");
        };
        let count = match inst.name().strip_prefix("s_load_dword") {
            Some("") => 1u32,
            Some("x2") => 2,
            Some("x4") => 4,
            Some("x8") => 8,
            Some("x16") => 16,
            _ => return Err("a scalar load from the code"),
        };
        // (First: a first-lane term may end the block.)
        let addr = self.address_offset(a)?;
        let c = self.cur;
        for k in 0..count {
            let off = self.s.m.i32(4 * k);
            let at = self.s.m.bin(c, Bin::Add, addr, off);
            let x = self.code_dword(at)?;
            self.write(inst, Reg::S(code + k as u8), x, None);
        }
        Ok(())
    }

    /// The dword at a computed byte address of a constant buffer.
    fn cbuffer_dword(&mut self, h: V, a: V) -> V {
        let c = self.cur;
        let four = self.s.m.i32(4);
        let ri = self.s.m.bin(c, Bin::LShr, a, four);
        let r = self.s.op(
            c,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            self.s.t.cbuf_ret,
            59,
            &[h, ri],
        );
        let two = self.s.m.i32(2);
        let three = self.s.m.i32(3);
        let comp = self.s.m.bin(c, Bin::LShr, a, two);
        let comp = self.s.m.bin(c, Bin::And, comp, three);
        let mut x = self.s.m.extract(c, r, 0);
        for j in 1..4 {
            let e = self.s.m.extract(c, r, j);
            let jc = self.s.m.i32(j);
            let is = self.s.m.cmp(c, pred::EQ, comp, jc);
            x = self.s.m.select(c, is, e, x);
        }
        x
    }

    /// `channels` channels of the element at byte `off` of raw buffer `h`,
    /// converted from the GFX10 buffer format `format` as the hardware's
    /// typed loads do (channels past the format's read 0, 0, 0, 1).
    // shortcut: the descriptor's channel swizzle is taken as the identity
    // (it is for raw views and the descriptors code builds), elements of 4
    // bytes or more as dword-aligned. Norm formats divide, which the AMD
    // driver does 1 ULP off the hardware's correctly rounded conversion for
    // some values; correct it with an integer check if a title compares
    // converted values exactly.
    fn format_load(&mut self, h: V, off: V, format: u8, channels: u32) -> Result<Vec<V>, Why> {
        let (bits, numeric) = buffer_format(format).ok_or("a buffer format")?;
        let c = self.cur;
        let total: u32 = bits.iter().sum();
        let (u, align) = (self.s.m.undef(self.s.t.i32), self.s.m.i32(4));
        let load = |l: &mut Self, at: V, n: u32| -> Vec<V> {
            let mask = l.s.m.i8((1u8 << n) - 1);
            let r = l.s.op(
                c,
                "rawBufferLoad.f32",
                Attr::ReadOnly,
                l.s.t.res_ret,
                139,
                &[h, at, u, mask, align],
            );
            (0..n)
                .map(|k| {
                    let x = l.s.m.extract(c, r, k);
                    l.as_int(x)
                })
                .collect()
        };
        let words = if total < 32 {
            // A byte or half inside the dword that holds it.
            let (not3, three) = (self.s.m.i32(!3), self.s.m.i32(3));
            let at = self.s.m.bin(c, Bin::And, off, not3);
            let w = load(self, at, 1)[0];
            let byte = self.s.m.bin(c, Bin::And, off, three);
            let sh = self.s.m.bin(c, Bin::Shl, byte, three);
            vec![self.s.m.bin(c, Bin::LShr, w, sh)]
        } else if let Some(at) = self.s.m.int_value(off) {
            if at % 4 == 0 {
                load(self, off, total / 32)
            } else {
                // A value of several dwords at a byte offset inside one: the
                // console reads the bytes the address names, so each word is
                // taken from the two dword-aligned words around it. Reading
                // at the byte address itself would promise the host an
                // alignment the console does not have, and a host that
                // believes it can hand back zeros (`notes/vtx.md`: a
                // position fetched at an odd byte offset collapsing every
                // vertex to the origin).
                #[cfg(test)]
                {
                    self.s.unaligned_typed_loads += 1;
                }
                let n = (total / 32) as usize;
                let shift = (at % 4) as u32 * 8;
                let aligned = self.s.m.i32((at & !3) as u32);
                let sh = self.s.m.i32(shift);
                let up = self.s.m.i32(32 - shift);
                let raw = load(self, aligned, n as u32 + 1);
                (0..n)
                    .map(|k| {
                        let lo = self.s.m.bin(c, Bin::LShr, raw[k], sh);
                        let hi = self.s.m.bin(c, Bin::Shl, raw[k + 1], up);
                        self.s.m.bin(c, Bin::Or, lo, hi)
                    })
                    .collect()
            }
        } else {
            load(self, off, total / 32)
        };
        let float = |l: &mut Self, f: V| l.as_int(f);
        let mut out = Vec::new();
        let mut at = 0;
        for &w in bits {
            let word = words[(at / 32) as usize];
            let field = if w == 32 {
                word
            } else {
                let sh = self.s.m.i32(at % 32);
                let t = self.s.m.bin(c, Bin::LShr, word, sh);
                let m = self.s.m.i32((1 << w) - 1);
                self.s.m.bin(c, Bin::And, t, m)
            };
            at += w;
            // Sign-extended from the field's top bit.
            let signed = |l: &mut Self| {
                if w == 32 {
                    return field;
                }
                let k = l.s.m.i32(32 - w);
                let t = l.s.m.bin(c, Bin::Shl, field, k);
                l.s.m.bin(c, Bin::AShr, t, k)
            };
            let f32t = self.s.t.f32;
            let x = match numeric {
                Numeric::Unorm => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    let max = self.s.m.f32((((1u64 << w) - 1) as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    float(self, q)
                }
                Numeric::Snorm => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    let max = self.s.m.f32((((1u64 << (w - 1)) - 1) as f32).to_bits());
                    let q = self.s.m.bin(c, Bin::SDiv, f, max);
                    let lo = self.s.m.f32((-1.0f32).to_bits());
                    let q = self.binary(35, q, lo);
                    float(self, q)
                }
                Numeric::Uscaled => {
                    let f = self.s.m.cast(c, Cast::UiToFp, field, f32t);
                    float(self, f)
                }
                Numeric::Sscaled => {
                    let s = signed(self);
                    let f = self.s.m.cast(c, Cast::SiToFp, s, f32t);
                    float(self, f)
                }
                Numeric::Uint => field,
                Numeric::Sint => signed(self),
                Numeric::Float if w == 32 => field,
                Numeric::Float => {
                    let h = match half_shift(w) {
                        0 => field,
                        k => {
                            let sh = self.s.m.i32(k);
                            self.s.m.bin(c, Bin::Shl, field, sh)
                        }
                    };
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    float(self, f)
                }
            };
            out.push(x);
        }
        let one = match numeric {
            Numeric::Uint | Numeric::Sint => 1,
            _ => 1.0f32.to_bits(),
        };
        for k in out.len() as u32..channels {
            out.push(self.s.m.i32(if k == 3 { one } else { 0 }));
        }
        out.truncate(channels as usize);
        Ok(out)
    }

    /// For a first-lane address ([`Address::first_lane`]), the first active
    /// lane's value of `d` at its `v_readfirstlane_b32`, in every lane.
    fn first_lane_value(&mut self, read: Value, d: V) -> Result<V, Why> {
        let (b, i) = self
            .x
            .p
            .blocks
            .iter()
            .enumerate()
            .find_map(|(b, block)| {
                let i = block
                    .insts
                    .iter()
                    .position(|i| i.writes.iter().any(|w| w.1 == read))?;
                Some((b, i))
            })
            .ok_or("a first-lane address")?;
        let guard = self.guard(b, i)?;
        self.s.flags |= flags::WAVE_OPS;
        // Guarded code runs in every lane; the first lane the guard holds in
        // reads the term, and the others take it from there (the largest of
        // that one and zeros).
        let at = self.begin_guarded(guard);
        let i32t = self.s.t.i32;
        let first = self.s.op(
            self.cur,
            "waveReadLaneFirst.i32",
            Attr::NoUnwind,
            i32t,
            118,
            &[d],
        );
        if at.is_none() {
            return Ok(first);
        }
        let zero = self.s.m.i32(0);
        let r = self.end_guarded(at, &[(first, zero)])[0];
        let (k, u) = (self.s.m.i8(3), self.s.m.i8(1));
        Ok(self.s.op(
            self.cur,
            "waveActiveOp.i32",
            Attr::NoUnwind,
            i32t,
            119,
            &[r, k, u],
        ))
    }

    /// The byte offset a buffer descriptor built around a root descriptor adds
    /// (constant and computed parts); none for other descriptors.
    fn built_offset(&mut self, src: &Source) -> Result<(u32, Option<V>), Why> {
        match src {
            Source::Built { address, .. } => {
                if address.first_lane.is_some() {
                    return Err("a descriptor built around a first-lane address");
                }
                if address.rel.is_some() {
                    Ok((0, Some(self.address_offset(*address)?)))
                } else {
                    let x = match address.dynamic {
                        Some(v) => Some(self.i32_of(v)?),
                        None => None,
                    };
                    Ok((address.offset, x))
                }
            }
            _ => Ok((0, None)),
        }
    }

    /// Computes the byte offset `a` adds to its root parameter (or code base).
    fn address_offset(&mut self, a: Address) -> Result<V, Why> {
        if let Some(RelOffset {
            value,
            base,
            extra_dynamic,
        }) = a.rel
        {
            let mut local = HashMap::new();
            let mut off = self.rel_offset(value, base, &mut local)?;
            if let Some(read) = a.first_lane {
                off = self.first_lane_value(read, off)?;
            }
            let c = self.cur;
            if a.offset != 0 {
                let k = self.s.m.i32(a.offset);
                off = self.s.m.bin(c, Bin::Add, off, k);
            }
            if let Some(d) = extra_dynamic {
                let d = self.i32_of(d)?;
                off = self.s.m.bin(c, Bin::Add, off, d);
            }
            return Ok(off);
        }
        let dyn_v = match a.dynamic {
            Some(d) => {
                let d = self.i32_of(d)?;
                Some(match a.first_lane {
                    Some(read) => self.first_lane_value(read, d)?,
                    None => d,
                })
            }
            None => None,
        };
        let c = self.cur;
        let mut off = self.s.m.i32(a.offset);
        if let Some(d) = dyn_v {
            off = self.s.m.bin(c, Bin::Add, off, d);
        }
        Ok(off)
    }

    fn op_value(inst: &Inst, op: &Operand) -> Option<Value> {
        match *op {
            Operand::Scalar { code, .. } if code != NULL => inst.read(Reg::S(code)),
            Operand::Vector { reg, .. } => inst.read(Reg::V(reg)),
            _ => None,
        }
    }

    /// Follows unmodified, unguarded `v_mov_b32` chains back to an SGPR value.
    fn uniform_vgpr_src(&self, mut inst: &'a Inst, mut op: &'a Operand) -> Option<Value> {
        loop {
            match *op {
                Operand::Scalar { code, .. } if code != NULL => return inst.read(Reg::S(code)),
                Operand::Vector { reg, .. } => {
                    let v = inst.read(Reg::V(reg))?;
                    let &(b, i) = self.def_site.get(&v)?;
                    if self.x.st.guards[b][i].is_some() {
                        return None;
                    }
                    let next = &self.x.p.blocks[b].insts[i];
                    if next.name() != "v_mov_b32" {
                        return None;
                    }
                    let Fields::Valu(f) = &next.fields else {
                        return None;
                    };
                    if f.abs != 0
                        || f.neg != 0
                        || f.sdwa.is_some()
                        || f.dpp.is_some()
                        || f.clamp
                        || f.omod != 0
                    {
                        return None;
                    }
                    inst = next;
                    op = f.src.first()?;
                }
                _ => return None,
            }
        }
    }

    /// Resolves a constant-lane `v_readlane_b32` reading an SGPR spilled via `v_writelane_b32`.
    fn spilled_value(&self, inst: &Inst, src: &Operand, lane: &Operand) -> Option<Value> {
        let lane_idx = match *lane {
            Operand::Const(c) => crate::resource::inline(c)?,
            Operand::Literal(x) => x,
            _ => return None,
        };
        let Operand::Vector { reg, .. } = *src else {
            return None;
        };
        let v = inst.read(Reg::V(reg))?;
        self.spilled_in_vgpr(v, lane_idx, &mut Vec::new())
            .ok()
            .flatten()
    }

    fn spilled_in_vgpr(
        &self,
        v: Value,
        lane: u32,
        seen: &mut Vec<Value>,
    ) -> Result<Option<Value>, ()> {
        if let Some(&(b, i)) = self.phi_at.get(&v) {
            if seen.contains(&v) {
                return Ok(None);
            }
            seen.push(v);
            let mut found = None;
            for &a in &self.x.p.blocks[b].phis[i].args {
                if let Some(m) = self.spilled_in_vgpr(a, lane, seen)? {
                    if found.is_some_and(|f| f != m) {
                        return Err(());
                    }
                    found = Some(m);
                }
            }
            seen.pop();
            return Ok(found);
        }
        let &(b, i) = self.def_site.get(&v).ok_or(())?;
        let inst = &self.x.p.blocks[b].insts[i];
        if inst.name() != "v_writelane_b32" {
            return Err(());
        }
        let Fields::Valu(w) = &inst.fields else {
            return Err(());
        };
        let (Some(Operand::Vector { reg, .. }), [value, at]) = (w.dst, &w.src[..]) else {
            return Err(());
        };
        let at_lane = match *at {
            Operand::Const(c) => crate::resource::inline(c).ok_or(())?,
            Operand::Literal(x) => x,
            _ => return Err(()),
        };
        if at_lane != lane {
            return self.spilled_in_vgpr(inst.read(Reg::V(reg)).ok_or(())?, lane, seen);
        }
        match *value {
            Operand::Scalar { code, .. } if code != NULL => Ok(inst.read(Reg::S(code))),
            _ => Err(()),
        }
    }

    fn op_reaches_base(&self, inst: &Inst, op: &Operand, base: Value) -> bool {
        Self::op_value(inst, op).is_some_and(|v| self.reaches_base(v, base))
    }

    fn reaches_base(&self, v: Value, base: Value) -> bool {
        fn rec(
            l: &Lower<'_>,
            v: Value,
            base: Value,
            visiting: &mut Vec<Value>,
            memo: &mut HashMap<Value, bool>,
        ) -> bool {
            if v == base {
                return true;
            }
            if let Some(&hit) = memo.get(&v) {
                return hit;
            }
            if visiting.contains(&v) {
                return false;
            }
            let r = if let Some(&(b, i)) = l.phi_at.get(&v) {
                visiting.push(v);
                let hit = l.x.p.blocks[b].phis[i]
                    .args
                    .iter()
                    .any(|&a| rec(l, a, base, visiting, memo));
                visiting.pop();
                hit
            } else if let Some(&(b, i)) = l.def_site.get(&v) {
                let inst = &l.x.p.blocks[b].insts[i];
                let name = inst.name();
                let mut op_reaches = |op: &Operand| {
                    Lower::op_value(inst, op).is_some_and(|x| rec(l, x, base, visiting, memo))
                };
                match &inst.fields {
                    Fields::Sop { dst, src, .. } => match name {
                        "s_mov_b32" => src.first().is_some_and(&mut op_reaches),
                        "s_mov_b64" => {
                            let k = inst.writes.iter().position(|w| w.1 == v);
                            match (k, src.first()) {
                                (Some(k), Some(&Operand::Scalar { code, .. })) if code != NULL => {
                                    inst.read(Reg::S(code + k as u8))
                                        .is_some_and(|x| rec(l, x, base, visiting, memo))
                                }
                                _ => false,
                            }
                        }
                        "s_addk_i32" => match dst {
                            Some(Operand::Scalar { code, .. }) if *code != NULL => inst
                                .read(Reg::S(*code))
                                .is_some_and(|x| rec(l, x, base, visiting, memo)),
                            _ => false,
                        },
                        "s_add_u32" | "s_add_i32" | "s_or_b32" => {
                            src.first().is_some_and(&mut op_reaches)
                                || src.get(1).is_some_and(&mut op_reaches)
                        }
                        "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32"
                        | "s_lshl4_add_u32" => src.get(1).is_some_and(&mut op_reaches),
                        "s_sub_u32" | "s_sub_i32" => src.first().is_some_and(&mut op_reaches),
                        _ => false,
                    },
                    Fields::Valu(f)
                        if f.abs == 0
                            && f.neg == 0
                            && f.sdwa.is_none()
                            && f.dpp.is_none()
                            && !f.clamp
                            && f.omod == 0 =>
                    {
                        match name {
                            "v_mov_b32" | "v_readfirstlane_b32" => {
                                f.src.first().is_some_and(&mut op_reaches)
                            }
                            "v_readlane_b32" => match (f.src.first(), f.src.get(1)) {
                                (Some(op0), Some(op1)) => {
                                    if let Some(sv) = l.spilled_value(inst, op0, op1) {
                                        rec(l, sv, base, visiting, memo)
                                    } else {
                                        op_reaches(op0)
                                    }
                                }
                                _ => false,
                            },
                            "v_add_co_u32" | "v_add_nc_u32" | "v_add_nc_i32" | "v_or_b32" => {
                                f.src.first().is_some_and(&mut op_reaches)
                                    || f.src.get(1).is_some_and(&mut op_reaches)
                            }
                            "v_lshl_add_u32" => f.src.get(2).is_some_and(&mut op_reaches),
                            "v_add3_u32" => f.src.iter().take(3).any(&mut op_reaches),
                            "v_sub_co_u32" | "v_sub_nc_u32" => {
                                f.src.first().is_some_and(&mut op_reaches)
                            }
                            "v_subrev_co_u32" | "v_subrev_nc_u32" => {
                                f.src.get(1).is_some_and(&mut op_reaches)
                            }
                            _ => false,
                        }
                    }
                    _ => false,
                }
            } else {
                false
            };
            if visiting.is_empty() {
                memo.insert(v, r);
            }
            r
        }
        rec(self, v, base, &mut Vec::new(), &mut HashMap::new())
    }

    fn rel_offset_op(
        &mut self,
        inst: &Inst,
        op: &Operand,
        base: Value,
        local: &mut HashMap<Value, V>,
    ) -> Result<V, Why> {
        let v = Self::op_value(inst, op).ok_or("an untracked relative address")?;
        self.rel_offset(v, base, local)
    }

    /// Emits `v - base` in `self.cur`, caching non-phi nodes only in `local`
    /// (per block / edge) and phi nodes globally in `self.addr_phis`.
    fn rel_offset(
        &mut self,
        v: Value,
        base: Value,
        local: &mut HashMap<Value, V>,
    ) -> Result<V, Why> {
        if v == base {
            return Ok(self.s.m.i32(0));
        }
        if let Some(&x) = local.get(&v) {
            return Ok(x);
        }
        if let Some(&(phi_b, phi_idx)) = self.phi_at.get(&v) {
            if let Some(&phi) = self.addr_phis.get(&(v, base)) {
                return Ok(phi);
            }
            let blk = self.block[phi_b].ok_or("an unreachable address phi")?;
            let phi = self.s.m.phi(blk, self.s.t.i32);
            self.addr_phis.insert((v, base), phi);
            self.pending_addr_phis.push((phi, phi_b, phi_idx, base));
            return Ok(phi);
        }
        let &(ib, ii) = self
            .def_site
            .get(&v)
            .ok_or("an untracked relative address")?;
        let inst: &'a Inst = &self.x.p.blocks[ib].insts[ii];
        let name = inst.name();
        let c = self.cur;
        let x = match &inst.fields {
            Fields::Sop { dst, src, imm } => match name {
                "s_mov_b32" => {
                    let op0 = src.first().ok_or("a missing operand")?;
                    self.rel_offset_op(inst, op0, base, local)?
                }
                "s_mov_b64" => {
                    let k = inst
                        .writes
                        .iter()
                        .position(|w| w.1 == v)
                        .ok_or("an untracked relative address")?;
                    match src.first() {
                        Some(&Operand::Scalar { code, .. }) if code != NULL => {
                            let prev = inst
                                .read(Reg::S(code + k as u8))
                                .ok_or("an unread operand")?;
                            self.rel_offset(prev, base, local)?
                        }
                        _ => return Err("an untracked relative address"),
                    }
                }
                "s_addk_i32" => {
                    let Some(Operand::Scalar { code, .. }) = dst else {
                        return Err("a missing operand");
                    };
                    let prev = inst.read(Reg::S(*code)).ok_or("an unread operand")?;
                    let a = self.rel_offset(prev, base, local)?;
                    let b = self.s.m.i32(i32::from(imm.unwrap_or(0) as i16) as u32);
                    self.s.m.bin(c, Bin::Add, a, b)
                }
                "s_add_u32" | "s_add_i32" | "s_or_b32" => {
                    let (op0, op1) = (
                        src.first().ok_or("a missing operand")?,
                        src.get(1).ok_or("a missing operand")?,
                    );
                    match (
                        self.op_reaches_base(inst, op0, base),
                        self.op_reaches_base(inst, op1, base),
                    ) {
                        (true, false) => {
                            let a = self.rel_offset_op(inst, op0, base, local)?;
                            let b = self.operand(inst, op1)?;
                            self.s.m.bin(c, Bin::Add, a, b)
                        }
                        (false, true) => {
                            let a = self.rel_offset_op(inst, op1, base, local)?;
                            let b = self.operand(inst, op0)?;
                            self.s.m.bin(c, Bin::Add, a, b)
                        }
                        _ => return Err("an untracked relative address"),
                    }
                }
                "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
                    let (op0, op1) = (
                        src.first().ok_or("a missing operand")?,
                        src.get(1).ok_or("a missing operand")?,
                    );
                    if self.op_reaches_base(inst, op0, base)
                        || !self.op_reaches_base(inst, op1, base)
                    {
                        return Err("an untracked relative address");
                    }
                    let a = self.operand(inst, op0)?;
                    let k = self.s.m.i32(u32::from(name.as_bytes()[6] - b'0'));
                    let sh = self.s.m.bin(c, Bin::Shl, a, k);
                    let b = self.rel_offset_op(inst, op1, base, local)?;
                    self.s.m.bin(c, Bin::Add, sh, b)
                }
                "s_sub_u32" | "s_sub_i32" => {
                    let (op0, op1) = (
                        src.first().ok_or("a missing operand")?,
                        src.get(1).ok_or("a missing operand")?,
                    );
                    if !self.op_reaches_base(inst, op0, base)
                        || self.op_reaches_base(inst, op1, base)
                    {
                        return Err("an untracked relative address");
                    }
                    let a = self.rel_offset_op(inst, op0, base, local)?;
                    let b = self.operand(inst, op1)?;
                    self.s.m.bin(c, Bin::Sub, a, b)
                }
                _ => return Err("an untracked relative address"),
            },
            Fields::Valu(f)
                if f.abs == 0
                    && f.neg == 0
                    && f.sdwa.is_none()
                    && f.dpp.is_none()
                    && !f.clamp
                    && f.omod == 0 =>
            {
                let mut raw = match name {
                    "v_mov_b32" => {
                        let op0 = f.src.first().ok_or("a missing operand")?;
                        self.rel_offset_op(inst, op0, base, local)?
                    }
                    "v_readfirstlane_b32" => {
                        let op0 = f.src.first().ok_or("a missing operand")?;
                        if let Some(sv) = self.uniform_vgpr_src(inst, op0) {
                            self.rel_offset(sv, base, local)?
                        } else {
                            let a = self.rel_offset_op(inst, op0, base, local)?;
                            self.s.flags |= flags::WAVE_OPS;
                            self.s.op(
                                c,
                                "waveReadLaneFirst.i32",
                                Attr::NoUnwind,
                                self.s.t.i32,
                                118,
                                &[a],
                            )
                        }
                    }
                    "v_readlane_b32" => {
                        let (op0, op1) = (
                            f.src.first().ok_or("a missing operand")?,
                            f.src.get(1).ok_or("a missing operand")?,
                        );
                        if let Some(sv) = self
                            .spilled_value(inst, op0, op1)
                            .or_else(|| self.uniform_vgpr_src(inst, op0))
                        {
                            self.rel_offset(sv, base, local)?
                        } else {
                            let a = self.rel_offset_op(inst, op0, base, local)?;
                            let lane = self.operand(inst, op1)?;
                            if self.mask_data == MaskData::Host && !matches!(lane, V::Const(_)) {
                                self.s.flags |= flags::WAVE_OPS;
                            } else {
                                self.cross_lane(false)?;
                            }
                            self.read_lane(a, lane)
                        }
                    }
                    "v_add_co_u32" | "v_add_nc_u32" | "v_add_nc_i32" | "v_or_b32" => {
                        let (op0, op1) = (
                            f.src.first().ok_or("a missing operand")?,
                            f.src.get(1).ok_or("a missing operand")?,
                        );
                        match (
                            self.op_reaches_base(inst, op0, base),
                            self.op_reaches_base(inst, op1, base),
                        ) {
                            (true, false) => {
                                let a = self.rel_offset_op(inst, op0, base, local)?;
                                let b = self.operand(inst, op1)?;
                                self.s.m.bin(c, Bin::Add, a, b)
                            }
                            (false, true) => {
                                let a = self.rel_offset_op(inst, op1, base, local)?;
                                let b = self.operand(inst, op0)?;
                                self.s.m.bin(c, Bin::Add, a, b)
                            }
                            _ => return Err("an untracked relative address"),
                        }
                    }
                    "v_lshl_add_u32" => {
                        let (op0, op1, op2) = (
                            f.src.first().ok_or("a missing operand")?,
                            f.src.get(1).ok_or("a missing operand")?,
                            f.src.get(2).ok_or("a missing operand")?,
                        );
                        if self.op_reaches_base(inst, op0, base)
                            || self.op_reaches_base(inst, op1, base)
                            || !self.op_reaches_base(inst, op2, base)
                        {
                            return Err("an untracked relative address");
                        }
                        let a = self.operand(inst, op0)?;
                        let sh = self.operand(inst, op1)?;
                        let k31 = self.s.m.i32(31);
                        let sh = self.s.m.bin(c, Bin::And, sh, k31);
                        let t = self.s.m.bin(c, Bin::Shl, a, sh);
                        let b = self.rel_offset_op(inst, op2, base, local)?;
                        self.s.m.bin(c, Bin::Add, t, b)
                    }
                    "v_add3_u32" => {
                        let (op0, op1, op2) = (
                            f.src.first().ok_or("a missing operand")?,
                            f.src.get(1).ok_or("a missing operand")?,
                            f.src.get(2).ok_or("a missing operand")?,
                        );
                        let ops = [op0, op1, op2];
                        let reaches: Vec<usize> = (0..3)
                            .filter(|&k| self.op_reaches_base(inst, ops[k], base))
                            .collect();
                        let [base_idx] = reaches[..] else {
                            return Err("an untracked relative address");
                        };
                        let mut acc = self.rel_offset_op(inst, ops[base_idx], base, local)?;
                        for (k, op) in ops.into_iter().enumerate() {
                            if k != base_idx {
                                let term = self.operand(inst, op)?;
                                acc = self.s.m.bin(c, Bin::Add, acc, term);
                            }
                        }
                        acc
                    }
                    "v_sub_co_u32" | "v_sub_nc_u32" => {
                        let (op0, op1) = (
                            f.src.first().ok_or("a missing operand")?,
                            f.src.get(1).ok_or("a missing operand")?,
                        );
                        if !self.op_reaches_base(inst, op0, base)
                            || self.op_reaches_base(inst, op1, base)
                        {
                            return Err("an untracked relative address");
                        }
                        let a = self.rel_offset_op(inst, op0, base, local)?;
                        let b = self.operand(inst, op1)?;
                        self.s.m.bin(c, Bin::Sub, a, b)
                    }
                    "v_subrev_co_u32" | "v_subrev_nc_u32" => {
                        let (op0, op1) = (
                            f.src.first().ok_or("a missing operand")?,
                            f.src.get(1).ok_or("a missing operand")?,
                        );
                        if self.op_reaches_base(inst, op0, base)
                            || !self.op_reaches_base(inst, op1, base)
                        {
                            return Err("an untracked relative address");
                        }
                        let a = self.operand(inst, op0)?;
                        let b = self.rel_offset_op(inst, op1, base, local)?;
                        self.s.m.bin(c, Bin::Sub, b, a)
                    }
                    _ => return Err("an untracked relative address"),
                };
                if let Some(Operand::Vector { reg, .. }) = f.dst
                    && let Some(g) = self.x.st.guards[ib][ii]
                    && let Some(old) = inst.read(Reg::V(reg))
                    && self.reaches_base(old, base)
                {
                    let cond = self.boolean(g, c)?;
                    let fallback = self.rel_offset(old, base, local)?;
                    raw = self.s.m.select(c, cond, raw, fallback);
                }
                raw
            }
            _ => return Err("an untracked relative address"),
        };
        local.insert(v, x);
        Ok(x)
    }

    fn desc_addr(&self, load: Site, first: u8) -> Option<(usize, u32, Option<Value>)> {
        let a = self.scalar.get(&load)?;
        Some((
            a.param,
            a.offset.wrapping_add(4 * u32::from(first)),
            a.dynamic,
        ))
    }

    fn is_uav(&self, src: &Source) -> bool {
        let leaves = src.leaves();
        let in_set = |set: &HashSet<(usize, u32, Option<Value>)>| {
            leaves.iter().any(|l| match **l {
                Source::Loaded { load, first } | Source::Fetched { load, first } => self
                    .desc_addr(load, first)
                    .is_some_and(|a| set.contains(&a)),
                _ => false,
            })
        };
        if in_set(&self.written_uavs) {
            return true;
        }
        match self.used_kind(src) {
            Some(u) => u.kind == ResourceType::Uav,
            None => in_set(&self.glc_uavs),
        }
    }

    /// The used-resource entry a descriptor loaded from a table falls in.
    fn used_kind(&self, src: &Source) -> Option<&'a UsedResource> {
        if let Source::Phi { args, .. } | Source::Select { args, .. } | Source::Lanes { args, .. } =
            src
        {
            return self.used_kind(args.first()?);
        }
        if let Source::LanePhi { args, .. } = src {
            return args.iter().flatten().find_map(|a| self.used_kind(a));
        }
        if let Source::Built { address, .. } = src {
            let rp = self.x.native.layout.get(address.param)?;
            return self
                .x
                .native
                .used
                .iter()
                .find(|u| u.root == RootType::Descriptor && u.user_data == rp.user_data);
        }
        let (Source::Loaded { load, first } | Source::Fetched { load, first }) = src else {
            return None;
        };
        let a = self.scalar.get(load)?;
        let rp = self.x.native.layout.get(a.param)?;
        if rp.root != RootType::Table {
            return None;
        }
        let dword = (a.offset / 4) + u32::from(*first);
        self.x
            .native
            .used
            .iter()
            .filter(|u| {
                u.root == RootType::Table
                    && u.user_data == rp.user_data
                    && u.table_offset_dwords <= dword
            })
            .max_by_key(|u| u.table_offset_dwords)
    }

    /// Records the DXIL annotation of a view descriptor loaded from a table at
    /// a constant offset, so the host can place a matching null descriptor when
    /// the title has no view at that slot.
    // shortcut: dynamically indexed table descriptors (`a.dynamic.is_some()`)
    // are not recorded; their empty slots keep the host's default null view.
    fn record_table_view(&mut self, load: Site, first: u8, props: (u32, u32)) {
        let Some(a) = self.scalar.get(&load) else {
            return;
        };
        if a.dynamic.is_some() {
            return;
        }
        let Some(rp) = self.x.native.layout.get(a.param) else {
            return;
        };
        if rp.root != RootType::Table {
            return;
        }
        let dword = (a.offset / 4) + u32::from(first);
        self.table_views
            .entry((rp.slot, dword))
            .and_modify(|cur| *cur = prefer_props(*cur, props))
            .or_insert(props);
    }

    /// A handle for the descriptor `src` read at `site`.
    fn handle(
        &mut self,
        site: Site,
        src: &Source,
        sampler: bool,
        props: (u32, u32),
    ) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => {
                // Keyed by both property words: one descriptor can be read
                // as f32 x4 (a size query) and as u32 (an atomic).
                let key = (self.cur, *load, *first, sampler, props);
                if let Some(&h) = self.handles.get(&key) {
                    if let Some(deps) = self.handle_deps.get(&key) {
                        let deps = deps.clone();
                        self.extend_cur_reads(&deps);
                    }
                    return Ok(h);
                }
                let saved = std::mem::take(&mut self.cur_reads);
                let index = self.heap_index(*load, *first, sampler);
                let sub = std::mem::replace(&mut self.cur_reads, saved);
                let index = index?;
                self.extend_cur_reads(&sub);
                if !sub.is_empty() {
                    self.handle_deps.insert(key, sub);
                }
                // A waterfall loop's load at the first lane's offset reads
                // each lane's own entry (see `resource::Waterfall`).
                let per_lane = self.x.res.waterfall.loads.contains(load);
                // Its offset is only the first lane's, so it is not recorded.
                if !sampler && !per_lane {
                    self.record_table_view(*load, *first, props);
                }
                let h = self
                    .s
                    .heap_handle(self.cur, index, sampler, per_lane, props);
                self.handles.insert(key, h);
                Ok(h)
            }
            Source::Constant(words) if sampler => {
                let words = sampler_words(words)?;
                self.immediate_sampler(words, props)
            }
            Source::Constant(_) => Err("a constant resource descriptor"),
            Source::Built { address, .. } => {
                let rp = *self
                    .x
                    .native
                    .layout
                    .get(address.param)
                    .ok_or("a descriptor built around a table in the code")?;
                let kind = self
                    .x
                    .native
                    .used
                    .iter()
                    .find(|u| u.user_data == rp.user_data)
                    .map(|u| u.kind);
                match (rp.root, kind, props.0) {
                    (RootType::Descriptor, Some(ResourceType::Cbv) | None, 13) => {
                        Ok(self.s.root_cbv_handle(rp.slot))
                    }
                    (RootType::Descriptor, Some(ResourceType::Srv), 11) => {
                        Ok(self.s.root_srv_handle(rp.slot))
                    }
                    _ => Err("a root UAV or typed root view"),
                }
            }
            Source::Phi { .. } => {
                let _ = site;
                let index = self.uniform_heap_index(src, sampler, Some(props))?;
                Ok(self.s.heap_handle(self.cur, index, sampler, false, props))
            }
            Source::Select { scc, args } => {
                // The heap index of each choice, picked by SCC (uniform).
                let [
                    Source::Loaded { load: a, first: fa },
                    Source::Loaded { load: b, first: fb },
                ] = args[..]
                else {
                    return Err("a descriptor chosen by a condition");
                };
                let cond = match self.get(*scc)? {
                    Val::Bit(x) => x,
                    _ => return Err("SCC without a value"),
                };
                let set = self.heap_index(a, fa, sampler)?;
                let clear = self.heap_index(b, fb, sampler)?;
                if !sampler {
                    self.record_table_view(a, fa, props);
                    self.record_table_view(b, fb, props);
                }
                let index = self.s.m.select(self.cur, cond, set, clear);
                Ok(self.s.heap_handle(self.cur, index, sampler, false, props))
            }
            Source::Fetched { .. } | Source::Lanes { .. } | Source::LanePhi { .. } => {
                // A descriptor each lane has its own of (a waterfall loop
                // reads it from the first active lane): each lane's heap
                // index, non-uniform.
                let index = self.lane_heap_index(src, sampler)?;
                Ok(self.s.heap_handle(self.cur, index, sampler, true, props))
            }
        }
    }

    /// The heap index of a descriptor every lane shares, computed in the
    /// current block. One chosen by control flow is a phi at its join of the
    /// index each predecessor computes at its end; an edge may bring another
    /// join's (nested if/else). When `record_props` is `Some` and `!sampler`,
    /// records each loaded leaf for `#704` table view tracking.
    fn uniform_heap_index(
        &mut self,
        src: &Source,
        sampler: bool,
        record_props: Option<(u32, u32)>,
    ) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => {
                let index = self.heap_index(*load, *first, sampler)?;
                if !sampler && let Some(props) = record_props {
                    self.record_table_view(*load, *first, props);
                }
                Ok(index)
            }
            Source::Phi { block, args } => {
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let tail = self.tail[q].ok_or("a descriptor chosen by control flow")?;
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let here = std::mem::replace(&mut self.cur, at);
                    let prev_edge = self.cur_edge.replace((q, *block));
                    let index = self.uniform_heap_index(arg, sampler, record_props);
                    self.cur_edge = prev_edge;
                    self.cur = here;
                    incoming.push((index?, at));
                }
                let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(phi)
            }
            _ => Err("a descriptor chosen by control flow"),
        }
    }

    /// The view-heap slot of `src` when it comes from a descriptor table or
    /// heap (`Loaded` or `Phi`, including nested phis).
    fn heap_slot(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        match src {
            Source::Loaded { .. } | Source::Phi { .. } => {
                self.uniform_heap_index(src, sampler, None)
            }
            _ => Err("a descriptor outside the heaps"),
        }
    }

    /// The entry of the host's runtime stride table (`t0`,
    /// [`super::STRIDE_SPACE`]) for the view-heap slot of `src`, read in the
    /// current block: the slot holding a raw view of the buffer's bytes
    /// (the slot itself when the table names none), the stride and the bias
    /// ([`super::StrideEntry`]).
    fn stride_entry(&mut self, src: &Source) -> Result<[V; 3], Why> {
        let key = match src {
            Source::Loaded { load, first } => Some((self.cur, *load, *first)),
            _ => None,
        };
        if let Some(e) = key.and_then(|k| self.entries.get(&k)) {
            return Ok(*e);
        }
        let slot = self
            .heap_slot(src, false)
            .map_err(|_| "a structured buffer of unknown stride")?;
        let c = self.cur;
        let table = self.s.stride_table_handle();
        let bytes = self.s.m.i32(super::StrideEntry::BYTES);
        let addr = self.s.m.bin(c, Bin::Mul, slot, bytes);
        let iu = self.s.m.undef(self.s.t.i32);
        let mask = self.s.m.i8(3);
        let align = self.s.m.i32(4);
        let res = self.s.op(
            c,
            "rawBufferLoad.f32",
            Attr::ReadOnly,
            self.s.t.res_ret,
            139,
            &[table, addr, iu, mask, align],
        );
        let stride = self.s.m.extract(c, res, 0);
        let stride = self.as_int(stride);
        let word = self.s.m.extract(c, res, 1);
        let word = self.as_int(word);
        // The raw slot plus one under the bias; 0: the slot itself.
        let low = self.s.m.i32(super::StrideEntry::RAW_SLOTS);
        let named = self.s.m.bin(c, Bin::And, word, low);
        let zero = self.s.m.i32(0);
        let own = self.s.m.cmp(c, pred::EQ, named, zero);
        let one = self.s.m.i32(1);
        let named = self.s.m.bin(c, Bin::Sub, named, one);
        let raw = self.s.m.select(c, own, slot, named);
        let k = self.s.m.i32(28);
        let bias = self.s.m.bin(c, Bin::LShr, word, k);
        let e = [raw, stride, bias];
        if let Some(k) = key {
            self.entries.insert(k, e);
        }
        Ok(e)
    }

    /// The raw handle (`props`: 11, with the UAV bit for one written) and
    /// the byte offset of `off` in element `index` of a buffer whose stride
    /// was not recovered statically. Both come from the host's runtime
    /// stride table: the handle is of the slot the table names, which holds
    /// a raw view of the buffer's bytes whatever view the title made (raw,
    /// structured or typed), and the byte is `index * stride + off` past
    /// that view's bias. The raw view ends with the title's view, so an
    /// index past its elements reads zeros (and its writes are dropped).
    /// shortcut: the console reads zeros (and drops writes) where `off` and
    /// the access pass the element's end; here they reach the next element,
    /// as the statically known strides' structured accesses may.
    fn runtime_view(
        &mut self,
        src: &Source,
        index: V,
        off: V,
        props: u32,
    ) -> Result<(V, V), Why> {
        let [raw, stride, bias] = self.stride_entry(src)?;
        let c = self.cur;
        let element = self.s.m.bin(c, Bin::Mul, index, stride);
        let byte = self.s.m.bin(c, Bin::Add, element, off);
        let byte = self.s.m.bin(c, Bin::Add, byte, bias);
        // The raw alias varies with the heap slot of a waterfall load.
        let from_waterfall = |arg: &Source| {
            matches!(arg, Source::Loaded { load, .. } if self.x.res.waterfall.loads.contains(load))
        };
        let nonuniform = match src {
            Source::Phi { args, .. } => args.iter().any(from_waterfall),
            _ => from_waterfall(src),
        };
        let handle = self.s.heap_handle(c, raw, false, nonuniform, (props, 0));
        Ok((handle, byte))
    }

    /// The handle of the immediate sampler `words` (a static sampler).
    fn immediate_sampler(&mut self, words: [u32; 4], props: (u32, u32)) -> Result<V, Why> {
        let n = match self.samplers.iter().position(|w| *w == words) {
            Some(n) => n,
            None => {
                self.samplers.push(words);
                self.samplers.len() - 1
            }
        };
        if props.0 & 0x8000 != 0 {
            // shortcut: one sampler used with and without comparison
            // would need two static samplers; it becomes a comparison one.
            self.s.cmp_samplers |= 1 << n;
        }
        Ok(self.s.sampler_handle(self.cur, n as u32))
    }

    /// The handles of the sampler `src` read at `site`, and which one each
    /// lane uses: one handle, or, for immediate samplers control flow chose
    /// among (each predecessor of the join naming a constant one), one per
    /// choice and the choice's position, picked at the join. Static samplers
    /// are one register each, so no shader array can index them; the
    /// sampling instruction runs once per choice and [`Lower::picked`] takes
    /// the chosen result. The choice is uniform, as the edge is.
    fn sampler_choice(
        &mut self,
        site: Site,
        src: &Source,
        props: (u32, u32),
    ) -> Result<(Vec<V>, Option<V>), Why> {
        let Source::Phi { block, args } = src else {
            return Ok((vec![self.handle(site, src, true, props)?], None));
        };
        if !args.iter().all(|a| matches!(a, Source::Constant(_))) {
            return Ok((vec![self.handle(site, src, true, props)?], None));
        }
        let mut edges = Vec::new();
        let mut choices: Vec<[u32; 4]> = Vec::new();
        for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
            if self.block[q].is_none() {
                continue;
            }
            let Source::Constant(words) = arg else {
                unreachable!("checked above");
            };
            let words = sampler_words(words)?;
            let tail = self.tail[q].ok_or("a descriptor chosen by control flow")?;
            let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
            let k = match choices.iter().position(|c| *c == words) {
                Some(k) => k,
                None => {
                    choices.push(words);
                    choices.len() - 1
                }
            };
            edges.push((at, k));
        }
        let mut handles = Vec::new();
        for words in &choices {
            handles.push(self.immediate_sampler(*words, props)?);
        }
        if handles.len() < 2 {
            return Ok((handles, None));
        }
        let join = self.block[*block].ok_or("a descriptor chosen by control flow")?;
        let mut incoming = Vec::new();
        for (at, k) in edges {
            incoming.push((self.s.m.i32(k as u32), at));
        }
        let phi = self.s.m.phi(join, self.s.t.i32);
        self.s.m.set_phi(phi, incoming);
        Ok((handles, Some(phi)))
    }

    /// Of `xs`, one value per handle [`Lower::sampler_choice`] gave, the one
    /// at position `pick` (the first without one).
    fn picked(&mut self, c: usize, pick: Option<V>, xs: &[V]) -> V {
        let mut x = xs[0];
        if let Some(pick) = pick {
            for (k, &other) in xs.iter().enumerate().skip(1) {
                let k = self.s.m.i32(k as u32);
                let here = self.s.m.cmp(c, pred::EQ, pick, k);
                x = self.s.m.select(c, here, other, x);
            }
        }
        x
    }

    /// The heap index of a descriptor that may differ between lanes,
    /// computed in the current block (or, chosen by control flow, at the
    /// join).
    fn lane_heap_index(&mut self, src: &Source, sampler: bool) -> Result<V, Why> {
        match src {
            Source::Loaded { load, first } => self.heap_index(*load, *first, sampler),
            Source::Fetched { load, first } => self.fetched_index(*load, *first, sampler),
            Source::Lanes { mask, args } | Source::Select { scc: mask, args } => {
                let [a, b] = &args[..] else {
                    return Err("a descriptor chosen per lane");
                };
                let cond = if matches!(src, Source::Lanes { .. }) {
                    let &m = self
                        .x
                        .st
                        .masks
                        .get(mask)
                        .ok_or("a descriptor chosen per lane by an untracked mask")?;
                    self.boolean(m, self.cur)?
                } else {
                    match self.get(*mask)? {
                        Val::Bit(x) => x,
                        _ => return Err("SCC without a value"),
                    }
                };
                let set = self.lane_heap_index(a, sampler)?;
                let clear = self.lane_heap_index(b, sampler)?;
                Ok(self.s.m.select(self.cur, cond, set, clear))
            }
            Source::LanePhi { block, args } => {
                // Each predecessor computes its heap index at its end (what
                // it reads is there: the dwords are, on that edge); the join
                // picks the one of the edge the lane came by. An edge no
                // lane reading it takes gives index 0.
                let join =
                    self.block[*block].ok_or("a per-lane descriptor chosen by control flow")?;
                let mut incoming = Vec::new();
                for (&q, arg) in self.x.p.blocks[*block].preds.iter().zip(args) {
                    if self.block[q].is_none() {
                        continue;
                    }
                    let tail =
                        self.tail[q].ok_or("a per-lane descriptor joined from a later block")?;
                    let at = self.edges.get(&(q, *block)).copied().unwrap_or(tail);
                    let index = match arg {
                        Some(arg) => {
                            let here = std::mem::replace(&mut self.cur, at);
                            let prev_edge = self.cur_edge.replace((q, *block));
                            let index = self.lane_heap_index(arg, sampler);
                            self.cur_edge = prev_edge;
                            self.cur = here;
                            index?
                        }
                        None => self.s.m.i32(0),
                    };
                    incoming.push((index, at));
                }
                let phi = self.s.m.phi(join, self.s.t.i32);
                self.s.m.set_phi(phi, incoming);
                Ok(phi)
            }
            _ => Err("a per-lane descriptor from neither a heap load nor a choice of them"),
        }
    }

    /// The heap index of the descriptor at dword `first` of what the global
    /// load at `load` reads, in each lane: its address (base plus offsets)
    /// over the descriptor size.
    fn fetched_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        const WHY: Why = "a per-lane descriptor load";
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Flat {
            addr: Some(Operand::Vector { reg, .. }),
            saddr,
            offset,
            ..
        } = li.fields
        else {
            return Err(WHY);
        };
        let c = self.cur;
        // The low dword of the address: SGPR base plus VGPR offset, or the
        // VGPR pair's low half (heap addresses fit in 32 bits).
        let mut at = self.i32_of(li.read(Reg::V(reg)).ok_or(WHY)?)?;
        if let Some(Operand::Scalar { code, .. }) = saddr
            && code != NULL
        {
            let base = self.i32_of(li.read(Reg::S(code)).ok_or(WHY)?)?;
            at = self.s.m.bin(c, Bin::Add, base, at);
        }
        let k = self
            .s
            .m
            .i32((offset as i32 as u32).wrapping_add(4 * u32::from(first)));
        at = self.s.m.bin(c, Bin::Add, at, k);
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(c, Bin::LShr, at, shift))
    }

    /// The heap index of the descriptor at dword `first` of what the scalar
    /// load at `load` reads, computed in the current block.
    fn heap_index(&mut self, load: Site, first: u8, sampler: bool) -> Result<V, Why> {
        let a = *self
            .scalar
            .get(&load)
            .ok_or("an untracked descriptor load")?;
        match self.x.native.layout.get(a.param).map(|p| p.root) {
            Some(RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap) => {}
            _ => return Err("a descriptor outside the heaps"),
        }
        let li: &Inst = &self.x.p.blocks[load.0].insts[load.1];
        let Fields::Smem {
            base,
            offset,
            soffset,
            ..
        } = &li.fields
        else {
            return Err("a descriptor load");
        };
        let Operand::Scalar { code, .. } = *base else {
            return Err("a descriptor load");
        };
        let c = self.cur;
        let base = li.read(Reg::S(code)).ok_or("a descriptor load")?;
        let mut addr = self.i32_of(base)?;
        let k = self
            .s
            .m
            .i32((*offset as u32).wrapping_add(4 * u32::from(first)));
        addr = self.s.m.bin(c, Bin::Add, addr, k);
        if !matches!(soffset, Operand::Scalar { code: NULL, .. }) {
            let so = self.operand(li, soffset)?;
            addr = self.s.m.bin(c, Bin::Add, addr, so);
        }
        let shift = self.s.m.i32(if sampler { 4 } else { 5 });
        Ok(self.s.m.bin(c, Bin::LShr, addr, shift))
    }

    fn valu(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        if self.x.res.waterfall.compares.contains(&(b, i)) {
            // The first lane's dwords of a waterfall loop are each lane's
            // own (see `resource::Waterfall`).
            self.lane(inst, Some(true));
            return Ok(());
        }
        if let Some(rest) = name.strip_prefix("v_cmp_").or(name.strip_prefix("v_cmpx_")) {
            let r = if v.dpp.is_some() {
                Err("a compare with DPP")
            } else {
                self.compare(inst, rest)
            };
            if let Err(why) = r {
                for &(_, w) in &inst.writes {
                    self.lane_fail.insert(w, why);
                }
            }
            return r;
        }
        let mut guard = self.guard(b, i)?;
        if let Some(d) = v.dpp {
            let (x, enable) = self.dpp(inst, d)?;
            guard = match (guard, enable) {
                (Some(g), Some(e)) => Some(self.s.m.bin(self.cur, Bin::And, g, e)),
                (g, e) => g.or(e),
            };
            self.src0 = Some(x);
        }
        let r = self.valu_op(inst, guard);
        self.src0 = None;
        r
    }

    /// DPP: src0 as read from another lane, and which lanes write their
    /// result (none: all).
    fn dpp(&mut self, inst: &Inst, dpp: Dpp) -> Result<(V, Option<V>), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(&Operand::Vector { reg, .. }) = v.src.first() else {
            return Err("a DPP operand");
        };
        let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("a DPP operand")?)?;
        // Quad permutations and DPP8 stay in groups the pixel layout shares.
        let quad = matches!(dpp, Dpp::Row { ctrl: 0..=0xFF, .. });
        let mesh_quad = self.s.mesh.is_some() && quad;
        if !mesh_quad {
            self.cross_lane(quad)?;
        }
        let c = self.cur;
        let lane = self.lane_index();
        let k = |l: &mut Self, n: u32| l.s.m.i32(n);
        let (sixteen_mask, low) = (k(self, !15), k(self, 15));
        let row = self.s.m.bin(c, Bin::And, lane, sixteen_mask);
        let pos = self.s.m.bin(c, Bin::And, lane, low);
        // Source lane, and whether it lies in the row.
        let (src, inside, masks, zero) = match dpp {
            Dpp::Lanes { sel, .. } => {
                let (seven, not7, three) = (k(self, 7), k(self, !7), k(self, 3));
                let p = self.s.m.bin(c, Bin::And, lane, seven);
                let sh = self.s.m.bin(c, Bin::Mul, p, three);
                let sel = k(self, sel);
                let pick = self.s.m.bin(c, Bin::LShr, sel, sh);
                let pick = self.s.m.bin(c, Bin::And, pick, seven);
                let base = self.s.m.bin(c, Bin::And, lane, not7);
                (self.s.m.bin(c, Bin::Or, base, pick), None, None, false)
            }
            Dpp::Row {
                ctrl,
                row_mask,
                bank_mask,
                bound_ctrl,
                ..
            } => {
                let n = u32::from(ctrl & 15);
                let (src, inside) = match ctrl {
                    0..=0xFF => {
                        let (three, one, not3) = (k(self, 3), k(self, 1), k(self, !3));
                        let p = self.s.m.bin(c, Bin::And, lane, three);
                        let sh = self.s.m.bin(c, Bin::Shl, p, one);
                        let perm = k(self, u32::from(ctrl));
                        let pick = self.s.m.bin(c, Bin::LShr, perm, sh);
                        let pick = self.s.m.bin(c, Bin::And, pick, three);
                        let base = self.s.m.bin(c, Bin::And, lane, not3);
                        (self.s.m.bin(c, Bin::Or, base, pick), None)
                    }
                    0x101..=0x10F | 0x111..=0x11F => {
                        let nn = k(self, n);
                        let left = ctrl < 0x110;
                        let (op, p) = if left {
                            (Bin::Add, pred::IULT)
                        } else {
                            (Bin::Sub, pred::IUGE)
                        };
                        let s = self.s.m.bin(c, op, lane, nn);
                        // Left: pos + n < 16; right: pos >= n.
                        let bound = if left { k(self, 16 - n) } else { nn };
                        (s, Some(self.s.m.cmp(c, p, pos, bound)))
                    }
                    0x121..=0x12F => {
                        let nn = k(self, n);
                        let t = self.s.m.bin(c, Bin::Sub, pos, nn);
                        let t = self.s.m.bin(c, Bin::And, t, low);
                        (self.s.m.bin(c, Bin::Or, row, t), None)
                    }
                    0x140 | 0x141 => {
                        let m = k(self, if ctrl == 0x140 { 15 } else { 7 });
                        let t = self.s.m.bin(c, Bin::Xor, lane, m);
                        (t, None)
                    }
                    0x150..=0x15F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Or, row, nn), None)
                    }
                    0x160..=0x16F => {
                        let nn = k(self, n);
                        (self.s.m.bin(c, Bin::Xor, lane, nn), None)
                    }
                    _ => return Err("a DPP lane pattern"),
                };
                let masks = (row_mask != 0xF || bank_mask != 0xF).then_some((row_mask, bank_mask));
                (src, inside, masks, bound_ctrl)
            }
        };
        let mut r = if mesh_quad {
            self.quad_pick(x, src)
        } else {
            self.read_lane(x, src)
        };
        let mut enable = None;
        if let Some(inside) = inside {
            if zero {
                let z = k(self, 0);
                r = self.s.m.select(c, inside, r, z);
            } else {
                enable = Some(inside);
            }
        }
        if let Some((rows, banks)) = masks {
            // Row (16 lanes) and bank (4 lanes of a row) write masks.
            let (four, two, three) = (k(self, 4), k(self, 2), k(self, 3));
            let ri = self.s.m.bin(c, Bin::LShr, lane, four);
            let ri = self.s.m.bin(c, Bin::And, ri, three);
            let bi = self.s.m.bin(c, Bin::LShr, lane, two);
            let bi = self.s.m.bin(c, Bin::And, bi, three);
            let (rm, bm) = (k(self, u32::from(rows)), k(self, u32::from(banks)));
            let rb = self.s.m.bin(c, Bin::LShr, rm, ri);
            let bb = self.s.m.bin(c, Bin::LShr, bm, bi);
            let both = self.s.m.bin(c, Bin::And, rb, bb);
            let (one, z) = (k(self, 1), k(self, 0));
            let both = self.s.m.bin(c, Bin::And, both, one);
            let on = self.s.m.cmp(c, pred::NE, both, z);
            enable = Some(match enable {
                Some(e) => self.s.m.bin(c, Bin::And, e, on),
                None => on,
            });
        }
        Ok((r, enable))
    }

    fn valu_op(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let name = inst.name();
        let c = self.cur;
        let f = |l: &mut Self, k| l.fsrc(inst, k);
        let s = |l: &mut Self, k| l.src(inst, k);
        let float = |l: &mut Self, r: V| l.write_f(inst, r, guard);
        let int = |l: &mut Self, r: V| l.write_i(inst, r, guard);
        if let Some(r) = self.valu16(inst, name, guard) {
            return r;
        }
        if name == "v_pk_fmac_f16" {
            return self.pk_fmac_f16(inst, guard);
        }
        if let Some(op) = name.strip_prefix("v_pk_") {
            return self.packed16(inst, op, guard);
        }
        match name {
            "v_lshlrev_b64" | "v_lshrrev_b64" | "v_ashrrev_i64" => {
                let n = s(self, 0)?;
                let x = self.src64(inst, 1)?;
                let i64t = self.s.m.int(64);
                let k63 = self.s.m.i32(63);
                let n = self.s.m.bin(c, Bin::And, n, k63);
                let n = self.s.m.cast(c, Cast::ZExt, n, i64t);
                let op = match name {
                    "v_lshlrev_b64" => Bin::Shl,
                    "v_lshrrev_b64" => Bin::LShr,
                    _ => Bin::AShr,
                };
                let r = self.s.m.bin(c, op, x, n);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a 64-bit result");
                };
                let (lo, hi) = self.split64(r);
                self.write(inst, Reg::V(reg), lo, guard);
                self.write(inst, Reg::V(reg + 1), hi, guard);
                Ok(())
            }
            "v_cubeid_f32" | "v_cubesc_f32" | "v_cubetc_f32" | "v_cubema_f32" => {
                // Cube-map face coordinates (AMD's ISA): the major axis is z
                // when |z| >= |x| and |z| >= |y|, else y when |y| >= |x|,
                // else x.
                let (x, y, z) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let (ax, ay, az) = (self.unary(6, x), self.unary(6, y), self.unary(6, z));
                let zx = self.s.m.cmp(c, pred::OGE, az, ax);
                let zy = self.s.m.cmp(c, pred::OGE, az, ay);
                let zmaj = self.s.m.bin(c, Bin::And, zx, zy);
                let ymaj = self.s.m.cmp(c, pred::OGE, ay, ax);
                let zero = self.s.m.f32(0);
                let neg = |l: &mut Self, v: V| l.s.m.cmp(c, pred::OLT, v, zero);
                let minus = |l: &mut Self, v: V| {
                    let z = l.s.m.f32(0x8000_0000);
                    l.s.m.bin(c, Bin::Sub, z, v)
                };
                let pick = |l: &mut Self, zv: V, yv: V, xv: V| {
                    let t = l.s.m.select(c, ymaj, yv, xv);
                    l.s.m.select(c, zmaj, zv, t)
                };
                let r = match name {
                    "v_cubeid_f32" => {
                        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
                        let (nz, ny, nx) = (neg(self, z), neg(self, y), neg(self, x));
                        let (f5, f4, f3, f2, f1, f0) = (
                            k(self, 5.0),
                            k(self, 4.0),
                            k(self, 3.0),
                            k(self, 2.0),
                            k(self, 1.0),
                            k(self, 0.0),
                        );
                        let zf = self.s.m.select(c, nz, f5, f4);
                        let yf = self.s.m.select(c, ny, f3, f2);
                        let xf = self.s.m.select(c, nx, f1, f0);
                        pick(self, zf, yf, xf)
                    }
                    "v_cubesc_f32" => {
                        let (nz, nx) = (neg(self, z), neg(self, x));
                        let (mx, mz) = (minus(self, x), minus(self, z));
                        let zf = self.s.m.select(c, nz, mx, x);
                        let xf = self.s.m.select(c, nx, z, mz);
                        pick(self, zf, x, xf)
                    }
                    "v_cubetc_f32" => {
                        let ny = neg(self, y);
                        let (my, mz) = (minus(self, y), minus(self, z));
                        let yf = self.s.m.select(c, ny, mz, z);
                        pick(self, my, yf, my)
                    }
                    _ => {
                        let t = pick(self, z, y, x);
                        let two = self.s.m.f32(2.0f32.to_bits());
                        self.s.m.bin(c, Bin::Mul, t, two)
                    }
                };
                float(self, r)
            }
            "v_movrels_b32" | "v_movreld_b32" | "v_movrelsd_b32" | "v_movrelsd_2_b32" => {
                // M0 indexes the source (movrels), the destination (movreld)
                // or both, from the encoded register up to the program's
                // VGPR count (the registers the instruction reads or writes).
                // movrelsd_2 indexes them apart: the source by M0[9:0], the
                // destination by M0[25:16].
                let m0 = self.i32_of(inst.read(Reg::S(crate::ir::M0)).ok_or("M0")?)?;
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("an M0-relative move");
                };
                let (si, di) = if name == "v_movrelsd_2_b32" {
                    self.m0_halves(m0)
                } else {
                    (m0, m0)
                };
                let value = if name != "v_movreld_b32" {
                    self.rel_read(inst, sreg, si)?
                } else {
                    s(self, 0)?
                };
                if name == "v_movrels_b32" {
                    return int(self, value);
                }
                for k in 0..=255u8.saturating_sub(dreg) {
                    let Some(_) = inst.write(Reg::V(dreg + k)) else {
                        break;
                    };
                    let old = self.old_vgpr(inst, dreg + k)?;
                    let kk = self.s.m.i32(u32::from(k));
                    let is = self.s.m.cmp(c, pred::EQ, di, kk);
                    let x = self.s.m.select(c, is, value, old);
                    self.write_placed(inst, Reg::V(dreg + k), x, guard);
                }
                Ok(())
            }
            "v_swaprel_b32" => {
                // v[dst + M0[25:16]] and v[src + M0[9:0]] trade values. The
                // console writes the destination first and the source last,
                // so a register both name keeps its value.
                let m0 = self.i32_of(inst.read(Reg::S(crate::ir::M0)).ok_or("M0")?)?;
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("an M0-relative swap");
                };
                let (si, di) = self.m0_halves(m0);
                let to_dst = self.rel_read(inst, sreg, si)?;
                let to_src = self.rel_read(inst, dreg, di)?;
                let sbase = self.s.m.i32(u32::from(sreg));
                let dbase = self.s.m.i32(u32::from(dreg));
                let at_src = self.s.m.bin(c, Bin::Add, sbase, si);
                let at_dst = self.s.m.bin(c, Bin::Add, dbase, di);
                // Both ranges run to the program's last VGPR: their union
                // starts at the lower base.
                let first = sreg.min(dreg);
                for k in 0..=255u8.saturating_sub(first) {
                    let r = first + k;
                    let Some(_) = inst.write(Reg::V(r)) else {
                        break;
                    };
                    let old = self.old_vgpr(inst, r)?;
                    let rr = self.s.m.i32(u32::from(r));
                    let is_dst = self.s.m.cmp(c, pred::EQ, at_dst, rr);
                    let x = self.s.m.select(c, is_dst, to_dst, old);
                    let is_src = self.s.m.cmp(c, pred::EQ, at_src, rr);
                    let x = self.s.m.select(c, is_src, to_src, x);
                    self.write_placed(inst, Reg::V(r), x, guard);
                }
                Ok(())
            }
            "v_swap_b32" => {
                let (
                    Some(&Operand::Vector { reg: sreg, .. }),
                    Some(Operand::Vector { reg: dreg, .. }),
                ) = (v.src.first(), v.dst)
                else {
                    return Err("v_swap_b32");
                };
                let a = self.i32_of(inst.read(Reg::V(dreg)).ok_or("v_swap_b32")?)?;
                let b = self.i32_of(inst.read(Reg::V(sreg)).ok_or("v_swap_b32")?)?;
                self.write_placed(inst, Reg::V(dreg), b, guard);
                if dreg != sreg {
                    self.write_placed(inst, Reg::V(sreg), a, guard);
                }
                Ok(())
            }
            "v_ffbh_u32" | "v_ffbh_i32" => {
                // Counted from the top bit, -1 when none (DXIL's too).
                let a = s(self, 0)?;
                let op = if name == "v_ffbh_u32" { 33 } else { 34 };
                let r = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, op, &[a]);
                int(self, r)
            }
            "v_mov_b32" => {
                if v.abs != 0 || v.neg != 0 || v.clamp || v.omod != 0 {
                    let a = f(self, 0)?;
                    return float(self, a);
                }
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_add_f32" | "v_sub_f32" | "v_subrev_f32" | "v_mul_f32" | "v_mul_legacy_f32"
            | "v_max_f32" | "v_min_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let r = match name {
                    "v_add_f32" => self.s.m.bin(c, Bin::Add, a, bb),
                    "v_sub_f32" => self.s.m.bin(c, Bin::Sub, a, bb),
                    "v_subrev_f32" => self.s.m.bin(c, Bin::Sub, bb, a),
                    "v_mul_f32" => self.s.m.bin(c, Bin::Mul, a, bb),
                    "v_mul_legacy_f32" => self.legacy_mul(a, bb),
                    "v_max_f32" => self.max_min(true, a, bb),
                    _ => self.max_min(false, a, bb),
                };
                float(self, r)
            }
            "v_fma_f32" | "v_fmaak_f32" | "v_fmamk_f32" => {
                // shortcut: DXIL has no fused f32 multiply-add; FMad lets the driver fuse.
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = self.fmad(a, bb, cc);
                float(self, r)
            }
            "v_dot2_i32_i16" | "v_dot2_u32_u16" | "v_dot4_i32_i8" | "v_dot4_u32_u8"
            | "v_dot4c_i32_i8" | "v_dot8_u32_u4" => {
                if v.omod != 0 {
                    return Err("integer output modifiers");
                }
                let (a, bb) = (s(self, 0)?, s(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a result without a register");
                };
                let acc = if name == "v_dot4c_i32_i8" {
                    let prev = inst.read(Reg::V(reg)).ok_or("an accumulator")?;
                    self.i32_of(prev)?
                } else {
                    s(self, 2)?
                };
                let signed = matches!(name, "v_dot2_i32_i16" | "v_dot4_i32_i8" | "v_dot4c_i32_i8");
                let (lanes, width): (u32, u32) = match name {
                    "v_dot2_i32_i16" | "v_dot2_u32_u16" => (2, 16),
                    "v_dot4_i32_i8" | "v_dot4_u32_u8" | "v_dot4c_i32_i8" => (4, 8),
                    _ => (8, 4),
                };
                let extract = |l: &mut Self, x: V, lane: u32| -> V {
                    if signed {
                        let shl = 32 - width - lane * width;
                        let t = if shl != 0 {
                            let k = l.s.m.i32(shl);
                            l.s.m.bin(c, Bin::Shl, x, k)
                        } else {
                            x
                        };
                        let shr = l.s.m.i32(32 - width);
                        l.s.m.bin(c, Bin::AShr, t, shr)
                    } else {
                        let t = if lane != 0 {
                            let k = l.s.m.i32(lane * width);
                            l.s.m.bin(c, Bin::LShr, x, k)
                        } else {
                            x
                        };
                        if width < 32 {
                            let mask = l.s.m.i32((1u32 << width) - 1);
                            l.s.m.bin(c, Bin::And, t, mask)
                        } else {
                            t
                        }
                    }
                };
                let mut products = Vec::with_capacity(lanes as usize);
                for lane in 0..lanes {
                    let ea = extract(self, a, lane);
                    let eb = extract(self, bb, lane);
                    products.push(self.s.m.bin(c, Bin::Mul, ea, eb));
                }
                let r = if !v.clamp {
                    let mut sum = acc;
                    for p in products {
                        sum = self.s.m.bin(c, Bin::Add, sum, p);
                    }
                    sum
                } else if signed {
                    // Exact 64-bit signed accumulation in two 32-bit halves
                    // (no INT64 capability needed), saturated to [i32::MIN, i32::MAX].
                    let (k0, k1, k31) = (self.s.m.i32(0), self.s.m.i32(1), self.s.m.i32(31));
                    let mut lo = acc;
                    let mut hi = self.s.m.bin(c, Bin::AShr, acc, k31);
                    let terms: Vec<V> = if lanes == 2 {
                        products
                    } else {
                        // 4x i8*i8 sum lies in [-65024, 65536], always exact in i32.
                        let mut dot = products[0];
                        for &p in &products[1..] {
                            dot = self.s.m.bin(c, Bin::Add, dot, p);
                        }
                        vec![dot]
                    };
                    for t in terms {
                        let new_lo = self.s.m.bin(c, Bin::Add, lo, t);
                        let carry_b = self.s.m.cmp(c, pred::IULT, new_lo, lo);
                        let carry = self.s.m.select(c, carry_b, k1, k0);
                        let t_hi = self.s.m.bin(c, Bin::AShr, t, k31);
                        let hi1 = self.s.m.bin(c, Bin::Add, hi, t_hi);
                        hi = self.s.m.bin(c, Bin::Add, hi1, carry);
                        lo = new_lo;
                    }
                    let s_hi = self.s.m.bin(c, Bin::AShr, lo, k31);
                    let fits = self.s.m.cmp(c, pred::EQ, hi, s_hi);
                    let pos = self.s.m.cmp(c, pred::SGE, hi, k0);
                    let (imin, imax) = (
                        self.s.m.i32(i32::MIN as u32),
                        self.s.m.i32(i32::MAX as u32),
                    );
                    let sat = self.s.m.select(c, pos, imax, imin);
                    self.s.m.select(c, fits, lo, sat)
                } else {
                    let terms: Vec<V> = if lanes == 2 {
                        products
                    } else {
                        // 4x u8*u8 (<= 260100) and 8x u4*u4 (<= 1800) never overflow u32.
                        let mut dot = products[0];
                        for &p in &products[1..] {
                            dot = self.s.m.bin(c, Bin::Add, dot, p);
                        }
                        vec![dot]
                    };
                    let mut lo = acc;
                    let mut ov = self.s.m.bool(false);
                    for t in terms {
                        let new_lo = self.s.m.bin(c, Bin::Add, lo, t);
                        let carry = self.s.m.cmp(c, pred::IULT, new_lo, lo);
                        ov = self.s.m.bin(c, Bin::Or, ov, carry);
                        lo = new_lo;
                    }
                    let umax = self.s.m.i32(u32::MAX);
                    self.s.m.select(c, ov, umax, lo)
                };
                self.write(inst, Reg::V(reg), r, guard);
                Ok(())
            }
            "v_mad_u32_u16" | "v_mad_i32_i16" => {
                if v.omod != 0 || v.abs != 0 || v.neg != 0 {
                    return Err("integer output modifiers");
                }
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a result without a register");
                };
                let (mut a, mut bb) = (self.half_int_src(inst, 0)?, self.half_int_src(inst, 1)?);
                let acc = s(self, 2)?;
                let signed = name == "v_mad_i32_i16";
                if signed {
                    let k16 = self.s.m.i32(16);
                    let ta = self.s.m.bin(c, Bin::Shl, a, k16);
                    a = self.s.m.bin(c, Bin::AShr, ta, k16);
                    let tb = self.s.m.bin(c, Bin::Shl, bb, k16);
                    bb = self.s.m.bin(c, Bin::AShr, tb, k16);
                }
                let prod = self.s.m.bin(c, Bin::Mul, a, bb);
                let sum = self.s.m.bin(c, Bin::Add, prod, acc);
                let r = if !v.clamp {
                    sum
                } else if signed {
                    let k0 = self.s.m.i32(0);
                    let pa_xor = self.s.m.bin(c, Bin::Xor, prod, acc);
                    let same_sign = self.s.m.cmp(c, pred::SGE, pa_xor, k0);
                    let ps_xor = self.s.m.bin(c, Bin::Xor, prod, sum);
                    let flipped = self.s.m.cmp(c, pred::SLT, ps_xor, k0);
                    let ov = self.s.m.bin(c, Bin::And, same_sign, flipped);
                    let pos = self.s.m.cmp(c, pred::SGE, prod, k0);
                    let (imin, imax) = (
                        self.s.m.i32(i32::MIN as u32),
                        self.s.m.i32(i32::MAX as u32),
                    );
                    let sat = self.s.m.select(c, pos, imax, imin);
                    self.s.m.select(c, ov, sat, sum)
                } else {
                    let ov = self.s.m.cmp(c, pred::IULT, sum, acc);
                    let umax = self.s.m.i32(u32::MAX);
                    self.s.m.select(c, ov, umax, sum)
                };
                self.write(inst, Reg::V(reg), r, guard);
                Ok(())
            }
            "v_fmac_f32" | "v_mac_f32" | "v_mac_legacy_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let acc = inst.read(Reg::V(reg)).ok_or("an accumulator")?;
                let acc = self.float_of(acc)?;
                let r = match name {
                    "v_fmac_f32" => self.fmad(a, bb, acc),
                    "v_mac_f32" => {
                        let m = self.s.m.bin(c, Bin::Mul, a, bb);
                        self.s.m.bin(c, Bin::Add, m, acc)
                    }
                    _ => {
                        let m = self.legacy_mul(a, bb);
                        self.s.m.bin(c, Bin::Add, m, acc)
                    }
                };
                float(self, r)
            }
            "v_mad_f32" | "v_madak_f32" | "v_madmk_f32" | "v_mad_legacy_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let m = if name == "v_mad_legacy_f32" {
                    self.legacy_mul(a, bb)
                } else {
                    self.s.m.bin(c, Bin::Mul, a, bb)
                };
                let r = self.s.m.bin(c, Bin::Add, m, cc);
                float(self, r)
            }
            "v_max3_f32" | "v_min3_f32" | "v_med3_f32" | "v_div_fixup_f32" => {
                let (a, bb, cc) = (f(self, 0)?, f(self, 1)?, f(self, 2)?);
                let r = match name {
                    "v_max3_f32" => {
                        let t = self.max_min(true, a, bb);
                        self.max_min(true, t, cc)
                    }
                    "v_min3_f32" => {
                        let t = self.max_min(false, a, bb);
                        self.max_min(false, t, cc)
                    }
                    "v_div_fixup_f32" => self.div_fixup(a, bb, cc),
                    _ => self.med3(a, bb, cc),
                };
                float(self, r)
            }
            "v_rcp_f32" | "v_rcp_iflag_f32" => {
                // shortcut: the hardware reciprocal is approximate (1 ULP); this divides.
                let a = f(self, 0)?;
                let one = self.s.m.f32(1.0f32.to_bits());
                let r = self.s.m.bin(c, Bin::SDiv, one, a);
                float(self, r)
            }
            "v_rsq_f32" | "v_sqrt_f32" | "v_log_f32" | "v_exp_f32" | "v_fract_f32"
            | "v_trunc_f32" | "v_floor_f32" | "v_ceil_f32" | "v_rndne_f32" | "v_sin_f32"
            | "v_cos_f32" => {
                let a = f(self, 0)?;
                let op = match name {
                    "v_rsq_f32" => 25,
                    "v_sqrt_f32" => 24,
                    "v_log_f32" => 23,
                    "v_exp_f32" => 21,
                    "v_fract_f32" => 22,
                    "v_trunc_f32" => 29,
                    "v_floor_f32" => 27,
                    "v_ceil_f32" => 28,
                    "v_rndne_f32" => 26,
                    "v_sin_f32" => 13,
                    _ => 12,
                };
                let r = match op {
                    12 | 13 => self.trig_turns(a, op == 12),
                    22 => self.fract(a, 1.0, f32::from_bits(0x3F7F_FFFF)),
                    _ => self.unary(op, a),
                };
                float(self, r)
            }
            "v_frexp_mant_f32" => {
                let a = f(self, 0)?;
                let (mant, _) = self.frexp_f32(a);
                float(self, mant)
            }
            "v_frexp_exp_i32_f32" => {
                let a = f(self, 0)?;
                let (_, exp) = self.frexp_f32(a);
                int(self, exp)
            }
            "v_cvt_i32_f32" | "v_cvt_u32_f32" | "v_cvt_flr_i32_f32" | "v_cvt_rpi_i32_f32" => {
                let mut a = f(self, 0)?;
                if name == "v_cvt_rpi_i32_f32" {
                    let h = self.s.m.f32(0.5f32.to_bits());
                    a = self.s.m.bin(c, Bin::Add, a, h);
                }
                if name != "v_cvt_i32_f32" && name != "v_cvt_u32_f32" {
                    a = self.unary(27, a);
                }
                // D3D's conversions saturate and give 0 for NaN, as the
                // hardware does (checked on the RX 6700 XT and WARP).
                let op = if name == "v_cvt_u32_f32" {
                    Cast::FpToUi
                } else {
                    Cast::FpToSi
                };
                let r = self.s.m.cast(c, op, a, self.s.t.i32);
                int(self, r)
            }
            "v_cvt_f32_i32" | "v_cvt_f32_u32" | "v_cvt_f32_ubyte0" | "v_cvt_f32_ubyte1"
            | "v_cvt_f32_ubyte2" | "v_cvt_f32_ubyte3" => {
                let mut a = s(self, 0)?;
                if let Some(n) = name.strip_prefix("v_cvt_f32_ubyte") {
                    let sh = self.s.m.i32(8 * u32::from(n.as_bytes()[0] - b'0'));
                    let ff = self.s.m.i32(0xff);
                    a = self.s.m.bin(c, Bin::LShr, a, sh);
                    a = self.s.m.bin(c, Bin::And, a, ff);
                }
                let op = if name == "v_cvt_f32_i32" {
                    Cast::SiToFp
                } else {
                    Cast::UiToFp
                };
                let r = self.s.m.cast(c, op, a, self.s.t.f32);
                float(self, r)
            }
            "v_cvt_f32_f64" => {
                let (lo, hi) = self.src64_pair(inst, 0)?;
                self.s.flags |= flags::DOUBLE;
                let f64t = self.s.m.double();
                let d = self
                    .s
                    .op(c, "makeDouble.f64", Attr::ReadNone, f64t, 101, &[lo, hi]);
                let r = self.s.m.cast(c, Cast::FpTrunc, d, self.s.t.f32);
                float(self, r)
            }
            "v_fma_mix_f32" | "v_fma_mixlo_f16" | "v_fma_mixhi_f16" => {
                // Each source is f32, or with op_sel_hi the low (op_sel: high)
                // f16 half; neg_hi is the absolute value here.
                let mut xs = Vec::new();
                for i in 0..3 {
                    let op = v.src.get(i).ok_or("a missing operand")?;
                    let x = self.operand(inst, op)?;
                    let mut x = if v.opsel_hi >> i & 1 != 0 {
                        let h = if v.opsel >> i & 1 != 0 {
                            let k16 = self.s.m.i32(16);
                            self.s.m.bin(c, Bin::LShr, x, k16)
                        } else {
                            x
                        };
                        self.s
                            .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[h])
                    } else {
                        self.as_float(x)
                    };
                    if v.neg_hi >> i & 1 != 0 {
                        x = self.unary(6, x);
                    }
                    if v.neg >> i & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        x = self.s.m.bin(c, Bin::Sub, z, x);
                    }
                    xs.push(x);
                }
                let mut r = self.fmad(xs[0], xs[1], xs[2]);
                if name == "v_fma_mix_f32" {
                    return float(self, r);
                }
                if v.clamp {
                    r = self.unary(7, r);
                }
                // The f16 result goes to one half; the other half stays.
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a mixed-precision result");
                };
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => self.i32_of(o)?,
                    None => self.s.m.i32(0),
                };
                let h = self
                    .s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[r]);
                let (keep, h) = if name == "v_fma_mixhi_f16" {
                    let k16 = self.s.m.i32(16);
                    (0xFFFF, self.s.m.bin(c, Bin::Shl, h, k16))
                } else {
                    (0xFFFF_0000, h)
                };
                let keep = self.s.m.i32(keep);
                let old = self.s.m.bin(c, Bin::And, old, keep);
                let x = self.s.m.bin(c, Bin::Or, old, h);
                self.write(inst, Reg::V(reg), x, guard);
                Ok(())
            }
            "v_cvt_pkrtz_f16_f32" => {
                let (a, bb) = (f(self, 0)?, f(self, 1)?);
                let lo = self.f16_toward_zero(a);
                let hi = self.f16_toward_zero(bb);
                let sh = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, hi, sh);
                let r = self.s.m.bin(c, Bin::Or, lo, hi);
                int(self, r)
            }
            "v_cvt_f16_f32" if v.opsel == 0 => {
                // Rounded to nearest even (the MODE default) into the low half.
                let a = f(self, 0)?;
                let r = self
                    .s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, self.s.t.i32, 130, &[a]);
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a 16-bit result outside a VGPR");
                };
                let r = self.low_half(inst, reg, r);
                int(self, r)
            }
            "v_pack_b32_f16" => {
                // Two f16 sources (op_sel: their high halves; abs and neg on
                // the sign bit) into the low and high half.
                let mut halves = Vec::new();
                for k in 0..2 {
                    let op = v.src.get(k).ok_or("a missing operand")?;
                    let mut x = self.operand(inst, op)?;
                    if v.opsel >> k & 1 != 0 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let mask = if v.abs >> k & 1 != 0 { 0x7FFF } else { 0xFFFF };
                    let mask = self.s.m.i32(mask);
                    x = self.s.m.bin(c, Bin::And, x, mask);
                    if v.neg >> k & 1 != 0 {
                        let sign = self.s.m.i32(0x8000);
                        x = self.s.m.bin(c, Bin::Xor, x, sign);
                    }
                    halves.push(x);
                }
                let k16 = self.s.m.i32(16);
                let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
                let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
                int(self, r)
            }
            "v_cvt_f32_f16" if v.opsel == 0 => {
                let a = s(self, 0)?;
                let r = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, self.s.t.f32, 131, &[a]);
                float(self, r)
            }
            "v_cndmask_b32" => {
                let cond = match v.src.get(2) {
                    Some(&Operand::Scalar { code, .. }) if code == NULL => {
                        return if v.abs != 0 || v.neg != 0 {
                            let a = f(self, 0)?;
                            float(self, a)
                        } else {
                            let a = s(self, 0)?;
                            int(self, a)
                        };
                    }
                    Some(&Operand::Const(128)) | Some(&Operand::Literal(0)) => {
                        return if v.abs != 0 || v.neg != 0 {
                            let a = f(self, 0)?;
                            float(self, a)
                        } else {
                            let a = s(self, 0)?;
                            int(self, a)
                        };
                    }
                    Some(&Operand::Const(193)) => {
                        return if v.abs != 0 || v.neg != 0 {
                            let bb = f(self, 1)?;
                            float(self, bb)
                        } else {
                            let bb = s(self, 1)?;
                            int(self, bb)
                        };
                    }
                    other => self.mask_cond(inst, other, c, "a select on an untracked mask")?,
                };
                if v.abs != 0 || v.neg != 0 {
                    let (a, bb) = (f(self, 0)?, f(self, 1)?);
                    let r = self.s.m.select(c, cond, bb, a);
                    return float(self, r);
                }
                let (a, bb) = (s(self, 0)?, s(self, 1)?);
                let r = self.s.m.select(c, cond, bb, a);
                int(self, r)
            }
            "v_add_co_u32" | "v_sub_co_u32" | "v_subrev_co_u32" | "v_add_co_ci_u32"
            | "v_sub_co_ci_u32" | "v_subrev_co_ci_u32" => {
                // The carry (borrow) out is a lane mask: this lane's bit.
                let (mut a, mut bb) = (s(self, 0)?, s(self, 1)?);
                if name.contains("subrev") {
                    std::mem::swap(&mut a, &mut bb);
                }
                let cin = if name.ends_with("_ci_u32") {
                    Some(self.mask_cond(
                        inst,
                        v.src.get(2),
                        c,
                        "a carry-in from an untracked mask",
                    )?)
                } else {
                    None
                };
                let (r, carry) = self.add_sub_carry(c, name.starts_with("v_add"), a, bb, cin);
                int(self, r)?;
                if let Some(Operand::Scalar { code, .. }) = v.carry
                    && let Some(w) = inst.write(Reg::S(code))
                {
                    self.lane_bits.insert(w, carry);
                }
                for &(_, w) in &inst.writes {
                    self.vals
                        .entry(w)
                        .or_insert(Val::Missing("a lane mask read as data"));
                }
                Ok(())
            }
            "v_readlane_b32" => {
                let dst_v = match v.dst {
                    Some(Operand::Scalar { code, .. }) if code != NULL => inst.write(Reg::S(code)),
                    _ => None,
                };
                if self.is_mask(dst_v) || dst_v.is_some_and(|d| self.x.st.highs.contains_key(&d)) {
                    return Err(LANE_MASK);
                }
                if let [Operand::Vector { reg, .. }, lane_op] = v.src[..]
                    && let Some(lane_idx) = crate::resource::lane_of(lane_op)
                    && let Some(src_v) = inst.read(Reg::V(reg))
                    && let Some(spilled) = self.spilled_scalar(src_v, lane_idx)
                {
                    match spilled {
                        crate::resource::Moved::Const(x) => {
                            self.cross_lane(false)?;
                            let v = self.s.m.i32(x);
                            return int(self, v);
                        }
                        crate::resource::Moved::Value(sv) => {
                            if let Some(&Val::I32(x)) = self.vals.get(&sv) {
                                self.cross_lane(false)?;
                                return int(self, x);
                            }
                        }
                    }
                }
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                let mask = if self.x.native.wave64 { 63 } else { 31 };
                let lane = match lane {
                    V::Const(k) => self.s.m.i32(k & mask),
                    _ => {
                        let m = self.s.m.i32(mask);
                        self.s.m.bin(c, Bin::And, lane, m)
                    }
                };
                // A computed lane in a pixel or vertex program is a host lane
                // (from a mask or `v_mbcnt`); a constant one is the console's.
                if self.mask_data == MaskData::Host && !matches!(lane, V::Const(_)) {
                    self.s.flags |= flags::WAVE_OPS;
                } else {
                    self.cross_lane(false)?;
                }
                let r = self.read_lane(a, lane);
                int(self, r)
            }
            "v_permlanex16_b32" => {
                // Cross-lane gather across adjacent 16-lane rows:
                // S0 is the source vector value, S1/S2 provide nibble selectors.
                // For each 16-lane row, lane i selects nibble i of {S2, S1}:
                // lanes 0..7 use S1 nibbles 0..7; lanes 8..15 use S2 nibbles 0..7.
                let (src, s1, s2) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                self.cross_lane(false)?;
                let me = self.lane_index();
                // alt_row = (lane & !31) | ((lane & 16) ^ 16)
                let not31 = self.s.m.i32(!31);
                let row_base = self.s.m.bin(c, Bin::And, me, not31);
                let k16 = self.s.m.i32(16);
                let row_bit = self.s.m.bin(c, Bin::And, me, k16);
                let alt_bit = self.s.m.bin(c, Bin::Xor, row_bit, k16);
                let alt_row = self.s.m.bin(c, Bin::Or, row_base, alt_bit);

                // Nibble selector: lane_in_row = lane & 15.
                // Lanes 0..7 index S1; lanes 8..15 index S2.
                let k15 = self.s.m.i32(15);
                let lane_in_row = self.s.m.bin(c, Bin::And, me, k15);
                let k8 = self.s.m.i32(8);
                let is_s2 = self.s.m.cmp(c, pred::IUGE, lane_in_row, k8);
                let sel_word = self.s.m.select(c, is_s2, s2, s1);

                // Shift: (lane_in_row & 7) * 4
                let k7 = self.s.m.i32(7);
                let nibble_idx = self.s.m.bin(c, Bin::And, lane_in_row, k7);
                let four = self.s.m.i32(4);
                let sh = self.s.m.bin(c, Bin::Shl, nibble_idx, four);
                let shifted = self.s.m.bin(c, Bin::LShr, sel_word, sh);
                let pick = self.s.m.bin(c, Bin::And, shifted, k15);

                let target_lane = self.s.m.bin(c, Bin::Or, alt_row, pick);
                let r = self.read_lane(src, target_lane);
                int(self, r)
            }
            "v_writelane_b32" => {
                // The lane given takes the scalar; the others keep their value.
                // RDNA2 ignores EXEC for v_writelane_b32 so compilers can spill
                // SGPRs into inactive lanes.
                let (a, lane) = (s(self, 0)?, s(self, 1)?);
                let mask = if self.x.native.wave64 { 63 } else { 31 };
                let lane = match lane {
                    V::Const(k) => self.s.m.i32(k & mask),
                    _ => {
                        let m = self.s.m.i32(mask);
                        self.s.m.bin(c, Bin::And, lane, m)
                    }
                };
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a lane write");
                };
                let old = match inst.read(Reg::V(reg)) {
                    Some(o) => self.i32_of(o)?,
                    None => self.s.m.i32(0),
                };
                self.cross_lane(false)?;
                let me = self.lane_index();
                let is = self.s.m.cmp(c, pred::EQ, me, lane);
                let r = self.s.m.select(c, is, a, old);
                self.write(inst, Reg::V(reg), r, None);
                Ok(())
            }
            "v_readfirstlane_b32"
                if inst
                    .writes
                    .first()
                    .is_some_and(|w| self.x.res.waterfall.reads.contains(&w.1)) =>
            {
                // A waterfall loop's read of a descriptor dword: the lane's own.
                let a = s(self, 0)?;
                int(self, a)
            }
            "v_readfirstlane_b32" => {
                let a = s(self, 0)?;
                self.s.flags |= flags::WAVE_OPS;
                let r = self.s.op(
                    c,
                    "waveReadLaneFirst.i32",
                    Attr::NoUnwind,
                    self.s.t.i32,
                    118,
                    &[a],
                );
                let r = self.helper_reads_own(a, r);
                int(self, r)
            }
            "v_sad_u8" | "v_msad_u8" | "v_sad_u16" | "v_sad_u32" => {
                if v.omod != 0 || v.abs != 0 || v.neg != 0 {
                    return Err("integer output modifiers");
                }
                let (a, b, acc) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let delta = match name {
                    "v_sad_u8" | "v_msad_u8" => self.sad_lanes(a, b, 8, 4, name == "v_msad_u8"),
                    "v_sad_u16" => self.sad_lanes(a, b, 16, 2, false),
                    _ => self.abs_diff_u32(a, b),
                };
                let mut r = self.s.m.bin(c, Bin::Add, acc, delta);
                if v.clamp {
                    let over = self.s.m.cmp(c, pred::IULT, r, acc);
                    let max = self.s.m.i32(u32::MAX);
                    r = self.s.m.select(c, over, max, r);
                }
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("a result without a register");
                };
                self.write(inst, Reg::V(reg), r, guard);
                Ok(())
            }
            _ => {
                let r = self.integer(inst, name)?;
                int(self, r)
            }
        }
    }

    /// Traces lane `lane` of VGPR `v` back through `v_writelane_b32` spills.
    fn spilled_scalar(&self, v: Value, lane: u32) -> Option<crate::resource::Moved> {
        let p = self.x.p;
        crate::resource::spilled_lane(
            p,
            &|val| {
                let &b = self.def_block.get(&val)?;
                p.blocks[b]
                    .insts
                    .iter()
                    .find(|i| i.writes.iter().any(|w| w.1 == val))
            },
            &|val| self.phi_at.get(&val).copied(),
            v,
            lane,
            &mut Vec::new(),
        )
        .ok()
        .flatten()
    }

    /// The low and high 32-bit halves of a 64-bit source (without `INT64`).
    fn src64_pair(&mut self, inst: &Inst, k: usize) -> Result<(V, V), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 || v.sdwa.is_some() || v.dpp.is_some() {
            return Err("64-bit operand modifiers");
        }
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let pair = |l: &mut Self, r: Reg, r1: Reg| -> Result<(V, V), Why> {
            let lo = l.i32_of(inst.read(r).ok_or("an unread operand")?)?;
            let hi = l.i32_of(inst.read(r1).ok_or("an unread operand")?)?;
            Ok((lo, hi))
        };
        match op {
            Operand::Vector { reg, .. } => pair(self, Reg::V(reg), Reg::V(reg + 1)),
            Operand::Scalar { code, .. } if code != NULL => {
                pair(self, Reg::S(code), Reg::S(code + 1))
            }
            Operand::Const(k @ 128..=208) => {
                let x = crate::resource::inline(k).ok_or("a special operand")?;
                let lo = self.s.m.i32(x);
                let hi = self.s.m.i32(((x as i32) >> 31) as u32);
                Ok((lo, hi))
            }
            Operand::Const(code @ 240..=248) => {
                let bits = if code == 248 {
                    (std::f64::consts::FRAC_1_PI / 2.0).to_bits()
                } else {
                    let single = crate::resource::inline(code).ok_or("a special operand")?;
                    f64::from(f32::from_bits(single)).to_bits()
                };
                Ok((self.s.m.i32(bits as u32), self.s.m.i32((bits >> 32) as u32)))
            }
            // A literal supplies the upper word of an f64, with a zero low word.
            Operand::Literal(hi) => Ok((self.s.m.i32(0), self.s.m.i32(hi))),
            _ => {
                let x = self.operand(inst, &op)?;
                Ok((x, self.s.m.i32(0)))
            }
        }
    }

    /// `|a - b|` for unsigned 32-bit values.
    fn abs_diff_u32(&mut self, a: V, b: V) -> V {
        let c = self.cur;
        let ab = self.s.m.bin(c, Bin::Sub, a, b);
        let ba = self.s.m.bin(c, Bin::Sub, b, a);
        let ge = self.s.m.cmp(c, pred::IUGE, a, b);
        self.s.m.select(c, ge, ab, ba)
    }

    /// Sum of unsigned absolute differences across `lanes` packed elements of
    /// `bits` width (`8` or `16`). With `masked`, lanes where `a`'s element is
    /// zero contribute 0 (`v_msad_u8`).
    fn sad_lanes(&mut self, a: V, b: V, bits: u32, lanes: u32, masked: bool) -> V {
        let c = self.cur;
        let mask = self.s.m.i32((1u32 << bits) - 1);
        let zero = self.s.m.i32(0);
        let mut total = zero;
        for lane in 0..lanes {
            let (xa, xb) = if lane == 0 {
                (a, b)
            } else {
                let sh = self.s.m.i32(lane * bits);
                (
                    self.s.m.bin(c, Bin::LShr, a, sh),
                    self.s.m.bin(c, Bin::LShr, b, sh),
                )
            };
            let xa = self.s.m.bin(c, Bin::And, xa, mask);
            let xb = self.s.m.bin(c, Bin::And, xb, mask);
            let mut diff = self.abs_diff_u32(xa, xb);
            if masked {
                let is_zero = self.s.m.cmp(c, pred::EQ, xa, zero);
                diff = self.s.m.select(c, is_zero, zero, diff);
            }
            total = if lane == 0 {
                diff
            } else {
                self.s.m.bin(c, Bin::Add, total, diff)
            };
        }
        total
    }

    /// A 64-bit source (a register pair; inline constants sign-extend).
    fn src64(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.abs >> k & 1 != 0 || v.neg >> k & 1 != 0 || v.sdwa.is_some() || v.dpp.is_some() {
            return Err("64-bit operand modifiers");
        }
        let c = self.cur;
        let i64t = self.s.m.int(64);
        self.s.flags |= flags::INT64;
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let pair = |l: &mut Self, r: Reg, r1: Reg| -> Result<V, Why> {
            let lo = l.i32_of(inst.read(r).ok_or("an unread operand")?)?;
            let hi = l.i32_of(inst.read(r1).ok_or("an unread operand")?)?;
            Ok(l.pack64(lo, hi))
        };
        match op {
            Operand::Vector { reg, .. } => pair(self, Reg::V(reg), Reg::V(reg + 1)),
            Operand::Scalar { code, .. } if code != NULL => {
                pair(self, Reg::S(code), Reg::S(code + 1))
            }
            Operand::Const(k @ 128..=208) => {
                let x = crate::resource::inline(k).ok_or("a special operand")?;
                Ok(self.s.m.int_const(i64t, i64::from(x as i32)))
            }
            _ => {
                // Literals and the null register: zero-extended.
                let x = self.operand(inst, &op)?;
                Ok(self.s.m.cast(c, Cast::ZExt, x, i64t))
            }
        }
    }

    /// Packs low and high 32-bit dwords into an i64.
    fn pack64(&mut self, lo: V, hi: V) -> V {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        self.s.flags |= flags::INT64;
        let lo = self.s.m.cast(c, Cast::ZExt, lo, i64t);
        let hi = self.s.m.cast(c, Cast::ZExt, hi, i64t);
        let k32 = self.s.m.int_const(i64t, 32);
        let hi = self.s.m.bin(c, Bin::Shl, hi, k32);
        self.s.m.bin(c, Bin::Or, lo, hi)
    }

    /// The low and high dwords of an i64.
    fn split64(&mut self, x: V) -> (V, V) {
        let c = self.cur;
        let i64t = self.s.m.int(64);
        let k32 = self.s.m.int_const(i64t, 32);
        let lo = self.s.m.cast(c, Cast::Trunc, x, self.s.t.i32);
        let hi = self.s.m.bin(c, Bin::LShr, x, k32);
        let hi = self.s.m.cast(c, Cast::Trunc, hi, self.s.t.i32);
        (lo, hi)
    }

    /// Source `k` of a 16-bit f16 operation in the low 16 bits (the rest 0):
    /// op_sel (or an SDWA select) picks the slice; inline float constants
    /// are f16 values here.
    fn half_src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        self.half_src_typed(inst, k, true)
    }

    /// Source `k` of a 16-bit integer operation in the low 16 bits (the rest 0):
    /// op_sel (or an SDWA select, with sign extension) picks the slice; inline
    /// float constants use their f32 bit patterns on GFX10.
    fn half_int_src(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        self.half_src_typed(inst, k, false)
    }

    fn half_src_typed(&mut self, inst: &Inst, k: usize, float: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let x = match self.src0 {
            Some(x) if k == 0 => x,
            _ if float => self.f16_operand(inst, &op)?,
            _ => self.operand(inst, &op)?,
        };
        let c = self.cur;
        let m = self.s.m.i32(0xFFFF);
        if let Some(sd) = v.sdwa {
            let sel = sd.src_sel.get(k).copied().unwrap_or(6);
            return match sel {
                0..=5 => {
                    let sext = !float && (sd.sext >> k & 1 != 0);
                    let x = self.select_bits(x, sel, sext);
                    Ok(self.s.m.bin(c, Bin::And, x, m))
                }
                6 => Ok(self.s.m.bin(c, Bin::And, x, m)),
                _ => Err("an SDWA source select on a 16-bit operation"),
            };
        }
        let x = if v.opsel >> k & 1 != 0 {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// An f16 source as an f32 (exact), with its abs/neg modifiers.
    fn half_fsrc(&mut self, inst: &Inst, k: usize) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let h = self.half_src(inst, k)?;
        let f32t = self.s.t.f32;
        let mut f = self
            .s
            .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
        if v.abs >> k & 1 != 0 {
            f = self.unary(6, f);
        }
        if v.neg >> k & 1 != 0 {
            let z = self.s.m.f32(0x8000_0000);
            f = self.s.m.bin(self.cur, Bin::Sub, z, f);
        }
        Ok(f)
    }

    /// Writes a 16-bit result (low bits of `x`) into the destination's low
    /// half, or its high half with op_sel bit 3 (the other half is kept on
    /// GFX10), or places it according to SDWA `dst_sel`/`dst_unused`.
    fn write_half(&mut self, inst: &Inst, x: V, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a 16-bit result outside a VGPR");
        };
        if let Some(sd) = v.sdwa {
            let r = match (sd.dst_sel, sd.dst_unused) {
                (0..=5, 0..=2) => self.place_bits(inst, Reg::V(reg), x, sd.dst_sel, sd.dst_unused),
                (6, _) => {
                    let low = self.s.m.i32(0xFFFF);
                    self.s.m.bin(self.cur, Bin::And, x, low)
                }
                _ => return Err("an SDWA destination of a 16-bit operation"),
            };
            self.write_placed(inst, Reg::V(reg), r, guard);
            return Ok(());
        }
        let high = v.opsel >> 3 & 1 != 0;
        let c = self.cur;
        let old = self.old_i32_of(inst.read(Reg::V(reg)), false, None);
        let (k16, low) = (self.s.m.i32(16), self.s.m.i32(0xFFFF));
        let x = self.s.m.bin(c, Bin::And, x, low);
        let r = if high {
            let o = self.s.m.bin(c, Bin::And, old, low);
            let x = self.s.m.bin(c, Bin::Shl, x, k16);
            self.s.m.bin(c, Bin::Or, o, x)
        } else {
            let keep = self.s.m.i32(0xFFFF_0000);
            let o = self.s.m.bin(c, Bin::And, old, keep);
            self.s.m.bin(c, Bin::Or, o, x)
        };
        self.write_placed(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    /// 16-bit operations, computed in f32 or i32: f16 sums, differences and
    /// products are exact there and round once to f16 (f32 has the 2 x 11 +
    /// 2 bits that make double rounding harmless); `None` if `name` is none.
    fn valu16(&mut self, inst: &Inst, name: &str, guard: Option<V>) -> Option<Result<(), Why>> {
        const FLOAT2: &[&str] = &[
            "v_add_f16",
            "v_sub_f16",
            "v_subrev_f16",
            "v_mul_f16",
            "v_max_f16",
            "v_min_f16",
        ];
        const FLOAT3: &[&str] = &[
            "v_fma_f16",
            "v_mad_f16",
            "v_fmac_f16",
            "v_mac_f16",
            "v_max3_f16",
            "v_min3_f16",
            "v_med3_f16",
            "v_div_fixup_f16",
        ];
        const FLOAT1: &[&str] = &[
            "v_rcp_f16",
            "v_sqrt_f16",
            "v_rsq_f16",
            "v_log_f16",
            "v_exp_f16",
            "v_fract_f16",
            "v_trunc_f16",
            "v_floor_f16",
            "v_ceil_f16",
            "v_rndne_f16",
            "v_sin_f16",
            "v_cos_f16",
            "v_frexp_mant_f16",
            "v_cvt_f16_u16",
            "v_cvt_f16_i16",
        ];
        const INT: &[&str] = &[
            "v_add_nc_u16",
            "v_sub_nc_u16",
            "v_subrev_nc_u16",
            "v_mul_lo_u16",
            "v_lshlrev_b16",
            "v_lshrrev_b16",
            "v_ashrrev_i16",
            "v_max_u16",
            "v_max_i16",
            "v_min_u16",
            "v_min_i16",
            "v_min3_i16",
            "v_min3_u16",
            "v_max3_i16",
            "v_max3_u16",
            "v_med3_i16",
            "v_med3_u16",
            "v_cvt_u16_f16",
            "v_cvt_i16_f16",
            "v_cvt_norm_i16_f16",
            "v_cvt_norm_u16_f16",
            "v_frexp_exp_i16_f16",
        ];
        let all = [FLOAT1, FLOAT2, FLOAT3, INT];
        if !all.iter().any(|l| l.contains(&name))
            && !matches!(name, "v_cvt_f16_f32" | "v_cvt_f32_f16" | "v_ldexp_f16")
        {
            return None;
        }
        Some(self.valu16_op(inst, name, guard, FLOAT1, FLOAT2, FLOAT3))
    }

    #[allow(clippy::too_many_arguments)]
    fn valu16_op(
        &mut self,
        inst: &Inst,
        name: &str,
        guard: Option<V>,
        float1: &[&str],
        float2: &[&str],
        float3: &[&str],
    ) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let i32t = self.s.t.i32;
        let f32t = self.s.t.f32;
        if name == "v_cvt_f32_f16" {
            let f = self.half_fsrc(inst, 0)?;
            return self.write_f(inst, f, guard);
        }
        let float_result = |l: &mut Self, mut f: V| -> Result<(), Why> {
            if v.omod != 0 {
                return Err("an output modifier on a 16-bit operation");
            }
            if v.clamp {
                f = l.unary(7, f);
            }
            let h =
                l.s.op(l.cur, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
            l.write_half(inst, h, guard)
        };
        if name == "v_cvt_f16_f32" {
            let f = self.fsrc(inst, 0)?;
            return float_result(self, f);
        }
        if name == "v_ldexp_f16" {
            // D.f16 = S0.f16 * 2^(S1.i16). S1 is a signed 16-bit integer exponent,
            // so float abs/neg modifiers apply only to S0.
            if (v.abs | v.neg) & 2 != 0 {
                return Err("float modifier on v_ldexp_f16 exponent");
            }
            let a = self.half_fsrc(inst, 0)?;
            let raw_e = self.half_src(inst, 1)?;
            let k16 = self.s.m.i32(16);
            let e = self.s.m.bin(c, Bin::Shl, raw_e, k16);
            let e = self.s.m.bin(c, Bin::AShr, e, k16);
            // Clamp exponent to [-64, 64] so 2^e is always a normal finite f32:
            // - +-0 * finite = +-0 (never 0 * inf = NaN),
            // - +-inf * finite = +-inf (never inf * 0 = NaN),
            // - min f16 (2^-24) * 2^64 overflows f16 to +-inf,
            // - max f16 (~2^16) * 2^-64 underflows f16 to +-0.
            let lo = self.s.m.i32((-64i32) as u32);
            let hi = self.s.m.i32(64);
            let under = self.s.m.cmp(c, pred::SLT, e, lo);
            let e = self.s.m.select(c, under, lo, e);
            let over = self.s.m.cmp(c, pred::SGT, e, hi);
            let e = self.s.m.select(c, over, hi, e);
            let bias = self.s.m.i32(127);
            let k23 = self.s.m.i32(23);
            let biased = self.s.m.bin(c, Bin::Add, e, bias);
            let scale_bits = self.s.m.bin(c, Bin::Shl, biased, k23);
            let scale = self.as_float(scale_bits);
            let r = self.s.m.bin(c, Bin::Mul, a, scale);
            // DXIL `legacyF32ToF16` rounds toward zero on NVIDIA/WARP and clamps finite
            // values >= 65536 to 0x7BFF (max finite f16). Under RDNA2's default RTE mode,
            // |r| >= 65520.0 (the f16 overflow threshold) overflows to signed infinity.
            let abs_r = self.unary(6, r);
            let ov_thresh = self.s.m.f32(65520.0f32.to_bits());
            let overflows = self.s.m.cmp(c, pred::OGE, abs_r, ov_thresh);
            let r_bits = self.as_int(r);
            let sign_mask = self.s.m.i32(0x8000_0000);
            let inf_bits = self.s.m.i32(0x7F80_0000);
            let r_sign = self.s.m.bin(c, Bin::And, r_bits, sign_mask);
            let signed_inf_bits = self.s.m.bin(c, Bin::Or, r_sign, inf_bits);
            let signed_inf = self.as_float(signed_inf_bits);
            let r = self.s.m.select(c, overflows, signed_inf, r);
            return float_result(self, r);
        }
        if float2.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let r = match name {
                "v_add_f16" => self.s.m.bin(c, Bin::Add, a, b),
                "v_sub_f16" => self.s.m.bin(c, Bin::Sub, a, b),
                "v_subrev_f16" => self.s.m.bin(c, Bin::Sub, b, a),
                "v_mul_f16" => self.s.m.bin(c, Bin::Mul, a, b),
                "v_max_f16" => self.max_min(true, a, b),
                _ => self.max_min(false, a, b),
            };
            return float_result(self, r);
        }
        if float3.contains(&name) {
            let (a, b) = (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?);
            let acc = if matches!(name, "v_fmac_f16" | "v_mac_f16") {
                // The accumulator is the destination's half that is written.
                let Some(Operand::Vector { reg, .. }) = v.dst else {
                    return Err("an accumulator");
                };
                let x = self.i32_of(inst.read(Reg::V(reg)).ok_or("an accumulator")?)?;
                let x = if v.opsel >> 3 & 1 != 0 {
                    let k16 = self.s.m.i32(16);
                    self.s.m.bin(c, Bin::LShr, x, k16)
                } else {
                    x
                };
                let m = self.s.m.i32(0xFFFF);
                let x = self.s.m.bin(c, Bin::And, x, m);
                self.s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
            } else {
                self.half_fsrc(inst, 2)?
            };
            let r = match name {
                // shortcut: fused f16 multiply-adds as FMad in f32 (the
                // product is exact; the sum rounds twice).
                "v_fma_f16" | "v_fmac_f16" | "v_mad_f16" | "v_mac_f16" => self.fmad(a, b, acc),
                "v_max3_f16" => {
                    let t = self.max_min(true, a, b);
                    self.max_min(true, t, acc)
                }
                "v_min3_f16" => {
                    let t = self.max_min(false, a, b);
                    self.max_min(false, t, acc)
                }
                "v_div_fixup_f16" => self.div_fixup(a, b, acc),
                _ => self.med3(a, b, acc),
            };
            return float_result(self, r);
        }
        if float1.contains(&name) {
            let r = match name {
                "v_cvt_f16_u16" | "v_cvt_f16_i16" => {
                    let mut x = self.half_int_src(inst, 0)?;
                    let op = if name == "v_cvt_f16_i16" {
                        let k16 = self.s.m.i32(16);
                        let t = self.s.m.bin(c, Bin::Shl, x, k16);
                        x = self.s.m.bin(c, Bin::AShr, t, k16);
                        Cast::SiToFp
                    } else {
                        Cast::UiToFp
                    };
                    self.s.m.cast(c, op, x, f32t)
                }
                "v_rcp_f16" => {
                    let a = self.half_fsrc(inst, 0)?;
                    let one = self.s.m.f32(1.0f32.to_bits());
                    self.s.m.bin(c, Bin::SDiv, one, a)
                }
                "v_frexp_mant_f16" => {
                    let a = self.half_fsrc(inst, 0)?;
                    self.frexp_f32(a).0
                }
                _ => {
                    let a = self.half_fsrc(inst, 0)?;
                    let op = match name {
                        "v_sqrt_f16" => 24,
                        "v_rsq_f16" => 25,
                        "v_log_f16" => 23,
                        "v_exp_f16" => 21,
                        "v_fract_f16" => 22,
                        "v_trunc_f16" => 29,
                        "v_floor_f16" => 27,
                        "v_ceil_f16" => 28,
                        "v_rndne_f16" => 26,
                        "v_sin_f16" => 13,
                        _ => 12,
                    };
                    match op {
                        12 | 13 => self.trig_turns(a, op == 12),
                        // Any f16 in (-2^-11, 0) has fract >= 1 - 2^-12; clamp to 0x3bff (1 - 2^-11).
                        22 => self.fract(a, 1.0 - 1.0 / 4096.0, 1.0 - 1.0 / 2048.0),
                        _ => self.unary(op, a),
                    }
                }
            };
            return float_result(self, r);
        }
        if matches!(name, "v_cvt_norm_i16_f16" | "v_cvt_norm_u16_f16") {
            if v.clamp || v.omod != 0 {
                return Err("modifiers on a 16-bit integer operation");
            }
            let r = self.cvt_norm16(inst, name == "v_cvt_norm_i16_f16")?;
            return self.write_half(inst, r, guard);
        }
        if name == "v_frexp_exp_i16_f16" {
            if v.clamp || v.omod != 0 {
                return Err("modifiers on a 16-bit integer operation");
            }
            let a = self.half_fsrc(inst, 0)?;
            let (_, exp) = self.frexp_f32(a);
            return self.write_half(inst, exp, guard);
        }
        // 16-bit integers: computed on the low halves, the low 16 bits kept.
        if v.clamp || v.omod != 0 || v.abs != 0 || v.neg != 0 {
            return Err("modifiers on a 16-bit integer operation");
        }
        let r = match name {
            "v_cvt_u16_f16" | "v_cvt_i16_f16" => {
                // Toward zero, saturating at the 16-bit range (infinities
                // included); NaN gives 0. Clamped in f32 first, where every
                // f16 is exact, so the conversion never leaves i32's range.
                let a = self.half_fsrc(inst, 0)?;
                let (lo, hi) = if name == "v_cvt_u16_f16" {
                    (0.0f32, 65535.0f32)
                } else {
                    (-32768.0, 32767.0)
                };
                let (lo, hi) = (self.s.m.f32(lo.to_bits()), self.s.m.f32(hi.to_bits()));
                let t = self.binary(35, a, lo);
                let t = self.binary(36, t, hi);
                let r = self.s.m.cast(c, Cast::FpToSi, t, i32t);
                let nan = self.s.m.cmp(c, pred::UNO, a, a);
                let z = self.s.m.i32(0);
                self.s.m.select(c, nan, z, r)
            }
            _ => {
                let (mut a, mut b) = (self.half_int_src(inst, 0)?, self.half_int_src(inst, 1)?);
                let k16 = self.s.m.i32(16);
                let sext = |l: &mut Self, x: V| {
                    let t = l.s.m.bin(c, Bin::Shl, x, k16);
                    l.s.m.bin(c, Bin::AShr, t, k16)
                };
                if name.ends_with("_i16") {
                    a = sext(self, a);
                    b = sext(self, b);
                }
                let k15 = self.s.m.i32(15);
                match name {
                    "v_add_nc_u16" => self.s.m.bin(c, Bin::Add, a, b),
                    "v_sub_nc_u16" => self.s.m.bin(c, Bin::Sub, a, b),
                    "v_subrev_nc_u16" => self.s.m.bin(c, Bin::Sub, b, a),
                    "v_mul_lo_u16" => self.s.m.bin(c, Bin::Mul, a, b),
                    "v_lshlrev_b16" | "v_lshrrev_b16" | "v_ashrrev_i16" => {
                        // src0 is the shift (4 bits), src1 the value.
                        let sh = self.s.m.bin(c, Bin::And, a, k15);
                        let x = if name == "v_ashrrev_i16" {
                            sext(self, b)
                        } else {
                            b
                        };
                        let op = match name {
                            "v_lshlrev_b16" => Bin::Shl,
                            "v_lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        self.s.m.bin(c, op, x, sh)
                    }
                    "v_min3_i16" | "v_min3_u16" | "v_max3_i16" | "v_max3_u16" | "v_med3_i16"
                    | "v_med3_u16" => {
                        let mut d = self.half_int_src(inst, 2)?;
                        if name.ends_with("_i16") {
                            d = sext(self, d);
                        }
                        let (gt, lt) = if name.ends_with("_i16") {
                            (pred::SGT, pred::SLT)
                        } else {
                            (pred::IUGT, pred::IULT)
                        };
                        let pick = |l: &mut Self, p: u8, x: V, y: V| {
                            let t = l.s.m.cmp(c, p, x, y);
                            l.s.m.select(c, t, x, y)
                        };
                        match name {
                            "v_min3_i16" | "v_min3_u16" => {
                                let t = pick(self, lt, a, b);
                                pick(self, lt, t, d)
                            }
                            "v_max3_i16" | "v_max3_u16" => {
                                let t = pick(self, gt, a, b);
                                pick(self, gt, t, d)
                            }
                            _ => {
                                let hi = pick(self, gt, a, b);
                                let lo = pick(self, lt, a, b);
                                let t = pick(self, lt, hi, d);
                                pick(self, gt, t, lo)
                            }
                        }
                    }
                    _ => {
                        let p = match name {
                            "v_max_u16" => pred::IUGT,
                            "v_min_u16" => pred::IULT,
                            "v_max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, a, b);
                        self.s.m.select(c, t, a, b)
                    }
                }
            }
        };
        self.write_half(inst, r, guard)
    }

    /// `v_cvt_norm_i16_f16` / `v_cvt_norm_u16_f16`: the f16 source (with its
    /// abs/neg modifiers) clamped to [-1, 1] or [0, 1], times 32767 or 65535,
    /// rounded to nearest even once from the exact product; NaN gives 0.
    /// Worked on the f16's bits in integers, because in f32 the product
    /// (up to 26 significant bits) would round before the rounding to an
    /// integer.
    fn cvt_norm16(&mut self, inst: &Inst, signed: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let mut h = self.half_src(inst, 0)?;
        if v.abs & 1 != 0 {
            let m = k(self, 0x7FFF);
            h = self.s.m.bin(c, Bin::And, h, m);
        }
        if v.neg & 1 != 0 {
            let m = k(self, 0x8000);
            h = self.s.m.bin(c, Bin::Xor, h, m);
        }
        let scale = if signed { 0x7FFF } else { 0xFFFF };
        let (k0, k1, k10) = (k(self, 0), k(self, 1), k(self, 10));
        let mag_mask = k(self, 0x7FFF);
        let mag = self.s.m.bin(c, Bin::And, h, mag_mask);
        let k15 = k(self, 15);
        let neg = self.s.m.bin(c, Bin::LShr, h, k15);
        let neg = self.s.m.cmp(c, pred::NE, neg, k0);
        // |x| < 1: x = m * 2^-s with m < 2^11 and 11 <= s <= 24.
        let exp = self.s.m.bin(c, Bin::LShr, mag, k10);
        let man_mask = k(self, 0x3FF);
        let man = self.s.m.bin(c, Bin::And, mag, man_mask);
        let implicit = k(self, 0x400);
        let normal = self.s.m.bin(c, Bin::Or, man, implicit);
        let sub = self.s.m.cmp(c, pred::EQ, exp, k0);
        let m = self.s.m.select(c, sub, man, normal);
        let k25 = k(self, 25);
        let s_norm = self.s.m.bin(c, Bin::Sub, k25, exp);
        let k24 = k(self, 24);
        let sh = self.s.m.select(c, sub, k24, s_norm);
        let ks = k(self, scale);
        let p = self.s.m.bin(c, Bin::Mul, m, ks);
        let q = self.s.m.bin(c, Bin::LShr, p, sh);
        let one_s = self.s.m.bin(c, Bin::Shl, k1, sh);
        let low_mask = self.s.m.bin(c, Bin::Sub, one_s, k1);
        let rem = self.s.m.bin(c, Bin::And, p, low_mask);
        let half = self.s.m.bin(c, Bin::LShr, one_s, k1);
        let above = self.s.m.cmp(c, pred::IUGT, rem, half);
        let tie = self.s.m.cmp(c, pred::EQ, rem, half);
        let odd = self.s.m.bin(c, Bin::And, q, k1);
        let odd = self.s.m.cmp(c, pred::NE, odd, k0);
        let tie_up = self.s.m.bin(c, Bin::And, tie, odd);
        let up = self.s.m.bin(c, Bin::Or, above, tie_up);
        let r = self.s.m.select(c, up, k1, k0);
        let r = self.s.m.bin(c, Bin::Add, q, r);
        // |x| >= 1 (and infinity) saturates; NaN gives 0.
        let one_h = k(self, 0x3C00);
        let big = self.s.m.cmp(c, pred::IUGE, mag, one_h);
        let r = self.s.m.select(c, big, ks, r);
        let inf = k(self, 0x7C00);
        let nan = self.s.m.cmp(c, pred::IUGT, mag, inf);
        let r = self.s.m.select(c, nan, k0, r);
        let r = if signed {
            let n = self.s.m.bin(c, Bin::Sub, k0, r);
            self.s.m.select(c, neg, n, r)
        } else {
            self.s.m.select(c, neg, k0, r)
        };
        let m16 = k(self, 0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, r, m16))
    }

    /// Half `high` of source `k` of a packed (VOP3P) operation, in the low
    /// 16 bits: op_sel picks the half the low result reads, op_sel_hi the
    /// half the high result reads. Packed integer instructions see the f32
    /// bits of an inline float constant.
    fn packed_src(&mut self, inst: &Inst, k: usize, high: bool, float: bool) -> Result<V, Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let op = *v.src.get(k).ok_or("a missing operand")?;
        let sel = if high { v.opsel_hi } else { v.opsel } >> k & 1 != 0;
        let x = if float {
            self.f16_operand(inst, &op)?
        } else {
            self.operand(inst, &op)?
        };
        let c = self.cur;
        let x = if sel {
            let k16 = self.s.m.i32(16);
            self.s.m.bin(c, Bin::LShr, x, k16)
        } else {
            x
        };
        let m = self.s.m.i32(0xFFFF);
        Ok(self.s.m.bin(c, Bin::And, x, m))
    }

    /// `v_pk_fmac_f16` (VOP2 `0x3c` / VOP3 `0x13c`): two packed f16
    /// multiply-accumulates (`d.lo = s0.lo * s1.lo + d.lo`, same for `hi`).
    /// VOP3 `abs`/`neg`/`omod`/`clamp` apply to both halves.
    fn pk_fmac_f16(&mut self, inst: &Inst, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        if v.sdwa.is_some() {
            return Err("SDWA on v_pk_fmac_f16");
        }
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a packed result outside a VGPR");
        };
        let c = self.cur;
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let old = match inst.read(Reg::V(reg)) {
            Some(o) => self.i32_of(o)?,
            None => self.s.m.i32(0),
        };
        let s0_raw = match self.src0 {
            Some(x) => x,
            None => {
                let op = *v.src.first().ok_or("a missing operand")?;
                self.f16_operand(inst, &op)?
            }
        };
        let op1 = *v.src.get(1).ok_or("a missing operand")?;
        let s1_raw = self.f16_operand(inst, &op1)?;

        let k16 = self.s.m.i32(16);
        let m16 = self.s.m.i32(0xFFFF);
        let mut halves = Vec::with_capacity(2);
        for high in [false, true] {
            let mut lane_f = |raw: V, k: usize| {
                let h = if high {
                    self.s.m.bin(c, Bin::LShr, raw, k16)
                } else {
                    raw
                };
                let h = self.s.m.bin(c, Bin::And, h, m16);
                let mut f = self
                    .s
                    .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                if v.abs >> k & 1 != 0 {
                    f = self.unary(6, f);
                }
                if v.neg >> k & 1 != 0 {
                    let z = self.s.m.f32(0x8000_0000);
                    f = self.s.m.bin(c, Bin::Sub, z, f);
                }
                f
            };
            let a = lane_f(s0_raw, 0);
            let b = lane_f(s1_raw, 1);
            let acc_h = if high {
                self.s.m.bin(c, Bin::LShr, old, k16)
            } else {
                old
            };
            let acc_h = self.s.m.bin(c, Bin::And, acc_h, m16);
            let acc = self
                .s
                .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[acc_h]);
            let mut f = self.fmad(a, b, acc);
            if v.omod != 0 {
                let factor = [1.0f32, 2.0, 4.0, 0.5][usize::from(v.omod)];
                let kf = self.s.m.f32(factor.to_bits());
                f = self.s.m.bin(c, Bin::Mul, f, kf);
            }
            if v.clamp {
                f = self.unary(7, f);
            }
            let h = self
                .s
                .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
            halves.push(self.s.m.bin(c, Bin::And, h, m16));
        }
        let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
        let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
        self.write(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    /// Packed 16-bit operations (`v_pk_*`): each half computed as the
    /// 16-bit operation does, f16 through f32 as in [`Self::valu16`].
    fn packed16(&mut self, inst: &Inst, op: &str, guard: Option<V>) -> Result<(), Why> {
        let Fields::Valu(v) = &inst.fields else {
            unreachable!()
        };
        let float = op.ends_with("_f16");
        let n = match op {
            "fma_f16" | "mad_u16" | "mad_i16" => 3,
            "add_f16" | "mul_f16" | "min_f16" | "max_f16" | "add_u16" | "sub_u16" | "add_i16"
            | "sub_i16" | "mul_lo_u16" | "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16"
            | "max_u16" | "min_u16" | "max_i16" | "min_i16" => 2,
            _ => return Err("a packed 16-bit operation"),
        };
        if v.omod != 0 || !float && (v.clamp || v.neg != 0 || v.neg_hi != 0) {
            return Err("modifiers on a packed integer operation");
        }
        let c = self.cur;
        let (i32t, f32t) = (self.s.t.i32, self.s.t.f32);
        let mut halves = Vec::new();
        for high in [false, true] {
            let mut xs = Vec::new();
            for k in 0..n {
                let h = self.packed_src(inst, k, high, float)?;
                if float {
                    let mut f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[h]);
                    let neg = if high { v.neg_hi } else { v.neg };
                    if neg >> k & 1 != 0 {
                        let z = self.s.m.f32(0x8000_0000);
                        f = self.s.m.bin(c, Bin::Sub, z, f);
                    }
                    xs.push(f);
                } else {
                    xs.push(h);
                }
            }
            let r = if float {
                let mut f = match op {
                    "add_f16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "mul_f16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "min_f16" => self.max_min(false, xs[0], xs[1]),
                    "max_f16" => self.max_min(true, xs[0], xs[1]),
                    // shortcut: FMad in f32, as v_fma_f16.
                    _ => self.fmad(xs[0], xs[1], xs[2]),
                };
                if v.clamp {
                    f = self.unary(7, f);
                }
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
            } else {
                let k16 = self.s.m.i32(16);
                let sext = |l: &mut Self, x: V| {
                    let t = l.s.m.bin(c, Bin::Shl, x, k16);
                    l.s.m.bin(c, Bin::AShr, t, k16)
                };
                if op.ends_with("_i16") {
                    for x in &mut xs {
                        *x = sext(self, *x);
                    }
                }
                let k15 = self.s.m.i32(15);
                match op {
                    "add_u16" | "add_i16" => self.s.m.bin(c, Bin::Add, xs[0], xs[1]),
                    "sub_u16" | "sub_i16" => self.s.m.bin(c, Bin::Sub, xs[0], xs[1]),
                    "mul_lo_u16" => self.s.m.bin(c, Bin::Mul, xs[0], xs[1]),
                    "mad_u16" | "mad_i16" => {
                        let m = self.s.m.bin(c, Bin::Mul, xs[0], xs[1]);
                        self.s.m.bin(c, Bin::Add, m, xs[2])
                    }
                    "lshlrev_b16" | "lshrrev_b16" | "ashrrev_i16" => {
                        let sh = self.s.m.bin(c, Bin::And, xs[0], k15);
                        let o = match op {
                            "lshlrev_b16" => Bin::Shl,
                            "lshrrev_b16" => Bin::LShr,
                            _ => Bin::AShr,
                        };
                        let x = if op == "ashrrev_i16" {
                            sext(self, xs[1])
                        } else {
                            xs[1]
                        };
                        self.s.m.bin(c, o, x, sh)
                    }
                    _ => {
                        let p = match op {
                            "max_u16" => pred::IUGT,
                            "min_u16" => pred::IULT,
                            "max_i16" => pred::SGT,
                            _ => pred::SLT,
                        };
                        let t = self.s.m.cmp(c, p, xs[0], xs[1]);
                        self.s.m.select(c, t, xs[0], xs[1])
                    }
                }
            };
            let m = self.s.m.i32(0xFFFF);
            halves.push(self.s.m.bin(c, Bin::And, r, m));
        }
        let k16 = self.s.m.i32(16);
        let hi = self.s.m.bin(c, Bin::Shl, halves[1], k16);
        let r = self.s.m.bin(c, Bin::Or, halves[0], hi);
        let Some(Operand::Vector { reg, .. }) = v.dst else {
            return Err("a packed result outside a VGPR");
        };
        self.write(inst, Reg::V(reg), r, guard);
        Ok(())
    }

    fn integer(&mut self, inst: &Inst, name: &'static str) -> Result<V, Why> {
        let c = self.cur;
        let s = |l: &mut Self, k| l.src(inst, k);
        let k = |l: &mut Self, x: u32| l.s.m.i32(x);
        let m31 = |l: &mut Self, x: V| {
            let k31 = l.s.m.i32(31);
            l.s.m.bin(c, Bin::And, x, k31)
        };
        Ok(match name {
            "v_and_b32" | "v_or_b32" | "v_xor_b32" | "v_add_nc_u32" | "v_add_nc_i32"
            | "v_sub_nc_u32" | "v_subrev_nc_u32" | "v_mul_lo_u32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let (op, a, b) = match name {
                    "v_and_b32" => (Bin::And, a, b),
                    "v_or_b32" => (Bin::Or, a, b),
                    "v_xor_b32" => (Bin::Xor, a, b),
                    "v_add_nc_u32" | "v_add_nc_i32" => (Bin::Add, a, b),
                    "v_sub_nc_u32" => (Bin::Sub, a, b),
                    "v_subrev_nc_u32" => (Bin::Sub, b, a),
                    _ => (Bin::Mul, a, b),
                };
                self.s.m.bin(c, op, a, b)
            }
            "v_xnor_b32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let x = self.s.m.bin(c, Bin::Xor, a, b);
                let ones = k(self, u32::MAX);
                self.s.m.bin(c, Bin::Xor, x, ones)
            }
            "v_not_b32" => {
                let a = s(self, 0)?;
                let ones = k(self, u32::MAX);
                self.s.m.bin(c, Bin::Xor, a, ones)
            }
            "v_lshlrev_b32" | "v_lshrrev_b32" | "v_ashrrev_i32" => {
                let (sh, a) = (s(self, 0)?, s(self, 1)?);
                let sh = m31(self, sh);
                let op = match name {
                    "v_lshlrev_b32" => Bin::Shl,
                    "v_lshrrev_b32" => Bin::LShr,
                    _ => Bin::AShr,
                };
                self.s.m.bin(c, op, a, sh)
            }
            "v_lshl_add_u32" | "v_lshl_or_b32" | "v_add_lshl_u32" | "v_and_or_b32"
            | "v_or3_b32" | "v_xad_u32" | "v_add3_u32" | "v_xor3_b32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                match name {
                    "v_lshl_add_u32" | "v_lshl_or_b32" => {
                        let b = m31(self, b);
                        let t = self.s.m.bin(c, Bin::Shl, a, b);
                        let op = if name == "v_lshl_add_u32" {
                            Bin::Add
                        } else {
                            Bin::Or
                        };
                        self.s.m.bin(c, op, t, d)
                    }
                    "v_add_lshl_u32" => {
                        let t = self.s.m.bin(c, Bin::Add, a, b);
                        let d = m31(self, d);
                        self.s.m.bin(c, Bin::Shl, t, d)
                    }
                    _ => {
                        let (o1, o2) = match name {
                            "v_and_or_b32" => (Bin::And, Bin::Or),
                            "v_or3_b32" => (Bin::Or, Bin::Or),
                            "v_xad_u32" => (Bin::Xor, Bin::Add),
                            "v_add3_u32" => (Bin::Add, Bin::Add),
                            _ => (Bin::Xor, Bin::Xor),
                        };
                        let t = self.s.m.bin(c, o1, a, b);
                        self.s.m.bin(c, o2, t, d)
                    }
                }
            }
            "v_mul_u32_u24" | "v_mul_i32_i24" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let (sh, op) = (
                    k(self, 8),
                    if name == "v_mul_i32_i24" {
                        Bin::AShr
                    } else {
                        Bin::LShr
                    },
                );
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                self.s.m.bin(c, Bin::Mul, a, b)
            }
            "v_bfe_u32" | "v_bfe_i32" => {
                let (a, off, w) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let w = m31(self, w);
                self.bfe(a, off, w, name == "v_bfe_i32")
            }
            "v_bfi_b32" => {
                let (m, a, b) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let ones = k(self, u32::MAX);
                let nm = self.s.m.bin(c, Bin::Xor, m, ones);
                let x = self.s.m.bin(c, Bin::And, m, a);
                let y = self.s.m.bin(c, Bin::And, nm, b);
                self.s.m.bin(c, Bin::Or, x, y)
            }
            "v_max_i32" | "v_min_i32" | "v_max_u32" | "v_min_u32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let p = match name {
                    "v_max_i32" => pred::SGT,
                    "v_min_i32" => pred::SLT,
                    "v_max_u32" => pred::IUGT,
                    _ => pred::IULT,
                };
                let t = self.s.m.cmp(c, p, a, b);
                self.s.m.select(c, t, a, b)
            }
            "v_mul_hi_u32" | "v_mul_hi_i32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                self.mul_hi(a, b, name == "v_mul_hi_i32")
            }
            "v_alignbit_b32" | "v_alignbyte_b32" => {
                // The low dword of {a, b} >> shift.
                let (a, b, sh) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let sh = if name == "v_alignbyte_b32" {
                    let three = k(self, 3);
                    let t = self.s.m.bin(c, Bin::And, sh, three);
                    let k3 = k(self, 3);
                    self.s.m.bin(c, Bin::Shl, t, k3)
                } else {
                    m31(self, sh)
                };
                let k32 = k(self, 32);
                let lo = self.s.m.bin(c, Bin::LShr, b, sh);
                let left = self.s.m.bin(c, Bin::Sub, k32, sh);
                let left = m31(self, left);
                let hi = self.s.m.bin(c, Bin::Shl, a, left);
                let zero = k(self, 0);
                let none = self.s.m.cmp(c, pred::EQ, sh, zero);
                let hi = self.s.m.select(c, none, zero, hi);
                self.s.m.bin(c, Bin::Or, lo, hi)
            }
            "v_bfrev_b32" => {
                let a = s(self, 0)?;
                self.s
                    .op(c, "unary.i32", Attr::ReadNone, self.s.t.i32, 30, &[a])
            }
            "v_bcnt_u32_b32" => {
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let n = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[a]);
                self.s.m.bin(c, Bin::Add, n, b)
            }
            "v_mad_u32_u24" | "v_mad_i32_i24" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let (sh, op) = (
                    k(self, 8),
                    if name == "v_mad_i32_i24" {
                        Bin::AShr
                    } else {
                        Bin::LShr
                    },
                );
                let a = self.s.m.bin(c, Bin::Shl, a, sh);
                let a = self.s.m.bin(c, op, a, sh);
                let b = self.s.m.bin(c, Bin::Shl, b, sh);
                let b = self.s.m.bin(c, op, b, sh);
                let m = self.s.m.bin(c, Bin::Mul, a, b);
                self.s.m.bin(c, Bin::Add, m, d)
            }
            "v_min3_i32" | "v_max3_i32" | "v_min3_u32" | "v_max3_u32" | "v_med3_i32"
            | "v_med3_u32" => {
                let (a, b, d) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                let signed = name.ends_with("i32");
                let (lt, gt) = if signed {
                    (pred::SLT, pred::SGT)
                } else {
                    (pred::IULT, pred::IUGT)
                };
                let pick = |l: &mut Self, p, x, y| {
                    let t = l.s.m.cmp(c, p, x, y);
                    l.s.m.select(c, t, x, y)
                };
                match &name[2..6] {
                    "min3" => {
                        let t = pick(self, lt, a, b);
                        pick(self, lt, t, d)
                    }
                    "max3" => {
                        let t = pick(self, gt, a, b);
                        pick(self, gt, t, d)
                    }
                    _ => {
                        let lo = pick(self, lt, a, b);
                        let hi = pick(self, gt, a, b);
                        let t = pick(self, lt, hi, d);
                        pick(self, gt, lo, t)
                    }
                }
            }
            "v_perm_b32" => {
                // Byte i of the result is byte S of {src0, src1} (src1 the
                // low dword) for selector byte S below 8; 8-11 repeat the
                // sign of bytes 1 and 3 of src1, then of src0; 12 is 0x00,
                // higher 0xFF.
                let Fields::Valu(v) = &inst.fields else {
                    unreachable!()
                };
                let sel = match v.src.get(2) {
                    Some(&Operand::Literal(x)) => x,
                    Some(&Operand::Const(k)) => crate::resource::inline(k).ok_or(name)?,
                    _ => {
                        // A computed selector: each byte picked at run time.
                        let (a, b, sv) = (s(self, 0)?, s(self, 1)?, s(self, 2)?);
                        let mut r = k(self, 0);
                        for i in 0..4u32 {
                            let (sh, ff) = (k(self, 8 * i), k(self, 0xFF));
                            let sel = self.s.m.bin(c, Bin::LShr, sv, sh);
                            let sel = self.s.m.bin(c, Bin::And, sel, ff);
                            // Select a dword and a byte within it.
                            // Later selects handle codes 8 and above.
                            let four = k(self, 4);
                            let zero = k(self, 0);
                            let high = self.s.m.bin(c, Bin::And, sel, four);
                            let high = self.s.m.cmp(c, pred::NE, high, zero);
                            let word = self.s.m.select(c, high, a, b);
                            let three = k(self, 3);
                            let index = self.s.m.bin(c, Bin::And, sel, three);
                            let bits = self.s.m.bin(c, Bin::Shl, index, three);
                            let x = self.s.m.bin(c, Bin::LShr, word, bits);
                            let mut byte = self.s.m.bin(c, Bin::And, x, ff);
                            // 8-11: the sign of bytes 1, 3 of src1, then of src0.
                            for (n, src, bit) in [(8, b, 15), (9, b, 31), (10, a, 15), (11, a, 31)]
                            {
                                let left = k(self, 31 - bit);
                                let t = self.s.m.bin(c, Bin::Shl, src, left);
                                let k31 = k(self, 31);
                                let t = self.s.m.bin(c, Bin::AShr, t, k31);
                                let t = self.s.m.bin(c, Bin::And, t, ff);
                                let kn = k(self, n);
                                let is = self.s.m.cmp(c, pred::EQ, sel, kn);
                                byte = self.s.m.select(c, is, t, byte);
                            }
                            let (k12, z) = (k(self, 12), k(self, 0));
                            let is12 = self.s.m.cmp(c, pred::EQ, sel, k12);
                            byte = self.s.m.select(c, is12, z, byte);
                            let above = self.s.m.cmp(c, pred::IUGT, sel, k12);
                            byte = self.s.m.select(c, above, ff, byte);
                            let at = k(self, 8 * i);
                            let byte = self.s.m.bin(c, Bin::Shl, byte, at);
                            r = self.s.m.bin(c, Bin::Or, r, byte);
                        }
                        return Ok(r);
                    }
                };
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let mut r = k(self, 0);
                for i in 0..4 {
                    let pick = sel >> (8 * i) & 0xFF;
                    let byte = match pick {
                        0..=7 => {
                            let (src, n) = if pick < 4 { (b, pick) } else { (a, pick - 4) };
                            let sh = k(self, 8 * n);
                            let t = self.s.m.bin(c, Bin::LShr, src, sh);
                            let ff = k(self, 0xFF);
                            self.s.m.bin(c, Bin::And, t, ff)
                        }
                        8..=11 => {
                            let src = if pick < 10 { b } else { a };
                            let bit = if pick % 2 == 0 { 15 } else { 31 };
                            let left = k(self, 31 - bit);
                            let t = self.s.m.bin(c, Bin::Shl, src, left);
                            let k31 = k(self, 31);
                            let t = self.s.m.bin(c, Bin::AShr, t, k31);
                            let ff = k(self, 0xFF);
                            self.s.m.bin(c, Bin::And, t, ff)
                        }
                        12 => continue,
                        _ => k(self, 0xFF),
                    };
                    let at = k(self, 8 * i);
                    let byte = self.s.m.bin(c, Bin::Shl, byte, at);
                    r = self.s.m.bin(c, Bin::Or, r, byte);
                }
                r
            }
            "v_bfm_b32" => {
                // ((1 << src0[4:0]) - 1) << src1[4:0]
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                let one = k(self, 1);
                let (a, b) = (m31(self, a), m31(self, b));
                let m = self.s.m.bin(c, Bin::Shl, one, a);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                self.s.m.bin(c, Bin::Shl, m, b)
            }
            "v_cvt_pk_u8_f32" => {
                // src0 rounded to nearest even and saturated to a byte, into
                // byte src1[1:0] of src2.
                let f = self.fsrc(inst, 0)?;
                let (b, d) = (s(self, 1)?, s(self, 2)?);
                let f = self.unary(26, f);
                let (zero, max) = (self.s.m.f32(0), self.s.m.f32(255.0f32.to_bits()));
                let f = self.binary(35, f, zero);
                let f = self.binary(36, f, max);
                let x = self.s.m.cast(c, Cast::FpToUi, f, self.s.t.i32);
                let (three, k3) = (k(self, 3), k(self, 3));
                let sh = self.s.m.bin(c, Bin::And, b, three);
                let sh = self.s.m.bin(c, Bin::Shl, sh, k3);
                let x = self.s.m.bin(c, Bin::Shl, x, sh);
                let ff = k(self, 0xFF);
                let hole = self.s.m.bin(c, Bin::Shl, ff, sh);
                let ones = k(self, u32::MAX);
                let hole = self.s.m.bin(c, Bin::Xor, hole, ones);
                let d = self.s.m.bin(c, Bin::And, d, hole);
                self.s.m.bin(c, Bin::Or, d, x)
            }
            "v_ffbl_b32" => {
                // -1 for zero, as DXIL's first-bit-low.
                let a = s(self, 0)?;
                self.s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 32, &[a])
            }
            "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
                // Bits of the mask's low (high) half below this lane, plus src1:
                // with an all-ones mask, the lane index.
                let (a, b) = (s(self, 0)?, s(self, 1)?);
                // Mesh lanes are the console's by construction; pixel and
                // vertex lanes are the host's, as their masks are.
                if self.mask_data == MaskData::Host {
                    // Host lanes: the mask must be one (a ballot) or all lanes.
                    let Fields::Valu(v) = &inst.fields else {
                        unreachable!()
                    };
                    let host = match v.src.first() {
                        Some(&Operand::Const(193)) => true,
                        Some(&Operand::Scalar { code, .. }) => {
                            inst.read(Reg::S(code)).is_some_and(|m| {
                                self.x.st.masks.contains_key(&m) || self.x.st.highs.contains_key(&m)
                            })
                        }
                        _ => false,
                    };
                    if !host {
                        return Err("v_mbcnt of a mask that is no ballot (console lane layout)");
                    }
                    self.s.flags |= flags::WAVE_OPS;
                } else if self.s.mesh.is_none() {
                    self.cross_lane(false)?;
                }
                let lane = self.lane_index();
                let below = if name == "v_mbcnt_lo_u32_b32" {
                    lane
                } else {
                    let (k32, z) = (k(self, 32), k(self, 0));
                    let t = self.s.m.bin(c, Bin::Sub, lane, k32);
                    let low = self.s.m.cmp(c, pred::IULT, lane, k32);
                    self.s.m.select(c, low, z, t)
                };
                // (1 << below) - 1, all ones from 32 on.
                let (one, k32, ones) = (k(self, 1), k(self, 32), k(self, u32::MAX));
                let sh = m31(self, below);
                let m = self.s.m.bin(c, Bin::Shl, one, sh);
                let m = self.s.m.bin(c, Bin::Sub, m, one);
                let full = self.s.m.cmp(c, pred::IUGE, below, k32);
                let m = self.s.m.select(c, full, ones, m);
                let x = self.s.m.bin(c, Bin::And, a, m);
                let n = self
                    .s
                    .op(c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 31, &[x]);
                self.s.m.bin(c, Bin::Add, n, b)
            }
            _ => return Err(name),
        })
    }

    fn compare(&mut self, inst: &Inst, rest: &str) -> Result<(), Why> {
        let (op, ty) = rest.rsplit_once('_').ok_or("a compare")?;
        let c = self.cur;
        let r = match ty {
            "f32" | "f16" if op == "class" => {
                // Bit k of src1 tests class k: signalling NaN, quiet NaN,
                // -inf, -normal, -denormal, -0, +0, +denormal, +normal, +inf.
                let f16 = ty == "f16";
                let x = if f16 {
                    let Fields::Valu(v) = &inst.fields else {
                        unreachable!()
                    };
                    let mut h = self.half_src(inst, 0)?;
                    if v.abs & 1 != 0 {
                        let m = self.s.m.i32(0x7FFF);
                        h = self.s.m.bin(c, Bin::And, h, m);
                    }
                    if v.neg & 1 != 0 {
                        let s = self.s.m.i32(0x8000);
                        h = self.s.m.bin(c, Bin::Xor, h, s);
                    }
                    h
                } else {
                    let x = self.fsrc(inst, 0)?;
                    self.as_int(x)
                };
                let mask = self.src(inst, 1)?;
                let k = |l: &mut Self, v: u32| l.s.m.i32(v);
                let (exp_sh, sign_sh, exp_mask, mant_mask, quiet_bit) = if f16 {
                    (10, 15, 0x1F, 0x03FF, 0x0200)
                } else {
                    (23, 31, 0xFF, 0x7F_FFFF, 0x40_0000)
                };
                let (k_exp, k_sign, k_emax, mant, quiet) = (
                    k(self, exp_sh),
                    k(self, sign_sh),
                    k(self, exp_mask),
                    k(self, mant_mask),
                    k(self, quiet_bit),
                );
                let e = self.s.m.bin(c, Bin::LShr, x, k_exp);
                let e = self.s.m.bin(c, Bin::And, e, k_emax);
                let m = self.s.m.bin(c, Bin::And, x, mant);
                let neg = self.s.m.bin(c, Bin::LShr, x, k_sign);
                let zero = k(self, 0);
                let (e0, emax) = (
                    self.s.m.cmp(c, pred::EQ, e, zero),
                    self.s.m.cmp(c, pred::EQ, e, k_emax),
                );
                let m0 = self.s.m.cmp(c, pred::EQ, m, zero);
                let q = self.s.m.bin(c, Bin::And, x, quiet);
                let q = self.s.m.cmp(c, pred::NE, q, zero);
                // Positive classes 6-9; negative ones are 11 minus those.
                let (normal, inf, zero_c, denorm) =
                    (k(self, 8), k(self, 9), k(self, 6), k(self, 7));
                let t = self.s.m.select(c, m0, zero_c, denorm);
                let t = self.s.m.select(c, e0, t, normal);
                let t = self.s.m.select(c, emax, inf, t);
                let eleven = k(self, 11);
                let mirrored = self.s.m.bin(c, Bin::Sub, eleven, t);
                let is_neg = self.s.m.cmp(c, pred::NE, neg, zero);
                let t = self.s.m.select(c, is_neg, mirrored, t);
                let (one, snan) = (k(self, 1), k(self, 0));
                let nan_class = self.s.m.select(c, q, one, snan);
                let yes = self.s.m.bool(true);
                let nan = self.s.m.bin(c, Bin::Xor, m0, yes);
                let nan = self.s.m.bin(c, Bin::And, nan, emax);
                let t = self.s.m.select(c, nan, nan_class, t);
                let bit = self.s.m.bin(c, Bin::LShr, mask, t);
                let bit = self.s.m.bin(c, Bin::And, bit, one);
                self.s.m.cmp(c, pred::NE, bit, zero)
            }
            "f32" | "f16" => {
                let (a, b) = if ty == "f16" {
                    (self.half_fsrc(inst, 0)?, self.half_fsrc(inst, 1)?)
                } else {
                    (self.fsrc(inst, 0)?, self.fsrc(inst, 1)?)
                };
                if op == "lg" {
                    // Ordered and not equal, false for a NaN, as `olt | ogt`.
                    // `fcmp one` means the same, but the AMD D3D12 driver
                    // runs it as unordered (true for a NaN) where NVIDIA
                    // and WARP do not; olt and ogt agree on all three.
                    let lt = self.s.m.cmp(c, pred::OLT, a, b);
                    let gt = self.s.m.cmp(c, pred::OGT, a, b);
                    self.s.m.bin(c, Bin::Or, lt, gt)
                } else {
                    let p = match op {
                        "f" => pred::FALSE,
                        "lt" => pred::OLT,
                        "eq" => pred::OEQ,
                        "le" => pred::OLE,
                        "gt" => pred::OGT,
                        "ge" => pred::OGE,
                        "o" => pred::ORD,
                        "u" => pred::UNO,
                        "nge" => pred::ULT,
                        "nlg" => pred::UEQ,
                        "ngt" => pred::ULE,
                        "nle" => pred::UGT,
                        "neq" => pred::UNE,
                        "nlt" => pred::UGE,
                        "tru" => pred::TRUE,
                        _ => return Err(inst.name()),
                    };
                    self.s.m.cmp(c, p, a, b)
                }
            }
            "u64" | "i64" => {
                let (a, b) = (self.src64(inst, 0)?, self.src64(inst, 1)?);
                let signed = ty == "i64";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    _ => return Err("a 64-bit compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            "u32" | "i32" | "u16" | "i16" => {
                let (a, b) = if ty.ends_with("16") {
                    let (mut a, mut b) = (self.half_int_src(inst, 0)?, self.half_int_src(inst, 1)?);
                    if ty == "i16" {
                        let k16 = self.s.m.i32(16);
                        for x in [&mut a, &mut b] {
                            let t = self.s.m.bin(c, Bin::Shl, *x, k16);
                            *x = self.s.m.bin(c, Bin::AShr, t, k16);
                        }
                    }
                    (a, b)
                } else {
                    (self.src(inst, 0)?, self.src(inst, 1)?)
                };
                let signed = ty == "i32" || ty == "i16";
                let p = match op {
                    "eq" => pred::EQ,
                    "ne" | "lg" => pred::NE,
                    "lt" if signed => pred::SLT,
                    "le" if signed => pred::SLE,
                    "gt" if signed => pred::SGT,
                    "ge" if signed => pred::SGE,
                    "lt" => pred::IULT,
                    "le" => pred::IULE,
                    "gt" => pred::IUGT,
                    "ge" => pred::IUGE,
                    "f" | "t" => {
                        self.lane(inst, Some(op == "t"));
                        return Ok(());
                    }
                    _ => return Err("an integer compare"),
                };
                self.s.m.cmp(c, p, a, b)
            }
            _ => return Err(inst.name()),
        };
        self.lane_bits_for(inst, r);
        Ok(())
    }

    fn lane(&mut self, inst: &Inst, value: Option<bool>) {
        let b = self.s.m.bool(value == Some(true));
        self.lane_bits_for(inst, b);
    }

    fn lane_bits_for(&mut self, inst: &Inst, r: V) {
        let Fields::Valu(v) = &inst.fields else {
            return;
        };
        let code = if inst.name().starts_with("v_cmpx") {
            Some(EXEC)
        } else {
            match v.dst {
                Some(Operand::Scalar { code, .. }) => Some(code),
                _ => None,
            }
        };
        if let Some(lo) = code.and_then(|c| inst.write(Reg::S(c))) {
            self.lane_bits.insert(lo, r);
        }
        for &(_, w) in &inst.writes {
            self.vals
                .entry(w)
                .or_insert(Val::Missing("a lane mask read as data"));
        }
    }

    fn interp(&mut self, inst: &Inst, site: Site) -> Result<(), Why> {
        let Some(&interp) = self.interps.get(&site) else {
            return Err("the first half of an interpolation");
        };
        let name = inst.name();
        let (dst, src) = match &inst.fields {
            Fields::Vintrp { dst, src, .. } => (*dst, Some(*src)),
            Fields::Valu(v) => {
                if v.abs != 0 || v.neg != 0 {
                    return Err("interpolation operand modifiers");
                }
                (v.dst.ok_or("an interpolation")?, v.src.get(1).copied())
            }
            _ => unreachable!(),
        };
        let Operand::Vector { reg, .. } = dst else {
            return Err("an interpolation");
        };
        // A float operand of the instruction.
        let float = |l: &mut Self, op: Option<Operand>| -> Result<V, Why> {
            let x = l.operand(inst, &op.ok_or("an interpolation operand")?)?;
            Ok(l.as_float(x))
        };
        // What the result adds to the evaluated channel (a second half's
        // first half, when the halves evaluate on their own).
        let mut add = None;
        let (attribute, eval) = match *interp {
            Interp::Interpolated {
                attribute,
                barycentric,
                ..
            } => {
                let mode = interp_mode(barycentric).ok_or("pull-model interpolation")?;
                let eval = if self.per_vertex(attribute.row) {
                    // The element is read per vertex elsewhere, so it is
                    // constant: weigh its vertices with SV_Barycentrics.
                    let mut ij = [None; 2];
                    for (index, w) in ij.iter_mut().enumerate() {
                        let x = match self.barycentric(barycentric, index as u8) {
                            Val::I32(x) => x,
                            Val::Missing(w) => return Err(w),
                            Val::Bit(_) => unreachable!(),
                        };
                        *w = Some(self.as_float(x));
                    }
                    Eval::Weights {
                        p0: true,
                        i: ij[0],
                        j: ij[1],
                    }
                } else {
                    Eval::Mode(mode)
                };
                (attribute, eval)
            }
            // P0, the provoking vertex: a flat (nointerpolation) input.
            Interp::Vertex {
                attribute,
                vertex: 2,
            } => (attribute, Eval::Mode(1)),
            // P10 and P20: what the first and second halves weigh, P1 - P0
            // and P2 - P0, which is what the hardware holds for an
            // attribute the program also interpolates.
            Interp::Vertex { attribute, vertex } if self.interpolated(attribute.row) => {
                let one = Some(self.s.m.f32(1f32.to_bits()));
                let (i, j) = if vertex == 0 {
                    (one, None)
                } else {
                    (None, one)
                };
                (attribute, Eval::Weights { p0: false, i, j })
            }
            Interp::Vertex { .. } => return Err("a non-provoking vertex value"),
            // A half of an interpolation with computed i or j: P0 + i P10,
            // or the old dst (f16: the third operand) + j P20.
            Interp::Weighted { attribute } if name.starts_with("v_interp_p1") => {
                let i = Some(float(self, src)?);
                (
                    attribute,
                    Eval::Weights {
                        p0: true,
                        i,
                        j: None,
                    },
                )
            }
            Interp::Weighted { attribute } => {
                let old = match &inst.fields {
                    Fields::Valu(v) => v.src.get(2).copied(),
                    _ => Some(dst),
                };
                add = Some(float(self, old)?);
                let j = Some(float(self, src)?);
                (
                    attribute,
                    Eval::Weights {
                        p0: false,
                        i: None,
                        j,
                    },
                )
            }
        };
        let packing = match &self.x.native.io {
            StageIo::Ps { inputs, .. } => inputs
                .iter()
                .find(|i| i.row == attribute.row && i.fp16)
                .map(|i| {
                    let b = if i.row2 != i.row {
                        Some(i.row2)
                    } else if i.pack2 {
                        half_input_partners(&self.x.native.io, &self.s.inputs)
                            .get(&i.row)
                            .copied()
                    } else {
                        None
                    };
                    (i.pack2, b)
                }),
            _ => None,
        };
        let f16 = name.ends_with("_f16");
        // The evaluated channel plus `add`, as a float.
        let total = |l: &mut Self, row: u8, ch: u8| -> Result<V, Why> {
            let x = l.eval_channel(row, ch, eval)?;
            let x = l.as_float(x);
            Ok(match add {
                Some(a) => l.s.m.bin(l.cur, Bin::Add, a, x),
                None => x,
            })
        };
        let x = match packing {
            None if f16 || attribute.high => return Err("a 16-bit read of a 32-bit input"),
            None => {
                let x = total(self, attribute.row, attribute.channel)?;
                self.as_int(x)
            }
            Some((pack2, b)) => {
                // 16-bit values: one in the low half of each dword, or two
                // per dword (`packed_half`); the host's are 32-bit floats.
                let half = |l: &mut Self, h: u8| -> Result<Option<V>, Why> {
                    let src = if pack2 {
                        packed_half(&l.s.inputs, attribute.row, b, h)
                    } else {
                        h.is_multiple_of(2).then_some((attribute.row, h / 2))
                    };
                    match src {
                        Some((row, ch)) => total(l, row, ch).map(Some),
                        None => Ok(None),
                    }
                };
                let to_f16 = |l: &mut Self, f: V| {
                    let i32t = l.s.t.i32;
                    l.s.op(l.cur, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f])
                };
                let h = 2 * attribute.channel + u8::from(attribute.high);
                let c = self.cur;
                if name == "v_interp_p1ll_f16" {
                    // An f32 result from 16-bit vertex values.
                    let f = half(self, h)?.unwrap_or_else(|| self.s.m.f32(0));
                    self.as_int(f)
                } else if f16 {
                    let lo = match half(self, h)? {
                        Some(f) => to_f16(self, f),
                        None => self.s.m.i32(0),
                    };
                    self.low_half(inst, reg, lo)
                } else if matches!(eval, Eval::Mode(1)) {
                    // The whole dword of a flat input: both halves.
                    let h = 2 * attribute.channel;
                    let z = self.s.m.i32(0);
                    let lo = match half(self, h)? {
                        Some(f) => to_f16(self, f),
                        None => z,
                    };
                    let hi = match half(self, h + 1)? {
                        Some(f) => to_f16(self, f),
                        None => z,
                    };
                    let k16 = self.s.m.i32(16);
                    let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                    self.s.m.bin(c, Bin::Or, lo, hi)
                } else {
                    return Err("32-bit interpolation of a 16-bit input");
                }
            }
        };
        // A pixel interpolates in every lane; guards only hide inactive ones.
        self.write(inst, Reg::V(reg), x, None);
        Ok(())
    }

    /// Whether the program reads attribute row `row` per vertex (an
    /// interpolation with computed barycentrics, or P10 or P20): then its
    /// element is declared constant and read with `attributeAtVertex`.
    fn per_vertex(&self, row: u8) -> bool {
        self.x.io.interps.iter().any(|(_, i)| match *i {
            Interp::Weighted { attribute } => attribute.row == row,
            Interp::Vertex { attribute, vertex } => {
                attribute.row == row && vertex != 2 && self.interpolated(row)
            }
            Interp::Interpolated { .. } => false,
        })
    }

    /// Whether the program interpolates attribute row `row` (so the
    /// hardware holds P1 - P0 and P2 - P0 for it).
    fn interpolated(&self, row: u8) -> bool {
        self.x.io.interps.iter().any(|(_, i)| match *i {
            Interp::Interpolated { attribute, .. } | Interp::Weighted { attribute } => {
                attribute.row == row
            }
            Interp::Vertex { .. } => false,
        })
    }

    /// Channel `ch` of input row `row` evaluated as `eval` says, as raw bits.
    fn eval_channel(&mut self, row: u8, ch: u8, eval: Eval) -> Result<V, Why> {
        let (id, col) = self
            .find_input(row, ch)
            .ok_or("an input the signature lacks")?;
        let mode = match eval {
            Eval::Mode(mode) => mode,
            Eval::Weights { .. } => 1,
        };
        let e = &mut self.s.inputs[id];
        if e.interp != mode && e.usage != 0 {
            return Err("one input interpolated two ways");
        }
        e.interp = mode;
        let Eval::Weights { p0, i, j } = eval else {
            return match self.load_any_input(id, col) {
                Val::I32(x) => Ok(x),
                _ => Err("an input"),
            };
        };
        let v0 = self.vertex_input(id, col, 0);
        let c = self.cur;
        let mut acc = p0.then_some(v0);
        for (k, w) in [(1, i), (2, j)] {
            let Some(w) = w else { continue };
            let vk = self.vertex_input(id, col, k);
            let d = self.s.m.bin(c, Bin::Sub, vk, v0);
            acc = Some(match acc {
                Some(a) => self.fmad(w, d, a),
                None => self.s.m.bin(c, Bin::Mul, w, d),
            });
        }
        let acc = acc.ok_or("an interpolation")?;
        Ok(self.as_int(acc))
    }

    /// Vertex `k` (0 the provoking vertex, as `barycentric` orders them)
    /// of a constant input element's channel, as a float.
    fn vertex_input(&mut self, id: usize, col: u8, k: u8) -> V {
        let e = &mut self.s.inputs[id];
        e.usage |= 1 << col;
        let int = e.comp != 9;
        let (ty, suffix) = if int {
            (self.s.t.i32, ".i32")
        } else {
            (self.s.t.f32, ".f32")
        };
        self.s.flags |= flags::BARYCENTRICS;
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.i8(k),
        ];
        let v = self.s.op(
            entry,
            &format!("attributeAtVertex{suffix}"),
            Attr::ReadNone,
            ty,
            137,
            &args,
        );
        if int {
            self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.f32)
        } else {
            v
        }
    }

    /// `x` (an f16 in the low bits) as the low half of VGPR `reg`: 16-bit
    /// results keep the register's high half (GFX10).
    fn low_half(&mut self, inst: &Inst, reg: u8, x: V) -> V {
        let c = self.cur;
        let old = self.old_i32_of(
            inst.read(Reg::V(reg)),
            false,
            Some("the first half of an interpolation"),
        );
        let (high, low) = (self.s.m.i32(0xFFFF_0000), self.s.m.i32(0xFFFF));
        let old = self.s.m.bin(c, Bin::And, old, high);
        let x = self.s.m.bin(c, Bin::And, x, low);
        self.s.m.bin(c, Bin::Or, old, x)
    }

    /// A 64-bit image atomic: each value a VGPR pair, low dword first, on a
    /// 64-bit integer texture UAV (the console's R32G32 view; its handle
    /// is tagged U32 as DXC tags one). `code` is the DXIL binary operation,
    /// `u32::MAX` a compare-exchange.
    #[allow(clippy::too_many_arguments)]
    fn image_atomic64(
        &mut self,
        inst: &Inst,
        site: Site,
        src: &Source,
        kind: u32,
        cs: Vec<V>,
        reg: u8,
        guard: Option<V>,
        op: &str,
        code: u32,
    ) -> Result<(), Why> {
        let Fields::Image { cache, .. } = &inst.fields else {
            unreachable!()
        };
        let c = self.cur;
        let i64t = self.s.m.int(64);
        self.s.flags |= flags::INT64 | flags::ATOMIC_INT64_TYPED;
        if matches!(src, Source::Loaded { .. }) {
            self.s.flags |= flags::ATOMIC_INT64_HEAP;
        }
        let pair = |l: &mut Self, r: u8| -> Result<V, Why> {
            let lo = l.i32_of(inst.read(Reg::V(r)).ok_or("image atomic data")?)?;
            let hi = l.i32_of(inst.read(Reg::V(r + 1)).ok_or("image atomic data")?)?;
            let lo = l.s.m.cast(c, Cast::ZExt, lo, i64t);
            let hi = l.s.m.cast(c, Cast::ZExt, hi, i64t);
            let k32 = l.s.m.int_const(i64t, 32);
            let hi = l.s.m.bin(c, Bin::Shl, hi, k32);
            Ok(l.s.m.bin(c, Bin::Or, lo, hi))
        };
        // Lanes the guard skips keep their registers.
        let old = pair(self, reg)?;
        let mut value = old;
        if op == "sub" {
            let z = self.s.m.int_const(i64t, 0);
            value = self.s.m.bin(c, Bin::Sub, z, value);
        }
        // A heap handle of its own: `handle` keys its cache by resource kind
        // only, so a float4 handle of the same view (a size query, say)
        // would come back with the wrong element type.
        let props = (kind | 0x1000, 0x105);
        let h = match src {
            Source::Loaded { load, first } => {
                let index = self.heap_index(*load, *first, false)?;
                let per_lane = self.x.res.waterfall.loads.contains(load);
                self.s.heap_handle(self.cur, index, false, per_lane, props)
            }
            _ => self.handle(site, src, false, props)?,
        };
        let at = self.begin_guarded(guard);
        let mut args = vec![h];
        let r = if code == u32::MAX {
            // cmpswap: data is (new, compare), a pair each.
            let cmp = pair(self, reg + 2)?;
            args.extend(cs);
            args.extend([cmp, value]);
            self.s.op(
                self.cur,
                "atomicCompareExchange.i64",
                Attr::NoUnwind,
                i64t,
                79,
                &args,
            )
        } else {
            args.push(self.s.m.i32(code));
            args.extend(cs);
            args.push(value);
            self.s
                .op(self.cur, "atomicBinOp.i64", Attr::NoUnwind, i64t, 78, &args)
        };
        let r = self.end_guarded(at, &[(r, old)])[0];
        if cache.glc {
            let (lo, hi) = self.split64(r);
            self.write(inst, Reg::V(reg), lo, None);
            self.write(inst, Reg::V(reg + 1), hi, None);
        }
        Ok(())
    }

    fn image(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Image {
            data,
            addr,
            dmask,
            dim,
            unorm,
            a16,
            d16,
            tfe,
            lwe,
            cache,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        if *tfe || *lwe {
            return Err("an image access with tfe or lwe");
        }
        let name = inst.name();
        let store = name.starts_with("image_store");
        // A load or sample with no channel enabled returns the first.
        let dmask = if !store && *dmask == 0 { &1 } else { dmask };
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let (kind, coords) = match dim {
            0 => (1, 1),
            1 => (2, 2),
            2 => (4, 3),
            // Sampled (face coordinates and face, see `cube_direction`) or
            // sized only; `TextureCubeArray` (kind 9) when the program uses
            // the cube slice.
            3 if name.starts_with("image_sample")
                || name.starts_with("image_gather")
                || name == "image_get_resinfo" =>
            {
                (
                    if self.cube_arrays.contains(&site) {
                        9
                    } else {
                        5
                    },
                    3,
                )
            }
            4 => (6, 2),
            5 => (7, 3),
            _ => return Err("an image dimension (cube loads or MSAA)"),
        };
        // Address VGPRs in order.
        let mut regs = Vec::new();
        for a in addr {
            if let Operand::Vector { reg, dwords } = *a {
                regs.extend((0..dwords).map(|k| reg + k));
            }
        }
        // With a16, two 16-bit components per VGPR, low half first.
        let addr_val = |l: &mut Self, k: usize| -> Result<V, Why> {
            let r = *regs
                .get(if *a16 { k / 2 } else { k })
                .ok_or("an image address")?;
            let v = inst.read(Reg::V(r)).ok_or("an image address")?;
            let x = l.i32_of(v)?;
            if !*a16 {
                return Ok(x);
            }
            let c = l.cur;
            let (sh, mask) = (l.s.m.i32(16 * (k as u32 % 2)), l.s.m.i32(0xFFFF));
            let x = l.s.m.bin(c, Bin::LShr, x, sh);
            Ok(l.s.m.bin(c, Bin::And, x, mask))
        };
        // The words before the coordinates (texel offsets, bias, reference)
        // take a whole VGPR each, a16 or not; with a16 the bias is an f16 in
        // its low half, and the 16-bit coordinates start at the next VGPR.
        let extra_dword = |l: &mut Self, k: usize| -> Result<V, Why> {
            let r = *regs.get(k).ok_or("an image address")?;
            let v = inst.read(Reg::V(r)).ok_or("an image address")?;
            l.i32_of(v)
        };
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked image")?;
        let uav = store || self.is_uav(src);
        let props = (kind | uav_bits(uav, cache.glc), 0x409);
        if let Some(op) = name.strip_prefix("image_atomic_") {
            // DXIL's atomic operations on a 32-bit integer texture UAV; the
            // old value returns with glc. Sub is the add of the negation.
            let code = match op {
                "add" | "sub" => 0,
                "and" => 1,
                "or" => 2,
                "xor" => 3,
                "smin" => 4,
                "smax" => 5,
                "umin" => 6,
                "umax" => 7,
                "swap" => 8,
                "cmpswap" => u32::MAX,
                // No DXIL operation: a compare-and-swap loop (`atomic_loop`).
                "inc" | "dec" | "fcmpswap" | "fmin" | "fmax" => ATOMIC_LOOP,
                _ => return Err(name),
            };
            if *d16 || *a16 {
                return Err("a 16-bit image atomic");
            }
            // dmask: the data dwords (the compare-and-swaps: twice the
            // value's). Two dwords per value are a 64-bit atomic on an
            // R32G32 view (except compare-and-swap loops, which stay 32-bit).
            let pair = matches!(op, "cmpswap" | "fcmpswap");
            let wide = match (*dmask, pair) {
                (1, false) | (3, true) => false,
                (3, false) | (0xF, true) if code != ATOMIC_LOOP => true,
                _ => return Err("an image atomic's data mask"),
            };
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image atomic data");
            };
            if wide {
                return self.image_atomic64(inst, site, src, kind, cs, reg, guard, op, code);
            }
            let vals = self.vgprs(inst, data, if pair { 2 } else { 1 })?;
            let mut value = vals[0];
            if op == "sub" {
                let z = self.s.m.i32(0);
                value = self.s.m.bin(c, Bin::Sub, z, value);
            }
            let old = match inst.read(Reg::V(reg)) {
                Some(v) if cache.glc => self.i32_of(v)?,
                _ => self.s.m.i32(0),
            };
            let h = self.handle(site, src, false, (kind | uav_bits(true, cache.glc), 0x105))?;
            let at = self.begin_guarded(guard);
            let i32t = self.s.t.i32;
            let mut args = vec![h];
            let r = if code == ATOMIC_LOOP {
                self.atomic_loop(h, &cs, op, &vals)
            } else if code == u32::MAX {
                // cmpswap: data is (new, compare).
                args.extend(cs);
                args.extend([vals[1], vals[0]]);
                self.s.op(
                    self.cur,
                    "atomicCompareExchange.i32",
                    Attr::NoUnwind,
                    i32t,
                    79,
                    &args,
                )
            } else {
                args.push(self.s.m.i32(code));
                args.extend(cs);
                args.push(value);
                self.s
                    .op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args)
            };
            let r = self.end_guarded(at, &[(r, old)])[0];
            if cache.glc {
                self.write(inst, Reg::V(reg), r, None);
            }
            return Ok(());
        }
        if store {
            if name != "image_store" {
                return Err("an image store to a mip level");
            }
            if *dmask == 0 {
                return Ok(());
            }
            // DXIL requires typed UAV `textureStore` to write all four
            // components (`0xF`); accept contiguous masks from channel 0
            // (`0x1`, `0x3`, `0x7`, `0xF`) used for 1..=4-channel UAVs and
            // zero-pad the unused trailing channels.
            if !matches!(*dmask, 0x1 | 0x3 | 0x7 | 0xF) {
                return Err("an image store of some channels");
            }
            let channels = dmask.count_ones() as usize;
            let iu = self.s.m.undef(self.s.t.i32);
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let Operand::Vector { reg, .. } = *data else {
                return Err("image store data");
            };
            let zero = self.s.m.f32(0);
            let mut vals = vec![zero; 4];
            for (k, slot) in vals.iter_mut().enumerate().take(channels) {
                let k = k as u8;
                // d16: two f16 channels per VGPR, low half first.
                let r = if *d16 { k / 2 } else { k };
                let v = inst.read(Reg::V(reg + r)).ok_or("image store data")?;
                *slot = if *d16 {
                    let mut x = self.i32_of(v)?;
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(self.cur, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    self.s
                        .op(self.cur, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x])
                } else {
                    self.float_of(v)?
                };
            }
            let h = self.handle(site, src, false, props)?;
            let at = self.begin_guarded(guard);
            let mut args = vec![h];
            args.extend(cs);
            args.extend(vals);
            args.push(self.s.m.i8(0xF));
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                "textureStore.f32",
                Attr::NoUnwind,
                void,
                67,
                &args,
            );
            self.end_guarded(at, &[]);
            return Ok(());
        }
        let (op, sample) = match name {
            "image_sample" | "image_sample_o" | "image_sample_cl" => (60, true),
            "image_sample_lz" | "image_sample_l" => (62, true),
            "image_sample_b" | "image_sample_b_cl" => (61, true),
            "image_sample_c_lz" => (65, true),
            "image_sample_d" | "image_sample_d_cl" => (63, true),
            "image_get_lod" if self.s.pixel => (81, true),
            // Implicit derivatives: pixel shaders only.
            "image_sample_c" if self.s.pixel => (64, true),
            "image_gather4" | "image_gather4_lz" | "image_gather4_lz_o" | "image_gather4_l"
            | "image_gather4_l_o" => (73, true),
            "image_load" | "image_load_mip" => (66, false),
            "image_get_resinfo" => (72, false),
            _ => return Err(name),
        };
        if op == 72 {
            // Width, height, depth or layers, levels (per dmask) of the level
            // in the first address. Non-cube views return zero for unused
            // extents. A UAV still reports one level.
            let h = self.handle(site, src, false, props)?;
            let mip = if uav {
                self.s.m.undef(self.s.t.i32)
            } else {
                addr_val(self, 0)?
            };
            let i32t = self.s.t.i32;
            let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
            let d = self
                .s
                .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, mip]);
            let one = self.s.m.i32(1);
            // Keep the existing cube depth fallback.
            let missing_extent = self.s.m.i32(u32::from(kind == 5));
            let present = match kind {
                1 => [true, false, false, true],
                2 | 5 | 6 => [true, true, false, true],
                _ => [true, true, true, true],
            };
            let Operand::Vector { reg, .. } = *data else {
                return Err("an image result");
            };
            for (k, ch) in (0..4u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                let mut x = if ch == 3 && uav {
                    one
                } else if !present[ch as usize] {
                    missing_extent
                } else {
                    self.s.m.extract(c, d, ch)
                };
                if kind == 9 && ch == 2 {
                    // D3D reports cubes; the console's `resinfo` counts faces.
                    let six = self.s.m.i32(6);
                    x = self.s.m.bin(c, Bin::Mul, x, six);
                }
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if uav && op != 66 {
            return Err("sampling a UAV");
        }
        let h = self.handle(site, src, false, props)?;
        let fu = self.s.m.undef(self.s.t.f32);
        let iu = self.s.m.undef(self.s.t.i32);
        let (rs, pick) = if sample {
            let ss = self
                .descriptors
                .get(&(site, true))
                .copied()
                .ok_or("an untracked sampler")?;
            // Comparison: the reference value is the first address after
            // the bias, and the sampler is a comparison sampler.
            let cmp = matches!(op, 64 | 65);
            if cmp && (*a16 || dmask.count_ones() != 1) {
                return Err("a comparison sample of 16-bit coordinates or several channels");
            }
            let (shs, pick) =
                self.sampler_choice(site, ss, (14 | if cmp { 0x8000 } else { 0 }, 0))?;
            // Texel offsets (`_o`) come first: x in bits 0-5, y in 8-13,
            // signed (programmable offsets: gathers only).
            let offset_word = name.ends_with("_o");
            // Each of these takes a whole dword, with a16 too (a bias then
            // is f16 in the low half): the addresses after them start at the
            // next dword, two 16-bit halves per extra.
            let extras = usize::from(offset_word)
                + usize::from(name.starts_with("image_sample_b"))
                + usize::from(cmp);
            let first = extras;
            // Coordinates the size divides (not the array slice); explicit
            // gradients (x then y) come before the coordinates.
            let spatial = coords - usize::from(kind == 6 || kind == 7);
            // Gradients per direction: a cube's are in face space (s, t).
            let gspatial = if matches!(kind, 5 | 9) { 2 } else { spatial };
            let grads = if op == 63 { 2 * gspatial } else { 0 };
            // Where the coordinates start, in address components: with a16
            // two per VGPR, after `first` whole VGPRs.
            let base = if *a16 { 2 * first } else { first };
            let mut cs = vec![fu; 4];
            // A cube sample's face coordinates (s, t, face), for gradients.
            let mut face_coords = None;
            // With a16, sampling coordinates and LOD are f16.
            let to_float = |l: &mut Self, x: V| {
                if *a16 {
                    let f = l.s.t.f32;
                    l.s.op(l.cur, "legacyF16ToF32", Attr::ReadNone, f, 131, &[x])
                } else {
                    l.as_float(x)
                }
            };
            // CalculateLOD takes only spatial coordinates, even for an array.
            let active_coords = if op == 81 { spatial } else { coords };
            for (k, slot) in cs.iter_mut().enumerate().take(active_coords) {
                let x = addr_val(self, base + grads + k)?;
                *slot = to_float(self, x);
            }
            if matches!(kind, 5 | 9) {
                if *unorm || op == 81 {
                    return Err("a cube sample with texel coordinates or a level query");
                }
                let (dir, slice) = self.cube_direction(cs[0], cs[1], cs[2], kind == 9);
                face_coords = Some((cs[0], cs[1], cs[2]));
                cs[..3].copy_from_slice(&dir);
                if let Some(slice) = slice {
                    cs[3] = slice;
                }
            }
            if *unorm {
                // Texel coordinates: divided by the size of level 0.
                let i32t = self.s.t.i32;
                let dims = self.s.m.named_struct("dx.types.Dimensions", vec![i32t; 4]);
                let z = self.s.m.i32(0);
                let d = self
                    .s
                    .op(c, "getDimensions", Attr::ReadOnly, dims, 72, &[h, z]);
                for (k, slot) in cs.iter_mut().enumerate().take(spatial) {
                    let n = self.s.m.extract(c, d, k as u32);
                    let n = self.s.m.cast(c, Cast::UiToFp, n, self.s.t.f32);
                    *slot = self.s.m.bin(c, Bin::SDiv, *slot, n);
                }
            }
            if op == 81 {
                // The clamped then the unclamped level of detail.
                let Operand::Vector { reg, .. } = *data else {
                    return Err("an image result");
                };
                let f32t = self.s.t.f32;
                for (k, ch) in (0..2u32).filter(|ch| dmask >> ch & 1 != 0).enumerate() {
                    let clamped = self.s.m.bool(ch == 0);
                    let mut lods = Vec::new();
                    for &sh in &shs {
                        lods.push(self.s.op(
                            c,
                            "calculateLOD.f32",
                            Attr::ReadOnly,
                            f32t,
                            81,
                            &[h, sh, cs[0], cs[1], cs[2], clamped],
                        ));
                    }
                    let lod = self.picked(c, pick, &lods);
                    let x = self.as_int(lod);
                    self.write(inst, Reg::V(reg + k as u8), x, guard);
                }
                return Ok(());
            }
            let offsets = if matches!(kind, 5 | 9) {
                0
            } else {
                (coords - usize::from(kind == 6 || kind == 7)).min(3)
            };
            let zero = self.s.m.i32(0);
            let mut os = vec![iu; 3];
            for o in os.iter_mut().take(offsets) {
                *o = zero;
            }
            let mut args = vec![h, shs[0]];
            args.extend(&cs);
            let clamp = if name.contains("_cl") {
                let x = addr_val(self, base + grads + coords)?;
                to_float(self, x)
            } else {
                fu
            };
            let sample_bias = |l: &mut Self| -> Result<V, Why> {
                let bias = extra_dword(l, 0)?;
                Ok(if *a16 {
                    let mask = l.s.m.i32(0xFFFF);
                    let half = l.s.m.bin(c, Bin::And, bias, mask);
                    let f = l.s.t.f32;
                    l.s.op(c, "legacyF16ToF32", Attr::ReadNone, f, 131, &[half])
                } else {
                    l.as_float(bias)
                })
            };
            // In compute shaders an implicit level of detail comes from
            // the lanes' quads, as the hardware takes it: DXIL derivatives
            // there need conditions the validator does not grant these
            // programs, so the gradients are read from the quad and the
            // sample takes them explicitly.
            let quad_grads = matches!(op, 60 | 61)
                && self.s.threads.is_some()
                && self.s.mesh.is_none()
                && kind != 5;
            let (dx, dx_op) = match op {
                60 | 61 if quad_grads => {
                    let mut g = self.quad_gradients(&cs[..spatial]);
                    if op == 61 {
                        // A bias scales the footprint by 2^bias.
                        let bias = sample_bias(self)?;
                        let scale = self.unary(21, bias);
                        for x in g.iter_mut().filter(|x| **x != fu) {
                            *x = self.s.m.bin(c, Bin::Mul, *x, scale);
                        }
                    }
                    if offset_word {
                        return Err("a sample offset outside a pixel shader");
                    }
                    args.extend(&os);
                    args.extend(g);
                    args.push(clamp);
                    ("sampleGrad.f32", 63)
                }
                60 => {
                    if offset_word {
                        // DXIL takes immediate offsets in -8..=7; the
                        // console's are 6-bit and may be computed.
                        let w = extra_dword(self, 0)?;
                        let w = self
                            .s
                            .m
                            .int_value(w)
                            .ok_or("a sample offset that is not constant")?
                            as u32;
                        for (k, o) in os.iter_mut().enumerate().take(offsets) {
                            let x = (w >> (8 * k) & 0x3F) as i32;
                            let x = if x >= 32 { x - 64 } else { x };
                            if !(-8..=7).contains(&x) {
                                return Err("a sample offset past -8..7");
                            }
                            *o = self.s.m.i32(x as u32);
                        }
                    }
                    args.extend(&os);
                    args.push(clamp);
                    ("sample.f32", 60)
                }
                61 => {
                    let bias = sample_bias(self)?;
                    args.extend(&os);
                    args.extend([bias, clamp]);
                    ("sampleBias.f32", 61)
                }
                63 => {
                    let mut g = vec![fu; 6];
                    if let Some((s0, t0, face)) = face_coords {
                        // Face-space gradients: the direction is affine in s
                        // and t on one face, so a step's direction less the
                        // sample's is its gradient (the constants cancel).
                        for (k, at) in [(0, 0), (1, 2)] {
                            let ds = addr_val(self, base + at)?;
                            let ds = to_float(self, ds);
                            let dt = addr_val(self, base + at + 1)?;
                            let dt = to_float(self, dt);
                            let s1 = self.s.m.bin(c, Bin::Add, s0, ds);
                            let t1 = self.s.m.bin(c, Bin::Add, t0, dt);
                            let (stepped, _) = self.cube_direction(s1, t1, face, kind == 9);
                            for (n, &d) in stepped.iter().enumerate() {
                                g[3 * k + n] = self.s.m.bin(c, Bin::Sub, d, cs[n]);
                            }
                        }
                    } else {
                        for k in 0..spatial {
                            let x = addr_val(self, base + k)?;
                            g[k] = to_float(self, x);
                            let y = addr_val(self, base + spatial + k)?;
                            g[3 + k] = to_float(self, y);
                        }
                    }
                    // `_cl`: the level-of-detail clamp follows the
                    // coordinates (and the array slice).
                    let clamp = if name.ends_with("_cl") {
                        let x = addr_val(self, base + grads + coords)?;
                        to_float(self, x)
                    } else {
                        fu
                    };
                    args.extend(&os);
                    args.extend(g);
                    args.push(clamp);
                    ("sampleGrad.f32", 63)
                }
                64 | 65 => {
                    let reference = addr_val(self, first - 1)?;
                    let reference = self.as_float(reference);
                    args.extend(&os);
                    args.push(reference);
                    let name = if op == 64 {
                        args.push(fu);
                        "sampleCmp.f32"
                    } else {
                        "sampleCmpLevelZero.f32"
                    };
                    (name, op)
                }
                62 => {
                    let lod = if name == "image_sample_l" {
                        let x = addr_val(self, coords)?;
                        to_float(self, x)
                    } else {
                        self.s.m.f32(0)
                    };
                    args.extend(&os);
                    args.push(lod);
                    ("sampleLevel.f32", 62)
                }
                _ => {
                    if name == "image_gather4" {
                        return Err("a gather with implicit level");
                    }
                    if dmask.count_ones() != 1 {
                        return Err("a gather of several channels");
                    }
                    if offset_word {
                        if matches!(kind, 5 | 9) {
                            return Err("a cube gather with an offset");
                        }
                        let w = extra_dword(self, 0)?;
                        for (k, at) in [(0, 26), (1, 18)] {
                            // Bits at..at+5 sign-extended: shift up, then down.
                            let left = self.s.m.i32(at);
                            let t = self.s.m.bin(c, Bin::Shl, w, left);
                            let k26 = self.s.m.i32(26);
                            os[k] = self.s.m.bin(c, Bin::AShr, t, k26);
                        }
                    }
                    args.extend(&os[..2]);
                    args.push(self.s.m.i32(dmask.trailing_zeros()));
                    ("textureGather.f32", 73)
                }
            };
            // One sample per sampler choice (see `sampler_choice`).
            let mut rs = Vec::new();
            for &sh in &shs {
                args[1] = sh;
                rs.push(
                    self.s
                        .op(c, dx, Attr::ReadOnly, self.s.t.res_ret, dx_op, &args),
                );
            }
            (rs, pick)
        } else {
            let mut cs = vec![iu; 3];
            for (k, slot) in cs.iter_mut().enumerate().take(coords) {
                *slot = addr_val(self, k)?;
            }
            let mip = if name == "image_load_mip" {
                addr_val(self, coords)?
            } else if uav {
                // Our declarations have four channels.
                self.s.flags |= flags::TYPED_UAV_LOADS;
                iu
            } else {
                self.s.m.i32(0)
            };
            let mut args = vec![h, mip];
            args.extend(&cs);
            args.extend([iu, iu, iu]);
            let r = self.s.op(
                c,
                "textureLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                66,
                &args,
            );
            (vec![r], None)
        };
        let Operand::Vector { reg, .. } = *data else {
            return Err("an image result");
        };
        let channels: Vec<u32> = if op == 73 {
            vec![0, 1, 2, 3]
        } else {
            (0..4).filter(|ch| dmask >> ch & 1 != 0).collect()
        };
        let mut xs = Vec::new();
        for ch in channels {
            let mut each = Vec::new();
            for &r in &rs {
                each.push(self.s.m.extract(c, r, ch));
            }
            let x = self.picked(c, pick, &each);
            xs.push(if *d16 {
                // shortcut: d16 results are taken as floats (rounded to
                // nearest even); integer formats would keep 16 bits.
                let i32t = self.s.t.i32;
                self.s
                    .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[x])
            } else {
                self.as_int(x)
            });
        }
        if *d16 {
            // Two channels per VGPR, low half first (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            xs = xs
                .chunks(2)
                .map(|p| match *p {
                    [lo, hi] => {
                        let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                        self.s.m.bin(c, Bin::Or, lo, hi)
                    }
                    _ => p[0],
                })
                .collect();
        }
        for (k, x) in xs.into_iter().enumerate() {
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    /// The direction (and, for `array`, the cube slice) a cube sample's face
    /// coordinates stand for: the console takes s and t in [1, 2]
    /// (`v_cubesc`/`v_cubetc` over `|v_cubema|`, plus 1.5) and `8 * slice +
    /// face` (`v_cubeid`), clamping face ids 6 and 7 to face 5 (-z), which
    /// the host's cube sampling turns back into the same texel.
    fn cube_direction(&mut self, s: V, t: V, face: V, array: bool) -> ([V; 3], Option<V>) {
        let c = self.cur;
        let k = |l: &mut Self, v: f32| l.s.m.f32(v.to_bits());
        let (two, three) = (k(self, 2.0), k(self, 3.0));
        let u = self.s.m.bin(c, Bin::Mul, s, two);
        let u = self.s.m.bin(c, Bin::Sub, u, three);
        let v = self.s.m.bin(c, Bin::Mul, t, two);
        let v = self.s.m.bin(c, Bin::Sub, v, three);
        let z = self.s.m.f32(0x8000_0000);
        let (nu, nv) = (
            self.s.m.bin(c, Bin::Sub, z, u),
            self.s.m.bin(c, Bin::Sub, z, v),
        );
        let (one, mone) = (k(self, 1.0), k(self, -1.0));
        let idx = self.s.m.cast(c, Cast::FpToUi, face, self.s.t.i32);
        let seven = self.s.m.i32(7);
        let f = self.s.m.bin(c, Bin::And, idx, seven);
        let slice = array.then(|| {
            let k3 = self.s.m.i32(3);
            let s = self.s.m.bin(c, Bin::LShr, idx, k3);
            self.s.m.cast(c, Cast::UiToFp, s, self.s.t.f32)
        });
        // Per face +x, -x, +y, -y, +z, -z (6 and 7 clamp to -z on RDNA2).
        let table = [
            [one, mone, u, u, u, nu],
            [nv, nv, one, mone, nv, nv],
            [nu, u, v, nv, one, mone],
        ];
        let dir = table.map(|row| {
            let mut r = row[0];
            for (n, &x) in row.iter().enumerate().skip(1) {
                let kn = self.s.m.i32(n as u32);
                let p = if n == 5 { pred::IUGE } else { pred::EQ };
                let is = self.s.m.cmp(c, p, f, kn);
                r = self.s.m.select(c, is, x, r);
            }
            r
        });
        (dir, slice)
    }

    fn buffer(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Buffer {
            data,
            addr,
            soffset,
            offset,
            offen,
            idxen,
            format,
            lds,
            tfe,
            cache,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let name = inst.name();
        if *lds || *tfe {
            return Err("a buffer access to LDS or with tfe");
        }
        if format.is_some() && !name.starts_with("tbuffer_load_format_") {
            return Err(name);
        }
        let site = (b, i);
        let guard = self.guard(b, i)?;
        let c = self.cur;
        let store = name.starts_with("buffer_store_");
        // DXIL atomic operations: add, and, or, xor, signed and unsigned
        // min and max, exchange; compare-exchange apart.
        let atomic = match name.strip_prefix("buffer_atomic_") {
            None => None,
            Some("add") => Some(0),
            Some("and") => Some(1),
            Some("or") => Some(2),
            Some("xor") => Some(3),
            Some("smin") => Some(4),
            Some("smax") => Some(5),
            Some("umin") => Some(6),
            Some("umax") => Some(7),
            Some("swap") => Some(8),
            Some("cmpswap") => Some(u32::MAX),
            Some(_) => return Err(name),
        };
        // d16: two 16-bit channels per VGPR, low half first.
        let d16 = name.starts_with("buffer_") && name.contains("_format_d16_");
        let plain = name.replace("_format_d16_", "_format_");
        let (typed, count) = match plain
            .strip_prefix("buffer_load_")
            .or(plain.strip_prefix("buffer_store_"))
            .or(plain.strip_prefix("tbuffer_load_"))
        {
            _ if atomic.is_some() => (false, 1),
            Some("dword") => (false, 1),
            Some("dwordx2") => (false, 2),
            Some("dwordx3") => (false, 3),
            Some("dwordx4") => (false, 4),
            Some("format_x") => (true, 1),
            Some("format_xy") => (true, 2),
            Some("format_xyz") => (true, 3),
            Some("format_xyzw") => (true, 4),
            _ => return Err(name),
        };
        let Operand::Vector { reg: areg, .. } = *addr else {
            return Err("a buffer address");
        };
        let vaddr = |l: &mut Self, k: u8| -> Result<V, Why> {
            let v = inst.read(Reg::V(areg + k)).ok_or("a buffer address")?;
            l.i32_of(v)
        };
        let index = if *idxen { Some(vaddr(self, 0)?) } else { None };
        let mut off = self.s.m.i32(u32::from(*offset));
        if *offen {
            let o = vaddr(self, u8::from(*idxen))?;
            off = self.s.m.bin(c, Bin::Add, off, o);
        }
        if !matches!(
            soffset,
            Operand::Scalar { code: NULL, .. } | Operand::Const(128)
        ) {
            let so = self.operand(inst, soffset)?;
            off = self.s.m.bin(c, Bin::Add, off, so);
        }
        let src = self
            .descriptors
            .get(&(site, false))
            .copied()
            .ok_or("an untracked buffer")?;
        let used = self.used_kind(src);
        let (base, extra) = self.built_offset(src)?;
        if base != 0 {
            let b = self.s.m.i32(base);
            off = self.s.m.bin(c, Bin::Add, off, b);
        }
        if let Some(x) = extra {
            off = self.s.m.bin(c, Bin::Add, off, x);
        }
        if used.is_some_and(|u| u.kind == ResourceType::Cbv) && !typed {
            // A constant buffer read per lane (a dynamically indexed array).
            // shortcut: the console's CBV descriptor is taken to have a 16-byte
            // stride, as the vector loads of whole rows suggest.
            let h = self.handle(site, src, false, (13, 0x1_0000))?;
            let byte = match index {
                Some(i) => {
                    let sixteen = self.s.m.i32(16);
                    let rows = self.s.m.bin(c, Bin::Mul, i, sixteen);
                    self.s.m.bin(c, Bin::Add, rows, off)
                }
                None => off,
            };
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            for k in 0..count {
                let four = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, byte, four);
                let x = self.cbuffer_dword(h, a);
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let uav = store || atomic.is_some() || self.is_uav(src);
        let u = uav_bits(uav, cache.glc);
        let iu = self.s.m.undef(self.s.t.i32);
        if let Some(op) = atomic {
            let d = data.as_ref().ok_or("buffer atomic data")?;
            let vals = self.vgprs(inst, d, if op == u32::MAX { 2 } else { 1 })?;
            let (h, coords) = match index {
                Some(index) => {
                    let stride = used.map_or(0, |u| u.stride);
                    if stride != 0 && (stride < 4 || !stride.is_multiple_of(4)) {
                        return Err("a structured buffer of unknown stride");
                    }
                    if stride >= 4 && stride.is_multiple_of(4) {
                        (
                            self.handle(site, src, false, (12 | u, stride))?,
                            [index, off, iu],
                        )
                    } else {
                        let (h, byte) = self.runtime_view(src, index, off, 11 | u)?;
                        (h, [byte, iu, iu])
                    }
                }
                None => match structured_stride(used) {
                    Some(stride) => {
                        let h = self.handle(site, src, false, (12 | u, stride))?;
                        let sv = self.s.m.i32(stride);
                        let idx = self.s.m.bin(c, Bin::UDiv, off, sv);
                        let eoff = self.s.m.bin(c, Bin::URem, off, sv);
                        (h, [idx, eoff, iu])
                    }
                    None => (self.handle(site, src, false, (11 | u, 0))?, [off, iu, iu]),
                },
            };
            let Operand::Vector { reg, .. } = *d else {
                return Err("buffer atomic data");
            };
            let returns = cache.glc;
            let old = match inst.read(Reg::V(reg)) {
                Some(v) if returns => self.i32_of(v)?,
                _ => self.s.m.i32(0),
            };
            let at = self.begin_guarded(guard);
            let i32t = self.s.t.i32;
            let r = if op == u32::MAX {
                // cmpswap: data is (new, compare).
                let mut args = vec![h];
                args.extend(coords);
                args.extend([vals[1], vals[0]]);
                self.s.op(
                    self.cur,
                    "atomicCompareExchange.i32",
                    Attr::NoUnwind,
                    i32t,
                    79,
                    &args,
                )
            } else {
                let mut args = vec![h, self.s.m.i32(op)];
                args.extend(coords);
                args.push(vals[0]);
                self.s
                    .op(self.cur, "atomicBinOp.i32", Attr::NoUnwind, i32t, 78, &args)
            };
            let r = self.end_guarded(at, &[(r, old)])[0];
            if returns {
                self.write(inst, Reg::V(reg), r, None);
            }
            return Ok(());
        }
        if store {
            let regs = if d16 { count.div_ceil(2) } else { count };
            let mut vals =
                self.vgprs(inst, data.as_ref().ok_or("buffer store data")?, regs as u8)?;
            if d16 {
                // shortcut: d16 data is taken as f16 (integer formats would
                // keep 16 bits), widened for the 32-bit float store.
                let mut wide = Vec::new();
                for k in 0..count {
                    let mut x = vals[(k / 2) as usize];
                    if k % 2 == 1 {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                    wide.push(self.as_int(f));
                }
                vals = wide;
            }
            if !typed
                && index.is_none()
                && let Some(stride) = structured_stride(used)
            {
                let h = self.handle(site, src, false, (12 | u, stride))?;
                let sv = self.s.m.i32(stride);
                let mask = self.s.m.i8(1);
                let align = self.s.m.i32(4);
                let at = self.begin_guarded(guard);
                let void = self.s.t.void;
                for (k, &v) in vals.iter().enumerate() {
                    let four = self.s.m.i32(4 * k as u32);
                    let a = self.s.m.bin(self.cur, Bin::Add, off, four);
                    let idx = self.s.m.bin(self.cur, Bin::UDiv, a, sv);
                    let eoff = self.s.m.bin(self.cur, Bin::URem, a, sv);
                    self.s.op(
                        self.cur,
                        "rawBufferStore.i32",
                        Attr::NoUnwind,
                        void,
                        140,
                        &[h, idx, eoff, v, iu, iu, iu, mask, align],
                    );
                }
                self.end_guarded(at, &[]);
                return Ok(());
            }
            let mut args = Vec::new();
            let (op, name) = if typed {
                // shortcut: typed stores are taken to write all four channels
                // of a float format.
                if count != 4 || *offen || *offset != 0 {
                    return Err("a typed buffer store of some channels or at an offset");
                }
                args.push(self.handle(site, src, false, (10 | u, 0x409))?);
                args.extend([index.unwrap_or(self.s.m.i32(0)), iu]);
                for x in vals {
                    args.push(self.as_float(x));
                }
                (69, "bufferStore.f32")
            } else {
                match index {
                    Some(index) => {
                        let stride = used.map_or(0, |u| u.stride);
                        if stride >= 4 && stride.is_multiple_of(4) {
                            args.push(self.handle(site, src, false, (12 | u, stride))?);
                            args.extend([index, off]);
                        } else {
                            let (h, byte) = self.runtime_view(src, index, off, 11 | u)?;
                            args.push(h);
                            args.extend([byte, iu]);
                        }
                    }
                    None => {
                        args.push(self.handle(site, src, false, (11 | u, 0))?);
                        args.extend([off, iu]);
                    }
                }
                args.extend(&vals);
                args.resize(3 + 4, iu);
                (140, "rawBufferStore.i32")
            };
            args.push(self.s.m.i8(if typed { 0xF } else { (1u8 << count) - 1 }));
            if !typed {
                args.push(self.s.m.i32(4));
            }
            let at = self.begin_guarded(guard);
            let void = self.s.t.void;
            self.s.op(self.cur, name, Attr::NoUnwind, void, op, &args);
            self.end_guarded(at, &[]);
            return Ok(());
        }
        if let Some(f) = *format {
            if index.is_some() {
                return Err("a typed-format load by index");
            }
            let h = self.handle(site, src, false, (11 | u, 0))?;
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            for (k, x) in self.format_load(h, off, f, count)?.into_iter().enumerate() {
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        if !typed
            && index.is_none()
            && let Some(stride) = structured_stride(used)
        {
            let Some(Operand::Vector { reg, .. }) = *data else {
                return Err("a buffer result");
            };
            let h = self.handle(site, src, false, (12 | u, stride))?;
            let sv = self.s.m.i32(stride);
            let (mask, align) = (self.s.m.i8(1), self.s.m.i32(4));
            for k in 0..count {
                let four = self.s.m.i32(4 * k);
                let a = self.s.m.bin(c, Bin::Add, off, four);
                let idx = self.s.m.bin(c, Bin::UDiv, a, sv);
                let eoff = self.s.m.bin(c, Bin::URem, a, sv);
                let r = self.s.op(
                    c,
                    "rawBufferLoad.f32",
                    Attr::ReadOnly,
                    self.s.t.res_ret,
                    139,
                    &[h, idx, eoff, mask, align],
                );
                let x = self.s.m.extract(c, r, 0);
                let x = self.as_int(x);
                self.write(inst, Reg::V(reg + k as u8), x, guard);
            }
            return Ok(());
        }
        let r = if typed {
            if *offen || matches!(off, V::Inst(_)) || *offset != 0 {
                return Err("a typed buffer load at an offset");
            }
            if uav {
                self.s.flags |= flags::TYPED_UAV_LOADS;
            }
            let h = self.handle(site, src, false, (10 | u, 0x409))?;
            let index = index.unwrap_or(self.s.m.i32(0));
            self.s.op(
                c,
                "bufferLoad.f32",
                Attr::ReadOnly,
                self.s.t.res_ret,
                68,
                &[h, index, iu],
            )
        } else {
            let mask = self.s.m.i8((1u8 << count) - 1);
            let align = self.s.m.i32(4);
            match index {
                Some(index) => {
                    let stride = used.map_or(0, |u| u.stride);
                    if stride != 0 && (stride < 4 || !stride.is_multiple_of(4)) {
                        return Err("a structured buffer of unknown stride");
                    }
                    if stride >= 4 && stride.is_multiple_of(4) {
                        let h = self.handle(site, src, false, (12 | u, stride))?;
                        self.s.op(
                            c,
                            "rawBufferLoad.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            139,
                            &[h, index, off, mask, align],
                        )
                    } else {
                        let (h, byte) = self.runtime_view(src, index, off, 11 | u)?;
                        self.s.op(
                            c,
                            "rawBufferLoad.f32",
                            Attr::ReadOnly,
                            self.s.t.res_ret,
                            139,
                            &[h, byte, iu, mask, align],
                        )
                    }
                }
                None => {
                    let h = self.handle(site, src, false, (11 | u, 0))?;
                    self.s.op(
                        c,
                        "rawBufferLoad.f32",
                        Attr::ReadOnly,
                        self.s.t.res_ret,
                        139,
                        &[h, off, iu, mask, align],
                    )
                }
            }
        };
        let Some(Operand::Vector { reg, .. }) = *data else {
            return Err("a buffer result");
        };
        if d16 {
            // Two f16 channels per VGPR (an odd last high half 0).
            let k16 = self.s.m.i32(16);
            let i32t = self.s.t.i32;
            for k in 0..count.div_ceil(2) {
                let mut x = None;
                for ch in (2 * k..(2 * k + 2).min(count)).rev() {
                    let f = self.s.m.extract(c, r, ch);
                    let h = self
                        .s
                        .op(c, "legacyF32ToF16", Attr::ReadNone, i32t, 130, &[f]);
                    x = Some(match x {
                        Some(hi) => {
                            let hi = self.s.m.bin(c, Bin::Shl, hi, k16);
                            self.s.m.bin(c, Bin::Or, h, hi)
                        }
                        None => h,
                    });
                }
                self.write(inst, Reg::V(reg + k as u8), x.expect("a channel"), guard);
            }
            return Ok(());
        }
        for k in 0..count {
            let x = self.s.m.extract(c, r, k);
            let x = self.as_int(x);
            self.write(inst, Reg::V(reg + k as u8), x, guard);
        }
        Ok(())
    }

    fn export(&mut self, b: usize, i: usize) -> Result<(), Why> {
        let e = *self
            .exports
            .get(&(b, i))
            .ok_or("an export I/O raising did not see")?;
        if e.compressed {
            return Err("a compressed export");
        }
        let guard = self.guard(b, i)?;
        let mesh = self.s.mesh.is_some();
        let pixel_done = self.s.pixel && e.done;
        let c = self.cur;
        if let Some(g) = guard
            && !mesh
            && pixel_done
        {
            let t = self.s.m.bool(true);
            let kill = self.s.m.bin(c, Bin::Xor, g, t);
            self.s
                .op(c, "discard", Attr::NoUnwind, self.s.t.void, 82, &[kill]);
        }
        if mesh && e.target == Target::Primitive {
            return self.emit_indices(e, guard);
        }
        let param_row = |usage: Usage, row: u8| match usage {
            // The 2020 compiler leaves a mesh program's first entry zeroed
            // (usage "position"): its row is the one no other entry names.
            Usage::Other(0) if mesh => self.unnamed_row(),
            _ => Some(row),
        };
        #[derive(Clone, Copy)]
        enum Extract {
            Full,
            Half(bool),
            Low16,
            High16,
        }
        // (output element, channel, value, extraction) per exported channel;
        // per-primitive outputs are elements of `prims`.
        let mut stores: Vec<(usize, u8, Value, Extract)> = Vec::new();
        let primitive = matches!(e.target, Target::PrimitiveParam { .. });
        let outputs = if primitive {
            &self.s.prims
        } else {
            &self.s.outputs
        };
        let by_kind = |kind: u8| outputs.iter().position(|o| o.kind == kind);
        let has_misc = !primitive && outputs.iter().any(|o| matches!(o.kind, 4 | 5));
        let first_clip: u8 = if has_misc { 2 } else { 1 };
        // 16-bit vertex outputs: the halves each dword carries, as (output
        // element, channel, high half) per channel.
        let halves = match e.target {
            Target::Param { usage, row } => {
                param_row(usage, row).and_then(|r| self.half_outputs.get(&r).map(|&h| (r, h)))
            }
            _ => None,
        };
        if let Some((row, (pack2, b))) = halves {
            for ch in 0..4u8 {
                let Some(value) = e.values[usize::from(ch)] else {
                    continue;
                };
                for high in [false, true] {
                    let h = 2 * ch + u8::from(high);
                    let src = if pack2 {
                        packed_half(outputs, row, b, h)
                    } else {
                        (!high).then_some((row, ch))
                    };
                    let found = src.and_then(|(r, c)| {
                        outputs.iter().position(|o| {
                            o.row == Some(u32::from(r)) && c >= o.col && c < o.col + o.cols
                        })
                    });
                    if let Some(id) = found {
                        let col = src.expect("found").1 - outputs[id].col;
                        stores.push((id, col, value, Extract::Half(high)));
                    }
                }
            }
        }
        for ch in 0..4u8 {
            let Some(value) = e.values[usize::from(ch)] else {
                continue;
            };
            if halves.is_some() {
                break;
            }
            if has_misc && e.target == Target::Position(1) {
                // GFX9+/RDNA2 packs SV_RenderTargetArrayIndex in the low 16
                // bits and SV_ViewportArrayIndex in the high 16 bits of pos1.z.
                if ch == 2 {
                    if let Some(id) = by_kind(4) {
                        stores.push((id, 0, value, Extract::Low16));
                    }
                    if let Some(id) = by_kind(5) {
                        stores.push((id, 0, value, Extract::High16));
                    }
                }
                continue;
            }
            let found = match e.target {
                Target::Color(n) => outputs
                    .iter()
                    .position(|o| o.kind == 16 && o.row == Some(u32::from(n)))
                    .map(|id| (id, ch)),
                Target::Position(0) => by_kind(3).map(|id| (id, ch)),
                Target::Depth => match ch {
                    0 => by_kind(17)
                        .or(by_kind(18))
                        .or(by_kind(19))
                        .map(|id| (id, 0)),
                    1 => by_kind(20).map(|id| (id, 0)),
                    2 => by_kind(14).map(|id| (id, 0)),
                    _ => None,
                },
                Target::Param { usage, row } | Target::PrimitiveParam { usage, row } => {
                    let row = param_row(usage, row);
                    outputs
                        .iter()
                        .position(|o| {
                            row.is_some_and(|r| o.row == Some(u32::from(r)))
                                && ch >= o.col
                                && ch < o.col + o.cols
                        })
                        .map(|id| (id, ch - outputs[id].col))
                }
                Target::Null | Target::Unread => None,
                Target::Position(n) if n >= first_clip && n <= first_clip + 1 => {
                    // Clip then cull distances in semantic order; shifted to
                    // pos2..=pos3 when pos1 carries the misc vector.
                    let mut dists: Vec<usize> = (0..outputs.len())
                        .filter(|&i| matches!(outputs[i].kind, 6 | 7))
                        .collect();
                    dists.sort_by_key(|&i| (outputs[i].kind, outputs[i].index));
                    let flat: Vec<(usize, u8)> = dists
                        .iter()
                        .flat_map(|&i| (0..outputs[i].cols).map(move |c| (i, c)))
                        .collect();
                    // Channels past the distances the signature declares are
                    // ones the clipper does not enable (`PA_CL_VS_OUT_CNTL`).
                    let k = usize::from(n - first_clip) * 4 + usize::from(ch);
                    flat.get(k).copied()
                }
                Target::Position(_) => return Err("clip, cull or point-size exports"),
                Target::Primitive => return Err("a primitive export"),
            };
            if let Some((id, col)) = found {
                stores.push((id, col, value, Extract::Full));
            }
        }
        fn elements<'l>(l: &'l mut Lower, primitive: bool) -> &'l mut Vec<SigElement> {
            if primitive {
                &mut l.s.prims
            } else {
                &mut l.s.outputs
            }
        }
        if stores
            .iter()
            .any(|s| matches!(s.3, Extract::Full) && elements(self, primitive)[s.0].half)
        {
            return Err("a 16-bit output the stage data does not mark");
        }
        // A mesh lane writes its own vertex or primitive, where it exports.
        // Other conditional exports also run only for their enabled lanes;
        // a pixel's final `done` export instead discards the other lanes.
        let at = if mesh || !pixel_done {
            self.begin_guarded(guard)
        } else {
            None
        };
        let (name, opcode) = match (mesh, primitive) {
            (false, _) => ("storeOutput", 5),
            (true, false) => ("storeVertexOutput", 171),
            (true, true) => ("storePrimitiveOutput", 172),
        };
        for (id, col, value, extract) in stores {
            let mut x = self.i32_of(value)?;
            let is_half = matches!(extract, Extract::Half(_));
            match extract {
                Extract::Full => {}
                Extract::Half(high) => {
                    // An f16 half, declared a 32-bit float on the host.
                    let c = self.cur;
                    if high {
                        let k16 = self.s.m.i32(16);
                        x = self.s.m.bin(c, Bin::LShr, x, k16);
                    }
                    let f32t = self.s.t.f32;
                    let f = self
                        .s
                        .op(c, "legacyF16ToF32", Attr::ReadNone, f32t, 131, &[x]);
                    x = self.as_int(f);
                }
                Extract::Low16 => {
                    let mask = self.s.m.i32(0xFFFF);
                    x = self.s.m.bin(self.cur, Bin::And, x, mask);
                }
                Extract::High16 => {
                    let k16 = self.s.m.i32(16);
                    x = self.s.m.bin(self.cur, Bin::LShr, x, k16);
                }
            }
            let (kind, comp) = {
                let el = &mut elements(self, primitive)[id];
                el.usage |= 1 << col;
                (el.kind, el.comp)
            };
            if !mesh && !primitive && matches!(kind, 4 | 5) {
                self.s.flags |= flags::VIEWPORT_RT_INDEX;
            }
            let (suffix, x) = match comp {
                9 if is_half => (".f32", self.as_float(x)),
                9 => {
                    let f = match self.floats.get(&value) {
                        Some(&f) => f,
                        None => self.as_float(x),
                    };
                    (".f32", f)
                }
                _ => (".i32", x),
            };
            let mut args = vec![
                self.s.m.i32(id as u32),
                self.s.m.i32(0),
                self.s.m.i8(col),
                x,
            ];
            if mesh {
                args.push(self.mesh_thread());
            }
            let void = self.s.t.void;
            self.s.op(
                self.cur,
                &format!("{name}{suffix}"),
                Attr::NoUnwind,
                void,
                opcode,
                &args,
            );
        }
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// An LDS access of a merged ES or GS half, which must be a ring access
    /// ([`Io::rings`]). ES: a write to its ESGS item is an output (the row
    /// of the parameter, which must be its index). GS: a read of an input
    /// vertex's item is that vertex's input (row = dword / 4, the ES's
    /// parameter); a write to its GSVS area goes to the per-thread array
    /// the emits read.
    fn ring(&mut self, inst: &Inst, b: usize, i: usize) -> Result<(), Why> {
        let Fields::Ds {
            dst,
            data,
            offset0,
            offset1,
            ..
        } = &inst.fields
        else {
            unreachable!()
        };
        let ring = *self
            .rings
            .get(&(b, i))
            .ok_or("an LDS access off the stage's rings")?;
        let name = inst.name();
        let guard = self.guard(b, i)?;
        let width = match name.rsplit('_').next() {
            Some("b64") => 2,
            Some("b96") => 3,
            Some("b128") => 4,
            Some("b32") => 1,
            _ => return Err("a ring access of bytes or halves"),
        };
        // (byte offset, dwords) per element.
        let elements: Vec<(u32, u32)> =
            if name.starts_with("ds_read2_") || name.starts_with("ds_write2_") {
                let scale = 4 * width;
                vec![
                    (u32::from(*offset0) * scale, width),
                    (u32::from(*offset1) * scale, width),
                ]
            } else if name.starts_with("ds_read_") || name.starts_with("ds_write_") {
                vec![(u32::from(*offset0) | u32::from(*offset1) << 8, width)]
            } else {
                return Err("a ring operation");
            };
        let base = match ring {
            Ring::Vertex { byte, .. } | Ring::Own { byte } => byte,
        };
        let dword = |off: u32| -> Result<u32, Why> {
            let at = base.wrapping_add(off);
            if at % 4 != 0 {
                return Err("a ring access off dwords");
            }
            Ok(at / 4)
        };
        let parts = self.x;
        let gs = match &parts.native.io {
            StageIo::Gs(gs) => Some((gs.max_vertices, gs.vertex_dwords)),
            _ => None,
        };
        if name.starts_with("ds_read") {
            let Ring::Vertex { vertex, .. } = ring else {
                return Err("a ring read of the lane's own area");
            };
            if gs.is_none() || vertex >= 3 {
                return Err("a ring read of an input vertex past the primitive's");
            }
            let reg = match dst {
                Some(Operand::Vector { reg, .. }) => *reg,
                _ => return Err("an LDS read result"),
            };
            let mut n = 0;
            for (off, w) in elements {
                let d0 = dword(off)?;
                for k in 0..w {
                    let d = d0 + k;
                    let (row, col) = ((d / 4) as u8, (d % 4) as u8);
                    // shortcut: a channel the input signature leaves out
                    // reads 0.
                    let x = match self.find_input(row, col) {
                        Some((id, c)) => self.load_vertex_input(id, c, vertex)?,
                        None => self.s.m.i32(0),
                    };
                    self.write(inst, Reg::V(reg + n as u8), x, guard);
                    n += 1;
                }
            }
            return Ok(());
        }
        if !name.starts_with("ds_write") {
            return Err("a ring operation");
        }
        let Ring::Own { .. } = ring else {
            return Err("a ring write to another lane's area");
        };
        let mut stores = Vec::new();
        for (e, (off, w)) in elements.into_iter().enumerate() {
            let vals = self.vgprs(inst, data.get(e).ok_or("an LDS write")?, w as u8)?;
            let d0 = dword(off)?;
            for (k, x) in vals.into_iter().enumerate() {
                stores.push((d0 + k as u32, x));
            }
        }
        let at = self.begin_guarded(guard);
        for (d, x) in stores {
            match gs {
                Some((max, size)) => {
                    if d >= max * size {
                        return Err("a GSVS write past the maximum vertex count");
                    }
                    let g = self.gsvs_array();
                    let index = self.s.m.i32(d);
                    let p = self.s.m.element(self.cur, g, index);
                    self.s.m.store(self.cur, p, x);
                }
                None => {
                    let (param, col) = ((d / 4) as u8, (d % 4) as u8);
                    let StageIo::Vs { outputs, .. } = &parts.native.io else {
                        return Err("an ES half without its stage data");
                    };
                    let o = outputs
                        .iter()
                        .find(|o| o.param == param)
                        .ok_or("an ESGS write the stage data does not list")?;
                    // The GS half reads parameter n as input row n.
                    if o.row != param {
                        return Err("an ESGS parameter of another signature row");
                    }
                    if let Some((id, c)) = self.find_output(o.row, col) {
                        self.store_output(id, c, x)?;
                    }
                }
            }
        }
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// GS: the per-thread array of emitted vertices, the emit count after
    /// them.
    fn gsvs_array(&mut self) -> V {
        if let Some(g) = self.gsvs {
            return g;
        }
        let n = match &self.x.native.io {
            StageIo::Gs(gs) => gs.max_vertices * gs.vertex_dwords + 1,
            _ => 1,
        };
        let ty = self.s.m.array(n, self.s.t.i32);
        let z = self.s.m.i32(0);
        let init = self.s.m.aggregate(ty, &vec![z; n as usize]);
        let g = self.s.m.static_global("gsvs", init);
        self.gsvs = Some(g);
        g
    }

    /// GS: `s_sendmsg` GS_OP_EMIT stores the vertex the copy shader would
    /// export from the GSVS ring and emits it; GS_OP_CUT ends the strip.
    fn gs_message(&mut self, imm: u16) -> Result<(), Why> {
        let (msg, op, stream) = (imm & 0xF, imm >> 4 & 3, imm >> 8 & 3);
        if msg == MSG_GS_DONE {
            return Ok(());
        }
        if msg != MSG_GS {
            return Err("a GS message");
        }
        if stream != 0 {
            return Err("a second GS output stream");
        }
        let void = self.s.t.void;
        let s0 = self.s.m.i8(0);
        if op & 2 != 0 {
            let x = self.x;
            let StageIo::Gs(gs) = &x.native.io else {
                unreachable!()
            };
            let (max, size) = (gs.max_vertices, gs.vertex_dwords);
            let g = self.gsvs_array();
            let c = self.cur;
            let ci = self.s.m.i32(max * size);
            let cp = self.s.m.element(c, g, ci);
            let count = self.s.m.load(c, cp);
            let k = self.s.m.i32(size);
            let first = self.s.m.bin(c, Bin::Mul, count, k);
            for e in &x.io.copy {
                for (ch, src) in e.values.iter().enumerate() {
                    let Some(src) = *src else {
                        continue;
                    };
                    let ch = ch as u8;
                    let found = match e.target {
                        Target::Position(0) => self
                            .s
                            .outputs
                            .iter()
                            .position(|o| o.kind == 3)
                            .map(|id| (id, ch)),
                        Target::Param { row, .. } => self.find_output(row, ch),
                        _ => None,
                    };
                    let Some((id, col)) = found else {
                        continue;
                    };
                    let value = match src {
                        Copied::Constant(k) => self.s.m.i32(k),
                        Copied::Gsvs(d) if d < size => {
                            let d = self.s.m.i32(d);
                            let at = self.s.m.bin(c, Bin::Add, first, d);
                            let p = self.s.m.element(c, g, at);
                            self.s.m.load(c, p)
                        }
                        Copied::Gsvs(_) => return Err("a copy shader read past the vertex"),
                    };
                    self.store_output(id, col, value)?;
                }
            }
            self.s.op(c, "emitStream", Attr::NoUnwind, void, 97, &[s0]);
            let one = self.s.m.i32(1);
            let next = self.s.m.bin(c, Bin::Add, count, one);
            self.s.m.store(c, cp, next);
        }
        if op & 1 != 0 {
            self.s
                .op(self.cur, "cutStream", Attr::NoUnwind, void, 98, &[s0]);
        }
        Ok(())
    }

    /// The output element holding channel `channel` of signature row `row`,
    /// with the channel within it.
    fn find_output(&self, row: u8, channel: u8) -> Option<(usize, u8)> {
        self.s.outputs.iter().enumerate().find_map(|(id, e)| {
            let r = e.row?;
            (r == u32::from(row) && channel >= e.col && channel < e.col + e.cols)
                .then(|| (id, channel - e.col))
        })
    }

    /// `storeOutput` of a 32-bit value to output element `id`.
    fn store_output(&mut self, id: usize, col: u8, x: V) -> Result<(), Why> {
        let el = &mut self.s.outputs[id];
        if el.half {
            return Err("a 16-bit output of a ring");
        }
        el.usage |= 1 << col;
        let (suffix, x) = match el.comp {
            9 => (".f32", self.as_float(x)),
            _ => (".i32", x),
        };
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            x,
        ];
        let void = self.s.t.void;
        self.s.op(
            self.cur,
            &format!("storeOutput{suffix}"),
            Attr::NoUnwind,
            void,
            5,
            &args,
        );
        Ok(())
    }

    /// GS: channel `col` of input element `id` of input vertex `vertex`.
    fn load_vertex_input(&mut self, id: usize, col: u8, vertex: u8) -> Result<V, Why> {
        let e = &mut self.s.inputs[id];
        if e.half {
            return Err("a 16-bit input of a ring");
        }
        e.usage |= 1 << col;
        let int = e.comp != 9;
        let (ty, suffix) = if int {
            (self.s.t.i32, ".i32")
        } else {
            (self.s.t.f32, ".f32")
        };
        let entry = self.s.entry;
        let args = [
            self.s.m.i32(id as u32),
            self.s.m.i32(0),
            self.s.m.i8(col),
            self.s.m.i32(u32::from(vertex)),
        ];
        let v = self.s.op(
            entry,
            &format!("loadInput{suffix}"),
            Attr::ReadNone,
            ty,
            4,
            &args,
        );
        Ok(if int {
            v
        } else {
            self.s.m.cast(entry, Cast::Bitcast, v, self.s.t.i32)
        })
    }

    /// Mesh: `exp prim` as the lane's primitive. Bit 31 culls it (all
    /// indices 0: no area).
    fn emit_indices(&mut self, e: &Export, guard: Option<V>) -> Result<(), Why> {
        let v = e.values[0].ok_or("a primitive export without its indices")?;
        let x = self.i32_of(v)?;
        let at = self.begin_guarded(guard);
        let c = self.cur;
        let z = self.s.m.i32(0);
        let culled = self.s.m.cmp(c, pred::SLT, x, z);
        let mut args = vec![self.mesh_thread()];
        for k in 0..3 {
            let (shift, mask) = (self.s.m.i32(10 * k), self.s.m.i32(0x3FF));
            let i = self.s.m.bin(c, Bin::LShr, x, shift);
            let i = self.s.m.bin(c, Bin::And, i, mask);
            args.push(self.s.m.select(c, culled, z, i));
        }
        let void = self.s.t.void;
        self.s
            .op(c, "emitIndices", Attr::NoUnwind, void, 169, &args);
        self.end_guarded(at, &[]);
        Ok(())
    }

    /// The per-vertex arbitrary row no mesh output entry names but one
    /// (the 2020 compiler's zeroed entry).
    fn unnamed_row(&self) -> Option<u8> {
        let StageIo::Ms(ms) = &self.x.native.io else {
            return None;
        };
        let named = |r: u32| {
            ms.outputs
                .iter()
                .any(|o| o.param & 0x80 == 0 && o.usage != Usage::Other(0) && u32::from(o.row) == r)
        };
        let free: std::collections::BTreeSet<u32> = self
            .s
            .outputs
            .iter()
            .filter(|o| o.kind == 0)
            .filter_map(|o| o.row)
            .filter(|&r| !named(r))
            .collect();
        match free.into_iter().collect::<Vec<_>>()[..] {
            [r] => u8::try_from(r).ok(),
            _ => None,
        }
    }

    fn exit(&mut self, b: usize) -> Result<(), ShaderError> {
        let offset = self.x.p.blocks[b]
            .insts
            .last()
            .map_or(self.x.p.blocks[b].offset, |i| i.machine.offset);
        self.cur_reads.clear();
        self.cur_old_deps.clear();
        self.cur_edge = None;
        let term = match self.x.st.exits[b] {
            LaneExit::Jump(t) if self.counters.contains_key(&t) && self.rpo[t] <= self.rpo[b] => {
                let ok = self.count_iteration(t, pred::IULT);
                let abort = self.abort_block();
                Term::CondBr(ok, self.block[t].expect("reachable"), abort)
            }
            LaneExit::Jump(t) => {
                self.enter_loop(b, t, self.cur);
                Term::Br(self.block[t].expect("reachable"))
            }
            LaneExit::Branch {
                when,
                taken,
                not_taken,
            } => {
                // A negated condition branches the other way round.
                let (cond, taken, not_taken) = match self.x.st.bools.node(when) {
                    Some((atom, Bool::TRUE, Bool::FALSE)) => {
                        let a = self
                            .atom(atom, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (a, not_taken, taken)
                    }
                    _ => {
                        let c = self
                            .boolean(when, self.cur)
                            .map_err(|w| unsupported(offset, w))?;
                        (c, taken, not_taken)
                    }
                };
                let reads = std::mem::take(&mut self.cur_reads);
                if !reads.is_empty() {
                    self.sinks.push(SinkRead {
                        block: b,
                        guard: None,
                        offset,
                        deps: reads,
                    });
                }
                // A back edge is taken only below the iteration limit.
                let back =
                    |l: &Self, to: usize| l.counters.contains_key(&to) && l.rpo[to] <= l.rpo[b];
                // (No `xor` of a compare: a loop exit testing one hung the
                // AMD driver.)
                let cond = if back(self, taken) {
                    let ok = self.count_iteration(taken, pred::IULT);
                    self.s.m.bin(self.cur, Bin::And, cond, ok)
                } else if back(self, not_taken) {
                    let stop = self.count_iteration(not_taken, pred::IUGE);
                    self.s.m.bin(self.cur, Bin::Or, cond, stop)
                } else {
                    cond
                };
                // Critical edges get a block of their own, so loops have
                // dedicated exits and preheaders as DXC emits them.
                let (t, f) = (self.edge(b, taken), self.edge(b, not_taken));
                for (to, via) in [(taken, t), (not_taken, f)] {
                    if !back(self, to) {
                        // From an edge block, or from here.
                        let from = if Some(via) == self.block[to] {
                            self.cur
                        } else {
                            via
                        };
                        self.enter_loop(b, to, from);
                    }
                }
                Term::CondBr(cond, t, f)
            }
            LaneExit::End => {
                if self.s.pixel && !self.exported[b] && !self.has_done_export(b) {
                    // A lane leaving without its final export was killed.
                    let t = self.s.m.bool(true);
                    self.s
                        .op(self.cur, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
                }
                Term::Br(self.ret_block())
            }
            // A merged ES half ends jumping to its GS half.
            LaneExit::Indirect if self.x.native.stage == HwStage::Es => Term::Ret,
            LaneExit::Indirect => return Err(unsupported(offset, "an indirect jump")),
        };
        let _ = Exit::End;
        self.s.m.terminate(self.cur, term);
        Ok(())
    }

    /// A forward edge into loop header `to` (from DXIL block `via`) starts
    /// its count at 0.
    fn enter_loop(&mut self, _from: usize, to: usize, via: usize) {
        if let Some((_, incoming)) = self.counters.get_mut(&to) {
            let z = self.s.m.i32(0);
            incoming.push((z, via));
        }
    }

    /// A back edge to header `to` (a back edge keeps its block, no
    /// critical-edge block): the count goes up; the result compares it with
    /// [`LOOP_LIMIT`] by `p` (below: go on; at or above: stop).
    fn count_iteration(&mut self, to: usize, p: u8) -> V {
        let c = self.cur;
        let (phi, _) = self.counters[&to];
        let one = self.s.m.i32(1);
        let next = self.s.m.bin(c, Bin::Add, phi, one);
        let limit = self.s.m.i32(super::loop_limit());
        self.counters
            .get_mut(&to)
            .expect("header")
            .1
            .push((next, c));
        self.s.m.cmp(c, p, next, limit)
    }

    /// The block a lane stopped by [`LOOP_LIMIT`] leaves through.
    fn abort_block(&mut self) -> usize {
        if let Some(a) = self.abort {
            return a;
        }
        let a = self.s.m.new_block();
        if self.s.pixel {
            let t = self.s.m.bool(true);
            self.s
                .op(a, "discard", Attr::NoUnwind, self.s.t.void, 82, &[t]);
        }
        let r = self.ret_block();
        self.s.m.terminate(a, Term::Br(r));
        self.abort = Some(a);
        a
    }

    /// The function's single returning block (see `ret`).
    fn ret_block(&mut self) -> usize {
        if let Some(r) = self.ret {
            return r;
        }
        let r = self.s.m.new_block();
        self.s.m.terminate(r, Term::Ret);
        self.ret = Some(r);
        r
    }

    /// The DXIL block a branch from IR block `from` to `to` goes to.
    fn edge(&mut self, from: usize, to: usize) -> usize {
        let target = self.block[to].expect("reachable");
        let preds = self.x.p.blocks[to]
            .preds
            .iter()
            .filter(|&&q| self.block[q].is_some())
            .count();
        // Back edges stay as they are: the exiting block is the latch.
        if preds < 2 || self.rpo[to] <= self.rpo[from] {
            return target;
        }
        let e = self.s.m.new_block();
        self.s.m.terminate(e, Term::Br(target));
        self.edges.insert((from, to), e);
        e
    }

    fn fill_phis(&mut self) -> Result<(), ShaderError> {
        let p = self.x.p;
        while let Some((phi, b, i, base)) = self.pending_addr_phis.pop() {
            let ir_phi = &p.blocks[b].phis[i];
            let offset = p.blocks[b].offset;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let saved_cur = self.cur;
                let saved_ir = self.cur_ir;
                self.cur = qb;
                self.cur_ir = q;
                let mut local = HashMap::new();
                let x = self.rel_offset(ir_phi.args[n], base, &mut local);
                self.cur = saved_cur;
                self.cur_ir = saved_ir;
                let x = x.map_err(|w| unsupported(offset, w))?;
                incoming.push((x, qb));
            }
            self.s.m.set_phi(phi, incoming);
        }
        // Phis directly or transitively read by any sink or dependent instruction
        // (for lazy barycentric inputs and pixel/vertex lane-mask ballots).
        let mut live_vals: HashSet<Value> = self
            .sinks
            .iter()
            .flat_map(|s| s.deps.iter())
            .chain(self.handle_deps.values().flatten())
            .map(|dep| match *dep {
                ReadDep::Value(u) | ReadDep::Edge { value: u, .. } => u,
            })
            .collect();
        let mut work: Vec<Value> = live_vals.iter().copied().collect();
        while let Some(v) = work.pop() {
            if let Some(&(b, i)) = self.phi_at.get(&v) {
                for &arg in &p.blocks[b].phis[i].args {
                    if live_vals.insert(arg) {
                        work.push(arg);
                    }
                }
            }
            if let Some(infos) = self.deps.get(&v) {
                for info in infos {
                    for &dep in &info.deps {
                        let u = match dep {
                            ReadDep::Value(u) | ReadDep::Edge { value: u, .. } => u,
                        };
                        if live_vals.insert(u) {
                            work.push(u);
                        }
                    }
                }
            }
        }
        let data_phis = std::mem::take(&mut self.data_phis);
        let mut ballot_edges: HashSet<(usize, usize, usize)> = HashSet::new();
        let mut has_missing = false;
        for &(v, b, i) in &data_phis {
            let phi = &p.blocks[b].phis[i];
            let bit = phi.reg == Reg::Scc;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let arg = phi.args[n];
                if live_vals.contains(&phi.value) && !self.vals.contains_key(&arg) {
                    let _ = self.lazy_barycentric(arg);
                }
                let lo = self.x.st.highs.get(&arg).copied().unwrap_or(arg);
                let ballot_ok = self.mask_data == MaskData::Host
                    && !bit
                    && live_vals.contains(&phi.value)
                    && self.x.st.masks.contains_key(&lo)
                    && self.def_block.get(&lo).is_some_and(|&d| d != self.cur_ir);
                let ballot = if ballot_ok {
                    self.cur_reads.clear();
                    let r = self.mask_dword(arg).ok();
                    self.cur_reads.clear();
                    r
                } else {
                    None
                };
                let x = match (self.vals.get(&arg).copied(), bit) {
                    (Some(Val::I32(x)), false) | (Some(Val::Bit(x)), true) => x,
                    // A mask dword the other path left in the register: its ballot
                    // (a pixel or vertex program reads such a join only through a
                    // zero test or a count). shortcut: the ballot is the host
                    // wave's, so a join read some other way would differ from the
                    // console, which `mask_ok` does not see; upgrade if one shows up.
                    (None | Some(Val::Missing(_)), false) if ballot.is_some() => {
                        ballot_edges.insert((b, i, n));
                        ballot.unwrap()
                    }
                    // A lane mask's dword reaching a join that also takes
                    // data: the registers are reused once the mask is dead.
                    // shortcut: undefined on that edge, assuming the program
                    // reads the joined value only on paths from a data edge.
                    (Some(Val::Missing(_)), false)
                        if self.x.st.masks.contains_key(&arg)
                            || self.x.st.highs.contains_key(&arg) =>
                    {
                        self.s.m.undef(self.s.t.i32)
                    }
                    _ => {
                        has_missing = true;
                        if bit {
                            self.s.m.bool(false)
                        } else {
                            self.s.m.i32(0)
                        }
                    }
                };
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        for (phi, incoming) in std::mem::take(&mut self.counters).into_values() {
            self.s.m.set_phi(phi, incoming);
        }
        while let Some((v, b, i)) = self.pending_masks.pop() {
            let phi = &p.blocks[b].phis[i];
            let offset = p.blocks[b].offset;
            let mut incoming = Vec::new();
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                let Some(qb) = self.tail[q] else { continue };
                let qb = self.edges.get(&(q, b)).copied().unwrap_or(qb);
                let m =
                    *self.x.st.masks.get(&phi.args[n]).ok_or_else(|| {
                        unsupported(offset, "a lane-mask join of an untracked mask")
                    })?;
                self.cur_reads.clear();
                self.cur_edge = None;
                let x = self.boolean(m, qb).map_err(|w| unsupported(offset, w))?;
                let reads = std::mem::take(&mut self.cur_reads);
                if !reads.is_empty() {
                    let deps = reads
                        .into_iter()
                        .map(|dep| match dep {
                            ReadDep::Value(value) => ReadDep::Edge {
                                from: q,
                                to: b,
                                value,
                            },
                            other => other,
                        })
                        .collect();
                    self.sinks.push(SinkRead {
                        block: q,
                        guard: None,
                        offset,
                        deps,
                    });
                }
                incoming.push((x, qb));
            }
            self.s.m.set_phi(v, incoming);
        }
        if (!has_missing && self.direct_missing.is_empty()) || self.sinks.is_empty() {
            return Ok(());
        }
        self.check_missing_reach(&data_phis, &ballot_edges)
    }

    /// Rejects the program if any missing or mismatched phi incoming edge (or
    /// direct missing value surviving via a partial/guarded overwrite) can
    /// reach an observable sink along a feasible path in [`Structured`].
    fn check_missing_reach(
        &self,
        data_phis: &[(V, usize, usize)],
        ballot_edges: &HashSet<(usize, usize, usize)>,
    ) -> Result<(), ShaderError> {
        const MAX_TAINT_ROUNDS: usize = 256;
        let p = self.x.p;
        let st = self.x.st;
        let mut bools = BoolSet::new(&st.bools);
        let mut taint: HashMap<Value, Vec<(Why, Bool)>> = HashMap::new();

        let site_cond =
            |bools: &mut BoolSet, block: usize, guard: Option<Bool>, invert: bool| -> Bool {
                match guard {
                    Some(g) => {
                        let g = if invert { bools.not(g) } else { g };
                        bools.and(st.reach[block], g)
                    }
                    None => st.reach[block],
                }
            };
        let dep_cond = |bools: &mut BoolSet, base: Bool, dep: ReadDep| -> (Value, Bool) {
            match dep {
                ReadDep::Value(u) => (u, base),
                ReadDep::Edge { from, to, value: u } => {
                    let ec = bools.edge(st, from, to);
                    (u, bools.and(base, ec))
                }
            }
        };
        let merge = |taint: &mut HashMap<Value, Vec<(Why, Bool)>>,
                     dst: Value,
                     why: Why,
                     cond: Bool,
                     bools: &mut BoolSet|
         -> bool {
            if cond == Bool::FALSE {
                return false;
            }
            let list = taint.entry(dst).or_default();
            if let Some((_, existing)) = list.iter_mut().find(|(w, _)| *w == why) {
                let next = bools.or(*existing, cond);
                if next != *existing {
                    *existing = next;
                    return true;
                }
                false
            } else {
                list.push((why, cond));
                true
            }
        };

        for &(dst, why, block, guard, invert) in &self.direct_missing {
            let c = site_cond(&mut bools, block, guard, invert);
            merge(&mut taint, dst, why, c, &mut bools);
        }

        for &(_, b, i) in data_phis {
            let phi = &p.blocks[b].phis[i];
            let bit = phi.reg == Reg::Scc;
            for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                if self.tail[q].is_none() {
                    continue;
                }
                let arg = phi.args[n];
                let why = match (self.vals.get(&arg).copied(), bit) {
                    (Some(Val::I32(_)), false) | (Some(Val::Bit(_)), true) => continue,
                    (None | Some(Val::Missing(_)), false)
                        if ballot_edges.contains(&(b, i, n)) =>
                    {
                        continue;
                    }
                    (Some(Val::Missing(_)), false)
                        if st.masks.contains_key(&arg) || st.highs.contains_key(&arg) =>
                    {
                        continue;
                    }
                    (Some(Val::Bit(_)), false) => "an SCC joined with SGPR values",
                    (Some(Val::Missing(why)), _) => why,
                    _ => "a join of mismatched values",
                };
                let mut c = bools.edge(st, q, b);
                if self.rpo[b] <= self.rpo[q] {
                    c = bools.forget(c, self.rpo[b]);
                }
                merge(&mut taint, phi.value, why, c, &mut bools);
            }
        }

        let propagate_step =
            |taint: &mut HashMap<Value, Vec<(Why, Bool)>>, bools: &mut BoolSet, precise: bool| {
                let mut changed = false;
                for &(_, b, i) in data_phis {
                    let phi = &p.blocks[b].phis[i];
                    for (n, &q) in p.blocks[b].preds.iter().enumerate() {
                        if self.tail[q].is_none() {
                            continue;
                        }
                        let Some(list) = taint.get(&phi.args[n]).cloned() else {
                            continue;
                        };
                        let ec = if precise {
                            bools.edge(st, q, b)
                        } else {
                            Bool::TRUE
                        };
                        for (why, c) in list {
                            let tc = if precise {
                                let mut tc = bools.and(c, ec);
                                if self.rpo[b] <= self.rpo[q] {
                                    tc = bools.forget(tc, self.rpo[b]);
                                }
                                tc
                            } else {
                                Bool::TRUE
                            };
                            changed |= merge(taint, phi.value, why, tc, bools);
                        }
                    }
                }
                for &w in &self.deps_order {
                    for info in &self.deps[&w] {
                        let use_c = if precise {
                            site_cond(bools, info.block, info.guard, info.invert_guard)
                        } else {
                            Bool::TRUE
                        };
                        for &dep in &info.deps {
                            let (val, dc) = if precise {
                                dep_cond(bools, use_c, dep)
                            } else {
                                let u = match dep {
                                    ReadDep::Value(u) | ReadDep::Edge { value: u, .. } => u,
                                };
                                (u, Bool::TRUE)
                            };
                            let Some(list) = taint.get(&val).cloned() else {
                                continue;
                            };
                            for (why, c) in list {
                                if info.ignore_why == Some(why) {
                                    continue;
                                }
                                let tc = if precise {
                                    bools.and(c, dc)
                                } else {
                                    Bool::TRUE
                                };
                                changed |= merge(taint, w, why, tc, bools);
                            }
                        }
                    }
                }
                changed
            };

        let mut settled = false;
        for _ in 0..MAX_TAINT_ROUNDS {
            if !propagate_step(&mut taint, &mut bools, true) {
                settled = true;
                break;
            }
        }
        if !settled {
            // Conservative widening when cyclic propagation exceeds the round cap.
            for list in taint.values_mut() {
                for (_, c) in list.iter_mut() {
                    *c = Bool::TRUE;
                }
            }
            while propagate_step(&mut taint, &mut bools, false) {}
        }

        for sink in &self.sinks {
            let sink_c = site_cond(&mut bools, sink.block, sink.guard, false);
            for &dep in &sink.deps {
                let (val, dc) = dep_cond(&mut bools, sink_c, dep);
                if let Some(list) = taint.get(&val) {
                    for &(why, c) in list {
                        if bools.sat(c, dc) {
                            return Err(unsupported(sink.offset, why));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// Whether DXIL interpolation mode `mode` is a linear (noperspective) one.
fn linear(mode: u8) -> bool {
    matches!(mode, 4 | 5 | 7)
}

/// The DXIL interpolation mode of a kind of barycentrics (none for the
/// pull model).
fn interp_mode(kind: Barycentric) -> Option<u8> {
    Some(match kind {
        Barycentric::PerspectiveCenter => 2,
        Barycentric::PerspectiveCentroid => 3,
        Barycentric::PerspectiveSample => 6,
        Barycentric::LinearCenter => 4,
        Barycentric::LinearCentroid => 5,
        Barycentric::LinearSample => 7,
        Barycentric::PerspectivePullModel => return None,
    })
}

#[cfg(test)]
mod tests {
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
                (13, 0x1_0000), // s_buffer_load_dword on read-only CBV @ 2
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
}
