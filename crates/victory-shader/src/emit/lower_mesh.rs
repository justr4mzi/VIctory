/// Bitmask of the vertex / primitive count fields in `M0` for `GS_ALLOC_REQ`.
const GS_ALLOC_COUNT_MASK: u32 = 0x3FF;
/// Bit shift of the primitive count field in `M0` for `GS_ALLOC_REQ`.
const GS_ALLOC_PRIM_SHIFT: u32 = 12;
/// Maximum vertex / primitive count D3D12 / DXIL allows for a mesh shader.
const MAX_HOST_MESH_COUNT: u32 = 256;

fn is_gs_alloc_req(inst: &Inst) -> bool {
    matches!(inst.name(), "s_sendmsg" | "s_sendmsghalt")
        && matches!(inst.fields, Fields::Sopp { imm } if imm & 0xF == GS_ALLOC_REQ)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScalarBound {
    /// Bits guaranteed to be 1.
    ones: u32,
    /// Bits guaranteed to be 0.
    zeros: u32,
    /// Conservative unsigned minimum (`val >= umin`).
    umin: u32,
    /// Conservative unsigned maximum (`val <= umax`).
    umax: u32,
}

impl ScalarBound {
    const TOP: Self = Self {
        ones: 0,
        zeros: 0,
        umin: 0,
        umax: u32::MAX,
    };

    fn constant(k: u32) -> Self {
        Self {
            ones: k,
            zeros: !k,
            umin: k,
            umax: k,
        }
    }

    fn normalize(mut self) -> Self {
        self.umax = self.umax.min(!self.zeros);
        self.umin = self.umin.max(self.ones).min(self.umax);
        let top_mask = if self.umax == 0 {
            0
        } else {
            u32::MAX >> self.umax.leading_zeros()
        };
        self.zeros |= !top_mask;
        if self.umin == self.umax {
            return Self::constant(self.umin);
        }
        self
    }

    fn as_const(self) -> Option<u32> {
        (self.umin == self.umax).then_some(self.umin)
    }

    fn possible_ones(self) -> u32 {
        !self.zeros
    }

    fn join(self, other: Self) -> Self {
        Self {
            ones: self.ones & other.ones,
            zeros: self.zeros & other.zeros,
            umin: self.umin.min(other.umin),
            umax: self.umax.max(other.umax),
        }
        .normalize()
    }
}

fn sop_operand_bound(
    inst: &Inst,
    op: &Operand,
    bounds: &HashMap<Value, ScalarBound>,
) -> ScalarBound {
    match *op {
        Operand::Scalar { code: NULL, .. } => ScalarBound::constant(0),
        Operand::Scalar { code, .. } => inst
            .read(Reg::S(code))
            .and_then(|v| bounds.get(&v).copied())
            .unwrap_or(ScalarBound::TOP),
        Operand::Const(c) => crate::resource::inline(c)
            .map(ScalarBound::constant)
            .unwrap_or(ScalarBound::TOP),
        Operand::Literal(k) => ScalarBound::constant(k),
        _ => ScalarBound::TOP,
    }
}

fn bound_and(a: ScalarBound, b: ScalarBound) -> ScalarBound {
    if let (Some(x), Some(y)) = (a.as_const(), b.as_const()) {
        return ScalarBound::constant(x & y);
    }
    ScalarBound {
        ones: a.ones & b.ones,
        zeros: a.zeros | b.zeros,
        umin: a.ones & b.ones,
        umax: a.umax.min(b.umax),
    }
    .normalize()
}

fn bound_or(a: ScalarBound, b: ScalarBound) -> ScalarBound {
    if let (Some(x), Some(y)) = (a.as_const(), b.as_const()) {
        return ScalarBound::constant(x | y);
    }
    ScalarBound {
        ones: a.ones | b.ones,
        zeros: a.zeros & b.zeros,
        umin: a.umin.max(b.umin),
        umax: a.umax.saturating_add(b.umax),
    }
    .normalize()
}

fn bound_xor(a: ScalarBound, b: ScalarBound) -> ScalarBound {
    if let (Some(x), Some(y)) = (a.as_const(), b.as_const()) {
        return ScalarBound::constant(x ^ y);
    }
    ScalarBound {
        ones: (a.ones & b.zeros) | (a.zeros & b.ones),
        zeros: (a.zeros & b.zeros) | (a.ones & b.ones),
        umin: 0,
        umax: a.umax.saturating_add(b.umax),
    }
    .normalize()
}

fn bound_lshl(a: ScalarBound, b: ScalarBound) -> ScalarBound {
    let Some(sh) = b.as_const() else {
        return ScalarBound::TOP;
    };
    let s = sh & 31;
    if s == 0 {
        return a;
    }
    let ones = a.ones << s;
    let zeros = (a.zeros << s) | ((1u32 << s) - 1);
    let (umin, umax) = if a.umax <= (u32::MAX >> s) {
        (a.umin << s, a.umax << s)
    } else {
        (0, u32::MAX)
    };
    ScalarBound {
        ones,
        zeros,
        umin,
        umax,
    }
    .normalize()
}

fn bound_lshr(a: ScalarBound, b: ScalarBound) -> ScalarBound {
    if let Some(sh) = b.as_const() {
        let s = sh & 31;
        if s == 0 {
            return a;
        }
        return ScalarBound {
            ones: a.ones >> s,
            zeros: (a.zeros >> s) | (!0u32 << (32 - s)),
            umin: a.umin >> s,
            umax: a.umax >> s,
        }
        .normalize();
    }
    if b.umax < 32 {
        return ScalarBound {
            ones: 0,
            zeros: if b.umin > 0 {
                !0u32 << (32 - b.umin)
            } else {
                0
            },
            umin: a.umin >> b.umax,
            umax: a.umax >> b.umin,
        }
        .normalize();
    }
    ScalarBound::TOP
}

fn bound_bfe_u32(a: ScalarBound, b: ScalarBound) -> ScalarBound {
    let Some(k) = b.as_const() else {
        return ScalarBound::TOP;
    };
    let off = k & 31;
    let w = (k >> 16) & 0x7F;
    if w == 0 {
        return ScalarBound::constant(0);
    }
    let shifted = bound_lshr(a, ScalarBound::constant(off));
    if w >= 32 {
        shifted
    } else {
        bound_and(shifted, ScalarBound::constant((1u32 << w) - 1))
    }
}

fn bound_add(a: ScalarBound, b: ScalarBound) -> ScalarBound {
    if let (Some(x), Some(y)) = (a.as_const(), b.as_const()) {
        return ScalarBound::constant(x.wrapping_add(y));
    }
    if let Some(umax) = a.umax.checked_add(b.umax) {
        return ScalarBound {
            ones: 0,
            zeros: 0,
            umin: a.umin.saturating_add(b.umin),
            umax,
        }
        .normalize();
    }
    ScalarBound::TOP
}

fn bound_sub(a: ScalarBound, b: ScalarBound) -> ScalarBound {
    if let (Some(x), Some(y)) = (a.as_const(), b.as_const()) {
        return ScalarBound::constant(x.wrapping_sub(y));
    }
    if a.umin >= b.umax {
        return ScalarBound {
            ones: 0,
            zeros: 0,
            umin: a.umin - b.umax,
            umax: a.umax - b.umin,
        }
        .normalize();
    }
    ScalarBound::TOP
}

fn eval_sop_bound(inst: &Inst, bounds: &HashMap<Value, ScalarBound>) -> ScalarBound {
    let Fields::Sop { dst, src, imm } = &inst.fields else {
        return ScalarBound::TOP;
    };
    let name = inst.name();
    let op = |k: usize| {
        src.get(k)
            .map_or(ScalarBound::TOP, |o| sop_operand_bound(inst, o, bounds))
    };
    let old_dst = || {
        dst.as_ref()
            .map_or(ScalarBound::TOP, |o| sop_operand_bound(inst, o, bounds))
    };
    let sext16 = || ScalarBound::constant(i32::from(imm.unwrap_or(0) as i16) as u32);
    match name {
        "s_mov_b32" => op(0),
        "s_movk_i32" => sext16(),
        "s_cselect_b32" => op(0).join(op(1)),
        "s_cmov_b32" => old_dst().join(op(0)),
        "s_cmovk_i32" => old_dst().join(sext16()),
        "s_not_b32" => {
            let a = op(0);
            ScalarBound {
                ones: a.zeros,
                zeros: a.ones,
                umin: !a.umax,
                umax: !a.umin,
            }
            .normalize()
        }
        "s_and_b32" => bound_and(op(0), op(1)),
        "s_or_b32" => bound_or(op(0), op(1)),
        "s_xor_b32" => bound_xor(op(0), op(1)),
        "s_lshl_b32" => bound_lshl(op(0), op(1)),
        "s_lshr_b32" => bound_lshr(op(0), op(1)),
        "s_bfe_u32" => bound_bfe_u32(op(0), op(1)),
        "s_min_u32" => {
            let (a, b) = (op(0), op(1));
            ScalarBound {
                ones: a.ones & b.ones,
                zeros: a.zeros & b.zeros,
                umin: a.umin.min(b.umin),
                umax: a.umax.min(b.umax),
            }
            .normalize()
        }
        "s_max_u32" => {
            let (a, b) = (op(0), op(1));
            ScalarBound {
                ones: a.ones & b.ones,
                zeros: a.zeros & b.zeros,
                umin: a.umin.max(b.umin),
                umax: a.umax.max(b.umax),
            }
            .normalize()
        }
        "s_add_u32" | "s_add_i32" => bound_add(op(0), op(1)),
        "s_addk_i32" => bound_add(old_dst(), sext16()),
        "s_sub_u32" | "s_sub_i32" => bound_sub(op(0), op(1)),
        "s_lshl1_add_u32" | "s_lshl2_add_u32" | "s_lshl3_add_u32" | "s_lshl4_add_u32" => {
            let k = u32::from(name.as_bytes()[6] - b'0');
            bound_add(bound_lshl(op(0), ScalarBound::constant(k)), op(1))
        }
        _ => ScalarBound::TOP,
    }
}

#[derive(Clone, Copy)]
enum ScalarDef<'a> {
    Phi(&'a crate::ir::Phi),
    Inst(&'a Inst),
}

/// Computes a conservative `(umin, umax)` range for `(val(v) >> shift) & mask`
/// where `mask` is a contiguous low bitmask `(1 << w) - 1`.
fn slice_range_val(
    v: Value,
    shift: u32,
    mask: u32,
    bounds: &HashMap<Value, ScalarBound>,
    defs: &HashMap<Value, ScalarDef<'_>>,
    visiting: &mut HashSet<Value>,
) -> (u32, u32) {
    if mask == 0 || shift >= 32 {
        return (0, 0);
    }
    let b = bounds.get(&v).copied().unwrap_or(ScalarBound::TOP);
    let mut lo = (b.ones >> shift) & mask;
    let mut hi = (b.possible_ones() >> shift) & mask;
    if (b.umax >> shift) <= mask {
        hi = hi.min(b.umax >> shift);
        lo = lo.max(b.umin >> shift);
    }
    if lo == hi || !visiting.insert(v) {
        return (lo, hi);
    }
    let slice_op = |inst: &Inst, op: &Operand, sh: u32, m: u32, vis: &mut HashSet<Value>| {
        if m == 0 || sh >= 32 {
            return (0, 0);
        }
        match *op {
            Operand::Scalar { code: NULL, .. } => (0, 0),
            Operand::Scalar { code, .. } => match inst.read(Reg::S(code)) {
                Some(rv) => slice_range_val(rv, sh, m, bounds, defs, vis),
                None => (0, m),
            },
            Operand::Const(c) => match crate::resource::inline(c) {
                Some(k) => {
                    let x = (k >> sh) & m;
                    (x, x)
                }
                None => (0, m),
            },
            Operand::Literal(k) => {
                let x = (k >> sh) & m;
                (x, x)
            }
            _ => (0, m),
        }
    };
    let refined = match defs.get(&v).copied() {
        Some(ScalarDef::Phi(phi)) => {
            let mut plo = u32::MAX;
            let mut phi_hi = 0u32;
            for &arg in &phi.args {
                let (alo, ahi) = slice_range_val(arg, shift, mask, bounds, defs, visiting);
                plo = plo.min(alo);
                phi_hi = phi_hi.max(ahi);
            }
            if phi.args.is_empty() {
                (lo, hi)
            } else {
                (plo, phi_hi)
            }
        }
        Some(ScalarDef::Inst(inst)) => {
            if let Fields::Sop { dst, src, imm } = &inst.fields {
                let name = inst.name();
                match name {
                    "s_mov_b32" => slice_op(inst, &src[0], shift, mask, visiting),
                    "s_cselect_b32" => {
                        let (alo, ahi) = slice_op(inst, &src[0], shift, mask, visiting);
                        let (blo, bhi) = slice_op(inst, &src[1], shift, mask, visiting);
                        (alo.min(blo), ahi.max(bhi))
                    }
                    "s_cmov_b32" => {
                        let (alo, ahi) = dst
                            .as_ref()
                            .map_or((0, mask), |d| slice_op(inst, d, shift, mask, visiting));
                        let (blo, bhi) = slice_op(inst, &src[0], shift, mask, visiting);
                        (alo.min(blo), ahi.max(bhi))
                    }
                    "s_cmovk_i32" => {
                        let (alo, ahi) = dst
                            .as_ref()
                            .map_or((0, mask), |d| slice_op(inst, d, shift, mask, visiting));
                        let k = ((i32::from(imm.unwrap_or(0) as i16) as u32) >> shift) & mask;
                        (alo.min(k), ahi.max(k))
                    }
                    "s_or_b32" => {
                        let (alo, ahi) = slice_op(inst, &src[0], shift, mask, visiting);
                        let (blo, bhi) = slice_op(inst, &src[1], shift, mask, visiting);
                        if blo == 0 && bhi == 0 {
                            (alo, ahi)
                        } else if alo == 0 && ahi == 0 {
                            (blo, bhi)
                        } else {
                            (alo.max(blo), ahi.saturating_add(bhi).min(mask))
                        }
                    }
                    "s_add_u32" | "s_add_i32" if shift == 0 => {
                        let (alo, ahi) = slice_op(inst, &src[0], 0, mask, visiting);
                        let (blo, bhi) = slice_op(inst, &src[1], 0, mask, visiting);
                        if blo == 0 && bhi == 0 {
                            (alo, ahi)
                        } else if alo == 0 && ahi == 0 {
                            (blo, bhi)
                        } else {
                            (0, ahi.saturating_add(bhi).min(mask))
                        }
                    }
                    "s_addk_i32" if shift == 0 => {
                        let (alo, ahi) = dst
                            .as_ref()
                            .map_or((0, mask), |d| slice_op(inst, d, 0, mask, visiting));
                        let k = (i32::from(imm.unwrap_or(0) as i16) as u32) & mask;
                        if k == 0 {
                            (alo, ahi)
                        } else {
                            (0, ahi.saturating_add(k).min(mask))
                        }
                    }
                    "s_and_b32" => {
                        let (alo, ahi) = slice_op(inst, &src[0], shift, mask, visiting);
                        let (blo, bhi) = slice_op(inst, &src[1], shift, mask, visiting);
                        if blo == mask && bhi == mask {
                            (alo, ahi)
                        } else if alo == mask && ahi == mask {
                            (blo, bhi)
                        } else if alo == ahi && blo == bhi {
                            (alo & blo, alo & blo)
                        } else {
                            (0, ahi.min(bhi))
                        }
                    }
                    "s_lshl_b32" => {
                        if let Some(sh) = sop_operand_bound(inst, &src[1], bounds).as_const() {
                            let s = sh & 31;
                            if s > shift {
                                let d = s - shift;
                                if d >= 32 || (mask >> d) == 0 {
                                    (0, 0)
                                } else {
                                    let (rlo, rhi) =
                                        slice_op(inst, &src[0], 0, mask >> d, visiting);
                                    (rlo << d, rhi << d)
                                }
                            } else {
                                let valid = mask & (u32::MAX >> shift);
                                slice_op(inst, &src[0], shift - s, valid, visiting)
                            }
                        } else {
                            (lo, hi)
                        }
                    }
                    "s_lshr_b32" => {
                        if let Some(sh) = sop_operand_bound(inst, &src[1], bounds).as_const() {
                            let s = sh & 31;
                            if shift + s >= 32 {
                                (0, 0)
                            } else {
                                slice_op(inst, &src[0], shift + s, mask, visiting)
                            }
                        } else {
                            (lo, hi)
                        }
                    }
                    "s_bfe_u32" => {
                        if let Some(k) = sop_operand_bound(inst, &src[1], bounds).as_const() {
                            let off = k & 31;
                            let w = (k >> 16) & 0x7F;
                            if w == 0 || shift >= w || off + shift >= 32 {
                                (0, 0)
                            } else {
                                let rem = w - shift;
                                let w_mask = if rem >= 32 {
                                    u32::MAX
                                } else {
                                    (1u32 << rem) - 1
                                };
                                slice_op(inst, &src[0], off + shift, mask & w_mask, visiting)
                            }
                        } else {
                            (lo, hi)
                        }
                    }
                    _ => (lo, hi),
                }
            } else {
                (lo, hi)
            }
        }
        None => (lo, hi),
    };
    visiting.remove(&v);
    (lo.max(refined.0), hi.min(refined.1))
}

#[derive(Debug, Clone, Copy)]
enum MeshCountsMode {
    /// Emit constant counts in `mesh_prologue` (`self.s.entry`).
    EntryConst(u32, u32),
    /// Emit dynamic `M0` counts at the single `GS_ALLOC_REQ` instruction in block 0.
    AtSite {
        block: usize,
        inst: usize,
        v_const: Option<u32>,
        p_const: Option<u32>,
    },
    /// Emit dynamic `M0` counts at the end of block 0 (before its terminator).
    AtBlock0End {
        m0: Value,
        v_const: Option<u32>,
        p_const: Option<u32>,
    },
}

#[derive(Debug, Clone, Copy)]
enum MeshExportLimit {
    Const(u32),
    Dynamic(V),
}

struct MeshAllocPlan {
    max_vertices: u32,
    max_primitives: u32,
    counts: MeshCountsMode,
}

fn is_pure_32bit_sop(inst: &Inst) -> bool {
    let Fields::Sop { dst, .. } = &inst.fields else {
        return false;
    };
    if let Some(Operand::Scalar { dwords, .. }) = dst
        && *dwords != 1
    {
        return false;
    }
    let name = inst.name();
    matches!(
        name,
        "s_mov_b32"
            | "s_movk_i32"
            | "s_cselect_b32"
            | "s_cmov_b32"
            | "s_cmovk_i32"
            | "s_not_b32"
            | "s_add_u32"
            | "s_add_i32"
            | "s_addk_i32"
            | "s_sub_u32"
            | "s_sub_i32"
            | "s_mul_i32"
            | "s_mulk_i32"
            | "s_bfe_u32"
            | "s_bfe_i32"
            | "s_and_b32"
            | "s_or_b32"
            | "s_xor_b32"
            | "s_andn2_b32"
            | "s_orn2_b32"
            | "s_nand_b32"
            | "s_nor_b32"
            | "s_xnor_b32"
            | "s_lshl_b32"
            | "s_lshr_b32"
            | "s_ashr_i32"
            | "s_lshl1_add_u32"
            | "s_lshl2_add_u32"
            | "s_lshl3_add_u32"
            | "s_lshl4_add_u32"
            | "s_min_u32"
            | "s_max_u32"
            | "s_min_i32"
            | "s_max_i32"
            | "s_abs_i32"
            | "s_bitset0_b32"
            | "s_bitset1_b32"
            | "s_bfm_b32"
    ) || name.starts_with("s_cmp_")
        || name.starts_with("s_cmpk_")
        || matches!(name, "s_bitcmp0_b32" | "s_bitcmp1_b32")
}

struct MaterializeCtx<'b, 'a> {
    st: &'b Structured,
    def_block: &'b HashMap<Value, usize>,
    defs: &'b HashMap<Value, ScalarDef<'a>>,
    inputs: &'b HashSet<Value>,
}

/// Checks whether `v` can be materialized at the end of block 0 using only
/// values already defined in block 0 / entry inputs and pure 32-bit SOP ops.
fn collect_block0_materialization<'a>(
    ctx: &MaterializeCtx<'_, 'a>,
    v: Value,
    visited: &mut HashSet<Value>,
    out: &mut Vec<&'a Inst>,
) -> bool {
    if ctx.inputs.contains(&v) || ctx.def_block.get(&v) == Some(&0) {
        return true;
    }
    if !visited.insert(v) {
        return true;
    }
    if ctx.st.masks.contains_key(&v) || ctx.st.highs.contains_key(&v) {
        return false;
    }
    let Some(ScalarDef::Inst(inst)) = ctx.defs.get(&v).copied() else {
        return false;
    };
    if !is_pure_32bit_sop(inst) {
        return false;
    }
    for &(reg, rv) in &inst.reads {
        if matches!(reg, Reg::S(c) if c == EXEC || c == EXEC + 1) {
            return false;
        }
        if !collect_block0_materialization(ctx, rv, visited, out) {
            return false;
        }
    }
    out.push(inst);
    true
}

fn analyze_mesh_alloc(
    p: &Program,
    st: &Structured,
    ms: &crate::native::MeshIo,
    order: &[usize],
    def_block: &HashMap<Value, usize>,
) -> MeshAllocPlan {
    let mut bounds: HashMap<Value, ScalarBound> = p
        .inputs
        .iter()
        .map(|&(_, v)| (v, ScalarBound::TOP))
        .collect();
    let mut defs: HashMap<Value, ScalarDef<'_>> = HashMap::new();
    for &b in order {
        for phi in &p.blocks[b].phis {
            defs.insert(phi.value, ScalarDef::Phi(phi));
        }
        for inst in &p.blocks[b].insts {
            for &(_, w) in &inst.writes {
                defs.insert(w, ScalarDef::Inst(inst));
            }
        }
    }

    let mut phi_updates: HashMap<Value, u32> = HashMap::new();
    loop {
        let mut changed = false;
        for &b in order {
            for phi in &p.blocks[b].phis {
                let mut joined: Option<ScalarBound> = None;
                for &arg in &phi.args {
                    if let Some(&ab) = bounds.get(&arg) {
                        joined = Some(match joined {
                            Some(cur) => cur.join(ab),
                            None => ab,
                        });
                    }
                }
                let next = joined.unwrap_or(ScalarBound::TOP);
                let prev = bounds.get(&phi.value).copied();
                if prev != Some(next) {
                    let count = phi_updates.entry(phi.value).or_insert(0);
                    *count += 1;
                    let val = if *count > 4 { ScalarBound::TOP } else { next };
                    if prev != Some(val) {
                        bounds.insert(phi.value, val);
                        changed = true;
                    }
                }
            }
            for inst in &p.blocks[b].insts {
                if let Fields::Sop {
                    dst: Some(Operand::Scalar { code, dwords: 1 }),
                    ..
                } = &inst.fields
                    && *code != NULL
                    && let Some(w) = inst.write(Reg::S(*code))
                {
                    let bnd = eval_sop_bound(inst, &bounds);
                    if bounds.get(&w).copied() != Some(bnd) {
                        bounds.insert(w, bnd);
                        changed = true;
                    }
                }
                for &(_, w) in &inst.writes {
                    bounds.entry(w).or_insert(ScalarBound::TOP);
                }
            }
        }
        if !changed {
            break;
        }
    }

    let mut sites = Vec::new();
    for &b in order {
        for (i, inst) in p.blocks[b].insts.iter().enumerate() {
            if is_gs_alloc_req(inst) {
                sites.push((b, i));
            }
        }
    }

    let mut proven = !sites.is_empty();
    let mut site_ranges = Vec::with_capacity(sites.len());
    for &(b, i) in &sites {
        let m0 = p.blocks[b].insts[i].read(Reg::S(crate::ir::M0));
        let Some(m0_val) = m0 else {
            proven = false;
            continue;
        };
        let vr = slice_range_val(
            m0_val,
            0,
            GS_ALLOC_COUNT_MASK,
            &bounds,
            &defs,
            &mut HashSet::new(),
        );
        let pr = slice_range_val(
            m0_val,
            GS_ALLOC_PRIM_SHIFT,
            GS_ALLOC_COUNT_MASK,
            &bounds,
            &defs,
            &mut HashSet::new(),
        );
        if vr.1 > MAX_HOST_MESH_COUNT || pr.1 > MAX_HOST_MESH_COUNT {
            proven = false;
        }
        site_ranges.push((m0_val, vr, pr));
    }

    let (max_vertices, max_primitives) = if proven {
        let max_v = site_ranges.iter().map(|r| r.1.1).max().unwrap_or(0);
        let max_p = site_ranges.iter().map(|r| r.2.1).max().unwrap_or(0);
        (
            max_v.min(ms.max_vertices).min(MAX_HOST_MESH_COUNT),
            max_p.min(ms.max_primitives).min(MAX_HOST_MESH_COUNT),
        )
    } else {
        (ms.max_vertices, ms.max_primitives)
    };

    let has_exp_before = |b: usize, limit: usize| {
        p.blocks[b].insts[..limit]
            .iter()
            .any(|inst| matches!(inst.fields, Fields::Exp { .. }))
    };

    let counts = if sites.len() == 1
        && let [(m0_val, vr, pr)] = &site_ranges[..]
    {
        let v_const = (vr.0 == vr.1)
            .then_some(vr.0.min(max_vertices))
            .or((vr.0 >= max_vertices).then_some(max_vertices));
        let p_const = (pr.0 == pr.1)
            .then_some(pr.0.min(max_primitives))
            .or((pr.0 >= max_primitives).then_some(max_primitives));
        if let (Some(vc), Some(pc)) = (v_const, p_const) {
            MeshCountsMode::EntryConst(vc, pc)
        } else {
            let (alloc_b, alloc_i) = sites[0];
            if alloc_b == 0 && !has_exp_before(0, alloc_i) {
                MeshCountsMode::AtSite {
                    block: 0,
                    inst: alloc_i,
                    v_const,
                    p_const,
                }
            } else if alloc_b != 0 && !has_exp_before(0, p.blocks[0].insts.len()) {
                let inputs: HashSet<Value> = p.inputs.iter().map(|&(_, v)| v).collect();
                let ctx = MaterializeCtx {
                    st,
                    def_block,
                    defs: &defs,
                    inputs: &inputs,
                };
                let mut plan = Vec::new();
                if collect_block0_materialization(&ctx, *m0_val, &mut HashSet::new(), &mut plan) {
                    MeshCountsMode::AtBlock0End {
                        m0: *m0_val,
                        v_const,
                        p_const,
                    }
                } else {
                    MeshCountsMode::EntryConst(max_vertices, max_primitives)
                }
            } else {
                MeshCountsMode::EntryConst(max_vertices, max_primitives)
            }
        }
    } else if proven
        && sites.len() == site_ranges.len()
        && !site_ranges.is_empty()
        && site_ranges
            .iter()
            .all(|&(_, vr, pr)| vr.0 == vr.1 && pr.0 == pr.1)
        && site_ranges
            .windows(2)
            .all(|w| w[0].1 == w[1].1 && w[0].2 == w[1].2)
    {
        let (_, vr, pr) = site_ranges[0];
        MeshCountsMode::EntryConst(vr.0.min(max_vertices), pr.0.min(max_primitives))
    } else {
        MeshCountsMode::EntryConst(max_vertices, max_primitives)
    };

    MeshAllocPlan {
        max_vertices,
        max_primitives,
        counts,
    }
}
