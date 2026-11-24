//! Wave uniformity through phis: a loop counter or a value joined by a
//! uniform branch holds the same value in every lane that runs, so an
//! address computed from it and read back with `v_readfirstlane_b32` is the
//! address of every lane (the console compiler's table loops).
//!
//! A phi is uniform when all its arguments are, and no branch whose lanes
//! may part ways can send lanes to it along different edges: not a join of
//! the two arms of such a branch, nor a block of a loop it can leave or
//! stay in lane by lane. The set of uniform phis is the greatest fixed
//! point (a loop counter's own increment feeds its exit test).

use super::{MAX_ROUNDS, Tracer};
use crate::ir::{EXEC, Exit, Inst, Reg, VCC, Value, reverse_postorder};

/// Operations that combine lane masks bit by bit (or copy one).
const MASK_OPS: &[&str] = &[
    "s_and_b32",
    "s_or_b32",
    "s_xor_b32",
    "s_andn2_b32",
    "s_orn2_b32",
    "s_nand_b32",
    "s_nor_b32",
    "s_xnor_b32",
    "s_and_b64",
    "s_or_b64",
    "s_xor_b64",
    "s_andn2_b64",
    "s_orn2_b64",
    "s_mov_b32",
    "s_mov_b64",
    "s_not_b32",
    "s_not_b64",
];

/// Per block: the blocks a lane-divergent branch there taints.
pub(super) type Regions = Vec<Option<Vec<usize>>>;

fn is_exec(reg: Reg) -> bool {
    matches!(reg, Reg::S(c) if c == EXEC || c == EXEC + 1)
}

impl Tracer<'_> {
    /// Whether the phi `v` is the same in every lane that reads it.
    pub(super) fn phi_uniform(&self, v: Value) -> bool {
        if self.lanes.is_none() {
            return false;
        }
        if self.phi_ok.borrow().is_none() {
            self.settle_phis();
        }
        self.phi_ok
            .borrow()
            .as_ref()
            .is_some_and(|ok| ok[v.0 as usize])
    }

    /// Greatest fixed point: every phi uniform until an argument or a
    /// divergent branch says otherwise.
    fn settle_phis(&self) {
        let mut ok: Vec<bool> = self.phis.iter().map(Option::is_some).collect();
        *self.phi_ok.borrow_mut() = Some(ok.clone());
        let regions = self.regions();
        let mut settled = false;
        for _ in 0..MAX_ROUNDS {
            self.uniform_memo.borrow_mut().fill(None);
            let mut tainted = vec![false; self.p.blocks.len()];
            for (b, region) in regions.iter().enumerate() {
                if let Some(region) = region
                    && !self.branch_uniform(b)
                {
                    for &x in region {
                        tainted[x] = true;
                    }
                }
            }
            let mut changed = false;
            for (b, block) in self.p.blocks.iter().enumerate() {
                for phi in &block.phis {
                    let i = phi.value.0 as usize;
                    if ok[i] && (tainted[b] || !phi.args.iter().all(|&a| self.uniform(a))) {
                        ok[i] = false;
                        changed = true;
                    }
                }
            }
            *self.phi_ok.borrow_mut() = Some(ok.clone());
            if !changed {
                settled = true;
                break;
            }
        }
        if !settled {
            *self.phi_ok.borrow_mut() = Some(vec![false; ok.len()]);
        }
        self.uniform_memo.borrow_mut().fill(None);
    }

    /// Whether every lane that reaches the closing branch of block `b`
    /// takes it the same way.
    fn branch_uniform(&self, b: usize) -> bool {
        let Some(inst) = self.p.blocks[b].insts.last() else {
            return false;
        };
        match inst.name() {
            "s_cbranch_execz" | "s_cbranch_execnz" => true,
            "s_cbranch_scc0" | "s_cbranch_scc1" => {
                inst.read(Reg::Scc).is_some_and(|v| self.uniform(v))
            }
            "s_cbranch_vccz" | "s_cbranch_vccnz" => {
                let mut vcc = inst
                    .reads
                    .iter()
                    .filter(|(r, _)| matches!(r, Reg::S(c) if (VCC..VCC + 2).contains(c)))
                    .peekable();
                vcc.peek().is_some() && vcc.all(|&(_, v)| self.mask_uniform(v))
            }
            _ => false,
        }
    }

    /// Whether a lane mask holds the same bit in every lane it covers: a
    /// comparison of uniform values, combined with EXEC and other such masks.
    fn mask_uniform(&self, v: Value) -> bool {
        let Some(inst) = self.def(v) else {
            return false;
        };
        let name = inst.name();
        if name.starts_with("v_cmp") && !name.starts_with("v_cmpx") {
            return self.valu_reads_uniform(inst, false);
        }
        MASK_OPS.contains(&name)
            && inst.reads.iter().all(|&(reg, x)| {
                is_exec(reg) || (reg != Reg::Scc && (self.mask_uniform(x) || self.uniform(x)))
            })
    }

    /// Whether every register a VALU instruction reads is uniform, but EXEC
    /// and, unless `guarded`, the old value of the VGPRs it writes (which
    /// only the lanes it skips keep).
    pub(super) fn valu_reads_uniform(&self, inst: &Inst, guarded: bool) -> bool {
        inst.reads.iter().all(|&(reg, x)| {
            is_exec(reg)
                || (!guarded && matches!(reg, Reg::V(_)) && inst.writes.iter().any(|w| w.0 == reg))
                || self.uniform(x)
        })
    }

    /// Whether lanes run the instruction defining `v` under a condition
    /// beyond reaching it.
    pub(super) fn guarded(&self, v: Value) -> bool {
        match (self.lanes, self.defs[v.0 as usize]) {
            (Some(st), Some((b, i))) => st.guards[b][i].is_some(),
            _ => true,
        }
    }

    /// The blocks tainted by the branch closing each block, were it divergent.
    fn regions(&self) -> &Regions {
        self.regions.get_or_init(|| {
            let blocks = &self.p.blocks;
            let n = blocks.len();
            let mut fwd = vec![false; n];
            for b in reverse_postorder(blocks) {
                fwd[b] = true;
            }
            let succs: Vec<Vec<usize>> = blocks
                .iter()
                .map(|b| match b.exit {
                    Exit::Jump(t) => vec![t],
                    Exit::Branch { taken, not_taken } => vec![taken, not_taken],
                    Exit::End => vec![n],
                    Exit::Indirect => Vec::new(),
                })
                .collect();
            let (_, ipdom) = compute_postdoms(n, &succs, &fwd);
            let walk = |from: usize, stop: usize, next: &dyn Fn(usize) -> Vec<usize>| {
                let mut seen = vec![false; n];
                let mut stack = vec![from];
                while let Some(x) = stack.pop() {
                    if x >= n || seen[x] {
                        continue;
                    }
                    seen[x] = true;
                    if x != stop {
                        stack.extend(next(x));
                    }
                }
                seen
            };
            let forward = |x: usize| succs[x].clone();
            let backward = |x: usize| blocks[x].preds.clone();
            (0..n)
                .map(|b| {
                    let Exit::Branch { taken, not_taken } = blocks[b].exit else {
                        return None;
                    };
                    if !fwd[b] || taken == not_taken {
                        return None;
                    }
                    let stop = ipdom[b];
                    let (r1, r2) = (walk(taken, stop, &forward), walk(not_taken, stop, &forward));
                    // Its loop: the blocks it reaches that reach it back.
                    let (d1, d2) = (
                        walk(taken, usize::MAX, &forward),
                        walk(not_taken, usize::MAX, &forward),
                    );
                    let up = walk(b, usize::MAX, &backward);
                    Some(
                        (0..n)
                            .filter(|&x| (r1[x] && r2[x]) || ((d1[x] || d2[x]) && up[x]))
                            .collect(),
                    )
                })
                .collect()
        })
    }
}

/// Immediate post-dominator per block (or virtual exit `n`) using Cooper-Harvey-Kennedy.
/// Unreachable or never-exiting blocks have `rev_idx == usize::MAX`.
fn compute_postdoms(n: usize, succs: &[Vec<usize>], fwd: &[bool]) -> (Vec<usize>, Vec<usize>) {
    let mut rev_succs = vec![Vec::new(); n + 1];
    for b in 0..n {
        if !fwd[b] {
            continue;
        }
        for &s in &succs[b] {
            rev_succs[s].push(b);
        }
    }
    let mut post = Vec::with_capacity(n + 1);
    let mut seen = vec![false; n + 1];
    let mut stack = vec![(n, 0usize)];
    seen[n] = true;
    while let Some((node, idx)) = stack.last_mut() {
        let u = *node;
        if *idx < rev_succs[u].len() {
            let v = rev_succs[u][*idx];
            *idx += 1;
            if !seen[v] {
                seen[v] = true;
                stack.push((v, 0));
            }
        } else {
            post.push(u);
            stack.pop();
        }
    }
    post.reverse();
    let mut rev_idx = vec![usize::MAX; n + 1];
    for (idx, &b) in post.iter().enumerate() {
        rev_idx[b] = idx;
    }
    let mut ipdom = vec![usize::MAX; n + 1];
    ipdom[n] = n;
    let intersect = |mut a: usize, mut b: usize, ipdom: &[usize]| -> usize {
        while a != b {
            while rev_idx[a] > rev_idx[b] {
                a = ipdom[a];
            }
            while rev_idx[b] > rev_idx[a] {
                b = ipdom[b];
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in &post[1..] {
            let mut new = usize::MAX;
            for &s in &succs[b] {
                if ipdom[s] != usize::MAX {
                    new = if new == usize::MAX {
                        s
                    } else {
                        intersect(s, new, &ipdom)
                    };
                }
            }
            if new != usize::MAX && ipdom[b] != new {
                ipdom[b] = new;
                changed = true;
            }
        }
    }
    (rev_idx, ipdom)
}
