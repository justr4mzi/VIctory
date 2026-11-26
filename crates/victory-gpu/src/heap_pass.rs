//! Bookkeeping of the pass over the view heap's buffer views for shaders
//! that pick them by a computed index (`draw::State::heap_buffers`): which
//! slots it looks at again, and the span of each host copy their views
//! read. Kept apart from D3D12 so that it can be tested alone.
//!
//! The pass used to look again at buffer views of the heap and rebuild the
//! spans of host copies every pass. Now a slot is looked at again only when
//! - its buffer view was written or went (the heap's own set of changes);
//! - the view table entry its record names was freed (the title released
//!   the resource: `descriptor::forget`), which leaves the record as it was;
//! - its record in title memory is not the one seen at the last look (a
//!   record copied by the title itself, past the API); a compare of 12
//!   bytes per slot in slot order, no lock and no lookup;
//! - it found no host copy at its last look: every 16 passes, as before;
//! - every [`FULL_EVERY`] passes, all of them, as a safety net.
//!
//! The span of a copy is worked out again only when one of its slots went.
//!
//! Default ON outside unit tests (`VICTORY_HEAP_PASS_MARKED=0` disables, `=alt` alternates).

use std::collections::{BTreeMap, HashMap, HashSet};

/// Passes between two looks at every slot, besides those marked.
pub(crate) const FULL_EVERY: u64 = 256;
/// Passes between two looks at slots that found no host copy (and between
/// two looks at every slot with the old pass).
pub(crate) const ORPHANS_EVERY: u64 = 16;

/// What `VICTORY_HEAP_PASS_MARKED` says: 0 off, 1 on, 2 in turn.
fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_HEAP_PASS_MARKED").as_deref() {
        Ok("0") => 0,
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => u8::from(!cfg!(test)),
    })
}

/// Whether the switch is set at all (`1` or `alt`): the table entries freed
/// are then noted for the pass (`descriptor::forget`), in both phases of
/// `alt`.
pub(crate) fn marked_set() -> bool {
    mode() != 0
}

/// Set when `alt` turns the marked pass on again: what it kept is from
/// before the other pass's turn, and the other pass took the heap's changes
/// meanwhile. The marked pass then starts anew (every slot), and has the
/// other pass start anew at its next turn.
pub(crate) static RESUMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Set when `alt` turns the marked pass off: see [`marked_paused`].
static PAUSED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// With `alt` only: whether the marked pass's turn has just ended (true
/// once for each end), so that what it kept can be let go for the other
/// pass's turn. False, with nothing but the switch read, when the switch
/// is unset or `1`.
pub(crate) fn marked_paused() -> bool {
    mode() == 2 && PAUSED.swap(false, std::sync::atomic::Ordering::Relaxed)
}

/// Whether the pass over the heap's buffer views looks only at marked
/// slots now (`VICTORY_HEAP_PASS_MARKED`, default 1 outside tests; `0`
/// disables; `alt` alternates every 20s).
pub(crate) fn marked() -> bool {
    match mode() {
        0 => false,
        1 => true,
        _ => {
            use std::sync::atomic::Ordering::Relaxed;
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Relaxed) != u8::from(on) {
                if on {
                    RESUMED.store(true, Relaxed);
                } else {
                    PAUSED.store(true, Relaxed);
                }
                tracing::info!(
                    "the pass over the heap's buffer views looks only at marked slots from now (VICTORY_HEAP_PASS_MARKED=alt): {}",
                    if on { "ON" } else { "off, every slot every 16 passes as usual" }
                );
            }
            on
        }
    }
}

/// A slot's record in title memory: the tag and two words
/// (`descriptor::record_at`).
pub(crate) type Record = [u32; 3];

struct Slot {
    copy: usize,
    from: u64,
    to: u64,
    record: Record,
}

struct Copy<C> {
    copy: C,
    slots: HashSet<u32>,
    from: u64,
    to: u64,
    /// A slot went: the span is worked out again from those left.
    stale: bool,
}

/// What a pass looks at (see the module).
pub(crate) struct Picked<C> {
    passes: u64,
    /// In slot order: every pass reads the record of each of them from
    /// title memory ([`Picked::plan`]), and in the order of a hash table
    /// those reads fall all over the heap.
    slots: BTreeMap<u32, Slot>,
    copies: HashMap<usize, Copy<C>>,
    /// Slots that found no host copy, with the record seen. In slot order
    /// as `slots` is.
    orphans: BTreeMap<u32, Record>,
    /// View table entry -> slots whose record named it at their last look
    /// (some may name another by now: checked when used).
    by_entry: HashMap<u32, Vec<u32>>,
    /// Table entries freed since the last pass; `None`: not known.
    forgotten: Option<Vec<u32>>,
    /// The old pass's rules, kept for the tests.
    #[cfg_attr(not(test), allow(dead_code))]
    full: bool,
}

impl<C: Clone> Picked<C> {
    pub(crate) fn new(full: bool) -> Self {
        Picked {
            passes: 0,
            slots: BTreeMap::new(),
            copies: HashMap::new(),
            orphans: BTreeMap::new(),
            by_entry: HashMap::new(),
            forgotten: None,
            full,
        }
    }

    /// View table entries were freed (`None`: some were, which is not
    /// known): their slots are looked at next pass. Every heap's pass is
    /// told, though one of them asks.
    pub(crate) fn note_forgotten(&mut self, entries: Option<&[u32]>) {
        match (entries, &mut self.forgotten) {
            (Some(new), Some(kept)) => kept.extend_from_slice(new),
            _ => self.forgotten = None,
        }
    }

    /// The slots this pass looks at again, in order. `changed`: the slots
    /// whose buffer view was written or went since the last pass (and the
    /// entries [`Picked::note_forgotten`] was told of: if it was told it
    /// does not know, all slots are looked at); `all`: every slot that holds a
    /// buffer view; `record`: a slot's record now; `entry`: the table
    /// entry a record names, if it names one.
    pub(crate) fn plan(
        &mut self,
        changed: Vec<u32>,
        all: impl FnOnce() -> Vec<u32>,
        record: impl Fn(u32) -> Option<Record>,
        entry: impl Fn(Record) -> Option<u32>,
    ) -> Vec<u32> {
        let pass = self.passes;
        self.passes += 1;
        let forgotten = self.forgotten.replace(Vec::new());
        if self.full {
            if pass.is_multiple_of(ORPHANS_EVERY) {
                self.clear();
                return all();
            }
            return changed;
        }
        let Some(forgotten) = forgotten else {
            self.clear();
            return all();
        };
        if pass.is_multiple_of(FULL_EVERY) {
            self.clear();
            return all();
        }
        let mut due = changed;
        for e in forgotten {
            if let Some(slots) = self.by_entry.remove(&e) {
                // Only those whose record still names the entry.
                due.extend(
                    slots
                        .into_iter()
                        .filter(|s| self.record_of(*s).is_some_and(|r| entry(r) == Some(e))),
                );
            }
        }
        due.extend(
            self.slots
                .iter()
                .filter(|&(&slot, kept)| record(slot) != Some(kept.record))
                .map(|(&slot, _)| slot),
        );
        let orphans_too = pass.is_multiple_of(ORPHANS_EVERY);
        due.extend(
            self.orphans
                .iter()
                .filter(|&(&slot, &seen)| orphans_too || record(slot) != Some(seen))
                .map(|(&slot, _)| slot),
        );
        due.sort_unstable();
        due.dedup();
        due
    }

    /// The record seen at slot `slot`'s last look, if it is kept.
    pub(crate) fn record_of(&self, slot: u32) -> Option<Record> {
        self.slots
            .get(&slot)
            .map(|s| s.record)
            .or_else(|| self.orphans.get(&slot).copied())
    }

    fn clear(&mut self) {
        self.slots.clear();
        self.copies.clear();
        self.orphans.clear();
        self.by_entry.clear();
    }

    /// What slot `slot` held is let go (it is looked at now).
    pub(crate) fn forget(&mut self, slot: u32) {
        self.orphans.remove(&slot);
        let Some(old) = self.slots.remove(&slot) else {
            return;
        };
        if let Some(copy) = self.copies.get_mut(&old.copy) {
            copy.slots.remove(&slot);
            if copy.slots.is_empty() {
                self.copies.remove(&old.copy);
            } else if old.from <= copy.from || old.to >= copy.to {
                copy.stale = true;
            }
        }
    }

    /// Slot `slot` (with `record`, naming table entry `entry`) reads title
    /// memory `[from, to)` through host copy `copy` (`key`: its identity).
    pub(crate) fn keep(
        &mut self,
        slot: u32,
        key: usize,
        copy: &C,
        from: u64,
        to: u64,
        record: Record,
        entry: Option<u32>,
    ) {
        self.forget(slot);
        self.note_entry(slot, entry);
        self.slots.insert(
            slot,
            Slot {
                copy: key,
                from,
                to,
                record,
            },
        );
        let c = self.copies.entry(key).or_insert_with(|| Copy {
            copy: copy.clone(),
            slots: HashSet::new(),
            from,
            to,
            stale: false,
        });
        c.slots.insert(slot);
        c.from = c.from.min(from);
        c.to = c.to.max(to);
    }

    /// Slot `slot` holds a buffer view whose memory has no host copy yet.
    pub(crate) fn orphan(&mut self, slot: u32, record: Record, entry: Option<u32>) {
        self.forget(slot);
        self.note_entry(slot, entry);
        self.orphans.insert(slot, record);
    }

    fn note_entry(&mut self, slot: u32, entry: Option<u32>) {
        // Only kept with the incremental pass (the old one does not use it).
        if let Some(e) = entry
            && !self.full
        {
            // (Maybe twice: the pass's set of due slots takes it once.)
            self.by_entry.entry(e).or_default().push(slot);
        }
    }

    /// The views kept.
    pub(crate) fn views(&self) -> usize {
        self.slots.len()
    }

    /// Per host copy, the span its views read: `(copy, from, to)`.
    pub(crate) fn spans(&mut self) -> Vec<(C, u64, u64)> {
        if self.full {
            // As the old pass did: from every view, every pass.
            let mut spans: HashMap<usize, (u64, u64)> = HashMap::new();
            for s in self.slots.values() {
                let span = spans.entry(s.copy).or_insert((s.from, s.to));
                span.0 = span.0.min(s.from);
                span.1 = span.1.max(s.to);
            }
            return spans
                .into_iter()
                .filter_map(|(key, (from, to))| Some((self.copies.get(&key)?.copy.clone(), from, to)))
                .collect();
        }
        let slots = &self.slots;
        self.copies
            .values_mut()
            .map(|c| {
                if c.stale {
                    c.stale = false;
                    let (mut from, mut to) = (u64::MAX, 0);
                    for s in c.slots.iter().filter_map(|s| slots.get(s)) {
                        from = from.min(s.from);
                        to = to.max(s.to);
                    }
                    (c.from, c.to) = (from, to);
                }
                (c.copy.clone(), c.from, c.to)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A heap of slots with records, as title memory holds them.
    struct Heap {
        records: HashMap<u32, Record>,
    }

    impl Heap {
        fn record(&self, slot: u32) -> Option<Record> {
            self.records.get(&slot).copied()
        }
    }

    fn entry(r: Record) -> Option<u32> {
        Some(r[1])
    }

    /// One pass: plans, then "looks" at each due slot: a slot reads 16
    /// bytes from `slot * 16` of copy `slot / 4`.
    fn pass(p: &mut Picked<u32>, heap: &Heap, changed: Vec<u32>, forgotten: Vec<u32>) -> Vec<u32> {
        p.note_forgotten(Some(&forgotten));
        let due = p.plan(
            changed,
            || {
                let mut all: Vec<u32> = heap.records.keys().copied().collect();
                all.sort_unstable();
                all
            },
            |s| heap.record(s),
            entry,
        );
        for &slot in &due {
            let Some(record) = heap.record(slot) else {
                p.forget(slot);
                continue;
            };
            let copy = slot / 4;
            let from = u64::from(slot) * 16;
            p.keep(slot, copy as usize, &copy, from, from + 16, record, entry(record));
        }
        due
    }

    fn sorted(mut v: Vec<(u32, u64, u64)>) -> Vec<(u32, u64, u64)> {
        v.sort_by_key(|x| x.0);
        v
    }

    #[test]
    fn a_pass_and_the_old_one_agree_when_a_record_is_written_by_the_title() {
        let mut heap = Heap {
            records: HashMap::from([
                (0, [1, 10, 1]),
                (1, [1, 11, 1]),
                (4, [1, 12, 1]),
                (5, [1, 13, 1]),
            ]),
        };
        let mut old = Picked::new(true);
        let mut new = Picked::new(false);
        pass(&mut old, &heap, vec![], vec![]);
        pass(&mut new, &heap, vec![], vec![]);
        assert_eq!(sorted(new.spans()), sorted(old.spans()));

        // The title overwrites slot 5's record in memory itself (no API call
        // -> not in `changed`), extending nothing, removing no slot: the
        // old pass misses it until pass 16; the new one catches it at once.
        heap.records.insert(5, [1, 14, 1]);
        let due = pass(&mut new, &heap, vec![], vec![]);
        assert_eq!(due, [5]);
        assert_eq!(sorted(new.spans()), [(0, 0, 32), (1, 64, 96)]);

        // Removing a slot recomputes its copy's span.
        heap.records.remove(&0);
        pass(&mut old, &heap, vec![0], vec![]);
        pass(&mut new, &heap, vec![0], vec![]);
        assert_eq!(sorted(new.spans()), sorted(old.spans()));
        assert_eq!(sorted(new.spans()), [(0, 16, 32), (1, 64, 96)]);
    }

    #[test]
    fn freeing_a_view_entry_rechecks_its_slots_without_a_write() {
        let mut heap = Heap {
            records: HashMap::from([(0, [1, 10, 1]), (4, [1, 20, 1])]),
        };
        let mut new = Picked::new(false);
        pass(&mut new, &heap, vec![], vec![]);
        // Free entry 10 without touching slot 0's record.
        heap.records.remove(&0);
        let due = pass(&mut new, &heap, vec![], vec![10]);
        assert_eq!(due, [0]);
        assert_eq!(sorted(new.spans()), [(1, 64, 80)]);
    }

    #[test]
    fn orphans_are_retried_every_sixteen_passes_and_when_their_record_changes() {
        let mut new = Picked::<()>::new(false);
        let r0 = [1, 10, 1];
        let mut p = 0u64;
        new.plan(vec![], Vec::new, |_| None, entry);
        new.orphan(3, r0, entry(r0));
        while p < ORPHANS_EVERY - 1 {
            p += 1;
            let due = new.plan(vec![], Vec::new, |s| (s == 3).then_some(r0), entry);
            assert!(due.is_empty(), "pass {p} should not retry an unchanged orphan");
        }
        // Pass 16 retries it.
        let due = new.plan(vec![], Vec::new, |s| (s == 3).then_some(r0), entry);
        assert_eq!(due, [3]);
        // Changing its record retries it at once.
        let due = new.plan(vec![], Vec::new, |s| (s == 3).then_some([1, 11, 1]), entry);
        assert_eq!(due, [3]);
    }

    #[test]
    fn every_full_interval_rebuilds_everything() {
        let heap = Heap {
            records: HashMap::from([(0, [1, 10, 1])]),
        };
        let mut new = Picked::new(false);
        for p in 0..=FULL_EVERY {
            let due = pass(&mut new, &heap, vec![], vec![]);
            if p == 0 || p == FULL_EVERY {
                assert_eq!(due, [0], "pass {p}");
            } else {
                assert!(due.is_empty(), "pass {p}");
            }
        }
    }
}
