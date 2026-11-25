//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! The copies are the ones every other access uses ([`crate::shadow`]: a
//! title buffer's copy, or the 4 MiB window of memory outside buffers), so
//! a root view and a raw address reach the same host bytes. This module
//! keeps what a shader needs to find them:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; one buffer for all lists, entries only ever go from empty to
//!   filled;
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: the page numbers shaders found empty, read back here
//!   once a frame, so the copies exist from the next frame on. A pointer a
//!   shader takes out of memory cannot be known before it runs; the
//!   addresses a dispatch is given as root views are made ready before it.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next frame. For the structure builder that is the same as a dropped
//! stage, so before a dispatch the host makes ready everything it can see
//! the dispatch name: the whole copy around each root view, around each
//! pair of root constants that is a title address and around each root
//! constant that is one in 256-byte units (the builder passes structures
//! all three ways), and the 4 MiB after each (a structure may cross a
//! window's end).
//! shortcut: what a stage reaches only through pointers it reads out of
//! memory (InitHeaders' table of structure addresses, LoadInstances'
//! instance records) is not made ready unless something else named it;
//! the fault count in the log says whether that happens. The whole answer
//! is the extent of the title's allocation around an address, which the
//! memory module of the layer knows and this one does not ask yet.
//!
//! There are at most [`COPIES`] numbers (4 MiB windows mostly: 4 GiB at
//! most). When few are left, those of the copies dispatches named longest
//! ago are given back and taken again after a quarantine
//! ([`Memory::reclaim`]).
//! shortcut: "named" is what the host sees (root views, root constants,
//! faults); a copy only reached through pointers read out of memory looks
//! old and loses its number while still read, and its pages read zeros for
//! a frame before they are numbered again. Every dispatch of such a shader
//! brings all numbered copies up to date and, if it can store, declares
//! all of them written (the bytes are compared on the CPU afterwards, as
//! for any unordered access view): a dirty-page mask written by the shader
//! would bound both.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Copies that can be numbered (the run of heap slots is one longer). A
/// listed frame of the opening cutscene names about 50 windows.
const COPIES: u32 = 1023;
/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// A directory page.
const PAGE: u64 = 1 << 16;

struct Memory {
    /// Buffer, its mapping, its GPU address.
    directory: (ID3D12Resource, *mut u32, u64),
    faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]`.
    copies: Vec<Copy>,
    /// By copy: the frame a dispatch last named it or it was numbered in.
    used: Vec<u64>,
    /// Numbers given back ([`Memory::retire`]), oldest first: the index in
    /// `copies` and the frame it may be taken again from.
    free: std::collections::VecDeque<(usize, u64)>,
    /// Numbers taken again, in order: each heap's run rewrites their views.
    reused: Vec<u32>,
    /// Copies numbered during this [`ready`] call.
    fresh: Vec<usize>,
    /// Copies numbered in all, numbers taken again included.
    made: u64,
    /// Copies whose numbers were taken again, kept [`QUARANTINE`] frames
    /// more: a list recorded before may still reach them through a view
    /// not yet rewritten.
    gone: std::collections::VecDeque<(Arc<Shadow>, u64)>,
    /// Per host view heap (by address): the first slot of its run, how
    /// many copies have their view there, and how many of `reused` it has
    /// rewritten.
    heaps: HashMap<usize, (u32, u32, usize)>,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
}

// SAFETY: the mappings are of buffers the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

/// A numbered copy: live, or its number given back ([`Memory::retire`]),
/// kept until the number is taken again.
enum Copy {
    Live(Arc<Shadow>),
    Retired(Arc<Shadow>),
}

/// `VICTORY_MEMORY_RECLAIM=0`: numbers are never given back (as before).
/// Otherwise, when fewer than a sixteenth of [`COPIES`] are left, the
/// eighth used longest ago are given back ([`Memory::retire`]). With the
/// structure builder on, the numbers run out within minutes and keep
/// growing all the while.
fn reclaim_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MEMORY_RECLAIM").is_none_or(|v| v != "0"))
}

/// Frames a number given back waits before it is taken again: lists
/// recorded before it was given back have run by then, so none reads the
/// directory or a view as they were.
const QUARANTINE: u64 = 8;

/// Up to `k` of the indices `candidate` accepts, by `used` (the frame each
/// was last named in), named longest ago first.
fn longest_unused(used: &[u64], candidate: impl Fn(usize) -> bool, k: usize) -> Vec<(u64, usize)> {
    let mut old: Vec<(u64, usize)> = (0..used.len()).filter(|&i| candidate(i)).map(|i| (used[i], i)).collect();
    let k = k.max(1).min(old.len());
    if k == 0 {
        return old;
    }
    old.select_nth_unstable(k - 1);
    old.truncate(k);
    old.sort_unstable();
    old
}

/// The number given back longest ago, if its quarantine is over by `frame`.
fn due(free: &std::collections::VecDeque<(usize, u64)>, frame: u64) -> Option<usize> {
    free.front().filter(|&&(_, at)| at <= frame).map(|&(i, _)| i)
}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let make = |dwords: u64, what: &str| {
            let (buffer, data) = host.shared_buffer(dwords * 4)?;
            crate::host::name(&buffer, || format!("title memory by address: {what}"));
            // SAFETY: a fresh mapping of at least that many bytes.
            unsafe { std::ptr::write_bytes(data, 0, (dwords * 4) as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data.cast::<u32>(), gpu))
        };
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        Some(Memory {
            directory: make(u64::from(MEMORY_PAGES) * 2, "directory")?,
            faults: make(u64::from(MEMORY_FAULTS), "faults")?,
            copies: Vec::new(),
            used: Vec::new(),
            free: Default::default(),
            reused: Vec::new(),
            fresh: Vec::new(),
            made: 0,
            gone: Default::default(),
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: Default::default(),
        })
    }

    /// Gives the page of `address` its copy, if it has none and the memory
    /// is the title's.
    fn map(&mut self, address: u64) {
        let page = address / PAGE;
        // The last entry stays empty: addresses past the directory read it.
        if page >= u64::from(MEMORY_PAGES) - 1 {
            return;
        }
        // SAFETY: inside the directory's mapping.
        let entry = unsafe { self.directory.1.add(page as usize * 2) };
        // SAFETY: as above.
        let n = unsafe { entry.read() };
        if n != 0 {
            if let Some(used) = self.used.get_mut(n as usize - 1) {
                *used = self.frame;
            }
            return;
        }
        let start = page * PAGE;
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if self.refused.contains(&page) {
            return;
        }
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            self.refused.insert(page);
            return;
        }
        // The copy a root view at that address would get; a buffer that
        // ends inside the page is passed over for what holds all of it.
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let Some(shadow) = crate::shadow::find(start, 1)
            .filter(whole)
            .or_else(|| crate::shadow::find(start, PAGE).filter(whole))
        else {
            return;
        };
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return;
        }
        let n = match self.copies.iter().position(|c| matches!(c, Copy::Live(s) if Arc::ptr_eq(s, &shadow))) {
            Some(n) => n,
            None if self.number_ready() => {
                let Some(n) = self.number(Copy::Live(shadow.clone())) else { return };
                // Every page the copy holds whole, not only the one asked
                // for: a structure is more than 64 KiB.
                // shortcut: a page of a title buffer inside this one that
                // has a copy of its own is given this copy too.
                let (first, len) = shadow.span();
                let mut at = first.next_multiple_of(PAGE);
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    // SAFETY: inside the directory's mapping.
                    let other = unsafe { self.directory.1.add((at / PAGE) as usize * 2) };
                    // SAFETY: as above; the offset first, as below.
                    unsafe {
                        if other.read() == 0 {
                            other.add(1).write_volatile((at - first) as u32);
                            other.write_volatile(n as u32 + 1);
                        }
                    }
                    at += PAGE;
                }
                n
            }
            None => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(
                        COPIES,
                        "title memory by address: no number left for another copy; its pages read zeros (the first time; numbers given back are taken again after a quarantine)"
                    )
                });
                return;
            }
        };
        let offset = shadow.offset_in(start).unwrap_or(0);
        // The offset first: a shader that sees the number sees it too.
        // SAFETY: inside the directory's mapping.
        unsafe {
            entry.add(1).write_volatile(offset as u32);
            entry.write_volatile(n as u32 + 1);
        }
    }

    /// Whether a number is to be had: a new one, or one given back at
    /// least [`QUARANTINE`] frames ago.
    fn number_ready(&self) -> bool {
        self.copies.len() < COPIES as usize || due(&self.free, self.frame).is_some()
    }

    /// Numbers `copy`: a new number while there are fewer than [`COPIES`],
    /// else the one given back longest ago once its quarantine is over
    /// (`copies` index; none if neither, and `copy` is dropped).
    fn number(&mut self, copy: Copy) -> Option<usize> {
        let i = if self.copies.len() < COPIES as usize {
            self.copies.push(copy);
            self.used.push(self.frame);
            self.copies.len() - 1
        } else {
            let i = due(&self.free, self.frame)?;
            self.free.pop_front();
            if let Copy::Retired(was) = std::mem::replace(&mut self.copies[i], copy) {
                self.gone.push_back((was, self.frame + QUARANTINE));
            }
            self.used[i] = self.frame;
            self.reused.push(i as u32);
            i
        };
        self.fresh.push(i);
        self.made += 1;
        if self.made >= 256 && self.made.is_power_of_two() {
            // How fast the numbers go.
            tracing::info!(
                made = self.made,
                live = self.copies.len() - self.free.len(),
                of = COPIES,
                "title memory by address: copies numbered so far"
            );
        }
        Some(i)
    }

    /// Under pressure (fewer than a sixteenth of [`COPIES`] numbers left,
    /// given-back ones counted), gives back the numbers of the eighth of
    /// the copies that dispatches named longest ago ([`Memory::retire`]).
    /// Once a frame.
    fn reclaim(&mut self) {
        let max = COPIES as usize;
        if !reclaim_on() || self.copies.len() < max || self.free.len() >= max / 16 {
            return;
        }
        let old = longest_unused(&self.used, |i| matches!(self.copies[i], Copy::Live(_)), max / 8);
        let k = old.len();
        if k == 0 {
            return;
        }
        for &(_, i) in &old {
            self.retire(i);
        }
        static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 20 {
            tracing::info!(
                given = k,
                oldest = old.first().map_or(0, |o| o.0),
                newest = old.last().map_or(0, |o| o.0),
                frame = self.frame,
                "title memory by address: numbers of copies named longest ago given back"
            );
        }
    }

    /// Gives back copy `i`'s number: the directory entries that hold it are
    /// emptied (a shader that still reaches those pages finds them missing
    /// and they are numbered again the next frame), and the number is taken
    /// again [`QUARANTINE`] frames later.
    fn retire(&mut self, i: usize) {
        let Copy::Live(shadow) = &self.copies[i] else { return };
        let n = i as u32 + 1;
        let (first, len) = shadow.span();
        let last = (first + len).div_ceil(PAGE).min(u64::from(MEMORY_PAGES) - 1);
        for page in first / PAGE..last {
            // SAFETY: inside the directory's mapping.
            let entry = unsafe { self.directory.1.add(page as usize * 2) };
            // SAFETY: as above; the number first, so that no shader takes
            // the old offset with a new number.
            unsafe {
                if entry.read_volatile() == n {
                    entry.write_volatile(0);
                    entry.add(1).write_volatile(0);
                }
            }
        }
        let shadow = shadow.clone();
        self.copies[i] = Copy::Retired(shadow);
        self.free.push_back((i, self.frame + QUARANTINE));
    }

    /// The pages shaders found empty since the last frame.
    fn take_faults(&mut self) {
        let mut pages = Vec::new();
        for i in 1..MEMORY_FAULTS as usize {
            // SAFETY: inside the faults' mapping; the GPU may be writing,
            // a number missed now is written again next frame.
            let at = unsafe { self.faults.1.add(i) };
            // SAFETY: as above.
            let page = unsafe { at.read_volatile() };
            if page != 0 {
                pages.push(page);
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        if !pages.is_empty() {
            // Each is an access that read 0 or was dropped.
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                tracing::warn!(
                    pages = pages.len(),
                    first = format_args!("{:#x}", u64::from(pages[0]) * PAGE),
                    "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); the copies are made now"
                );
            }
        }
        for page in pages {
            self.map(u64::from(page) * PAGE);
        }
    }

    /// What a dispatch names: the copy around `address` and the 4 MiB
    /// after it.
    fn name(&mut self, address: u64) {
        self.map(address);
        self.map((address & !(CELL - 1)) + CELL);
    }
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name gets its copies (`addresses`: its root views;
/// `constants`: its root constants, see the module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted (and, with `writes`, copied back after). None if the host
/// cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    constants: &[u32],
    writes: bool,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        while m.gone.front().is_some_and(|g| g.1 <= frame) {
            m.gone.pop_front();
        }
        m.reclaim();
        m.take_faults();
    }
    m.fresh.clear();
    for &address in addresses {
        m.name(address);
    }
    // A title address in two constants (low dword, then bits 32 to 39), or
    // in one in 256-byte units. A constant that only looks like one names
    // memory nothing reads; one that is not the title's memory is refused.
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0xFF).contains(&high)
            && c >= 0x1_0000
        {
            m.name(u64::from(c) | u64::from(high) << 32);
        }
        if c >= 0x0100_0000 {
            m.name(u64::from(c) << 8);
        }
    }
    let key = std::ptr::from_ref(heap) as usize;
    let (base, made, seen) = match m.heaps.get(&key) {
        Some(&run) => run,
        None => {
            // Held for the heap's life: not given back with a list's slots.
            let base = heap.dynamic(COPIES + 1)?;
            heap.null_raw_uav_at(base);
            (base, 0, m.reused.len())
        }
    };
    let view = |n: usize, copy: &Copy| match copy {
        Copy::Live(shadow) => heap.raw_uav_at(base + 1 + n as u32, &shadow.buffer, 0, shadow.span().1),
        Copy::Retired(_) => heap.null_raw_uav_at(base + 1 + n as u32),
    };
    // Numbers taken again since this heap's run last looked: their views
    // show the new copies (the old ones were in quarantine till then).
    for &n in &m.reused[seen..] {
        if n < made {
            view(n as usize, &m.copies[n as usize]);
        }
    }
    for (n, copy) in m.copies.iter().enumerate().skip(made as usize) {
        view(n, copy);
    }
    m.heaps.insert(key, (base, m.copies.len() as u32, m.reused.len()));
    // Once per list, and for copies numbered since: the builder alone
    // dispatches hundreds of times a frame.
    // shortcut: a copy another list numbers meanwhile is brought up to date
    // by that list's submission, which may run after this one's.
    const READ: usize = usize::MAX - 0x6d;
    const WRITE: usize = usize::MAX - 0x6e;
    let first_read = list.first_use(READ);
    let first_write = writes && list.first_use(WRITE);
    let bring = |copy: &Copy, read: bool, write: bool| {
        // (a number given back has no pages)
        let Copy::Live(shadow) = copy else { return };
        let (start, len) = shadow.span();
        if read {
            shadow.refresh(list, start, len);
        }
        if write {
            shadow.gpu_writes(list, start, len);
        }
    };
    if first_read || first_write {
        for copy in &m.copies {
            bring(copy, first_read, first_write);
        }
    }
    for &n in &m.fresh {
        bring(&m.copies[n], !first_read, writes && !first_write);
    }
    if first_read {
        list.keep(&m.directory.0);
        list.keep(&m.faults.0);
    }
    Some(Ready {
        directory: m.directory.2,
        faults: m.faults.2,
        base,
    })
}

#[cfg(test)]
mod tests {
    use super::{QUARANTINE, due, longest_unused};

    /// Under pressure the copies named longest ago give their numbers back
    /// (oldest first, only those the filter takes); a number given back is
    /// taken again only once its quarantine is over, in the order given
    /// back.
    #[test]
    fn numbers_named_longest_ago_go_back_and_wait_out_a_quarantine() {
        // Frames each copy was last named in; copy 3 is not a candidate.
        let used = [50, 12, 90, 3, 40, 12];
        let candidate = |i: usize| i != 3;
        assert_eq!(longest_unused(&used, candidate, 3), [(12, 1), (12, 5), (40, 4)]);
        // More asked than there are: all of them, still oldest first.
        assert_eq!(longest_unused(&used, candidate, 99).len(), 5);
        assert!(longest_unused(&[], |_| true, 4).is_empty());
        let mut free = std::collections::VecDeque::new();
        assert_eq!(due(&free, 100), None);
        free.push_back((1, 100 + QUARANTINE));
        free.push_back((5, 101 + QUARANTINE));
        assert_eq!(due(&free, 100), None);
        assert_eq!(due(&free, 100 + QUARANTINE - 1), None);
        assert_eq!(due(&free, 100 + QUARANTINE), Some(1));
        free.pop_front();
        assert_eq!(due(&free, 100 + QUARANTINE), None);
        assert_eq!(due(&free, 101 + QUARANTINE), Some(5));
    }
}
