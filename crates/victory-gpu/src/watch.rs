//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection.
//!
//! shortcut: writes into physical pages through another view (xmem's
//! `write_physical`, DirectStorage's page destinations) are not seen at
//! the guarded addresses; they go to the texture pool, whose textures are
//! hashed, not guarded. A system call writing into a guarded block fails
//! (none is known to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// The thread preparing a submission ([`Sweep`]), if one is.
    sweeper: Option<std::thread::ThreadId>,
    /// Page ranges that thread asked the kernel about in this sweep
    /// (sorted, disjoint, end exclusive).
    swept: Vec<(usize, usize)>,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    found: Vec::new(),
    sweeper: None,
    swept: Vec::new(),
});

/// While alive, the calling thread prepares one submission: the title's
/// writes it must see were all made before the submission began (but for
/// steps the preparation runs itself, [`Sweep::title_written`]), so a page
/// the kernel was asked about once in the sweep need not be asked about
/// again. Nested buffers in the world (several views into one 174 MB
/// block) otherwise asked about the same pages ~20 times a frame, 120-180
/// ms of a frame on an RX 6700 XT (2026-11-24).
pub(crate) struct Sweep(());

impl Sweep {
    pub(crate) fn begin() -> Self {
        let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
        j.sweeper = Some(std::thread::current().id());
        j.swept.clear();
        Sweep(())
    }

    /// The thread in a sweep is about to write title memory itself (a step
    /// run on the CPU between lists): what it asked about before must be
    /// asked about again.
    pub(crate) fn title_written() {
        let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
        if j.sweeper == Some(std::thread::current().id()) {
            j.swept.clear();
        }
    }
}

impl Drop for Sweep {
    fn drop(&mut self) {
        let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
        j.sweeper = None;
        j.swept.clear();
    }
}

/// The parts of `[first, last)` not in `swept` (sorted, disjoint).
fn gaps(swept: &[(usize, usize)], first: usize, last: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut at = first;
    for &(a, e) in swept {
        if e <= at {
            continue;
        }
        if a >= last {
            break;
        }
        if a > at {
            out.push((at, a));
        }
        at = at.max(e);
        if at >= last {
            break;
        }
    }
    if at < last {
        out.push((at, last));
    }
    out
}

/// Adds `[first, last)` to `swept`, keeping it sorted and disjoint.
fn sweep_in(swept: &mut Vec<(usize, usize)>, first: usize, last: usize) {
    let i = swept.partition_point(|&(_, e)| e < first);
    let mut j = i;
    let (mut a, mut e) = (first, last);
    while j < swept.len() && swept[j].0 <= e {
        a = a.min(swept[j].0);
        e = e.max(swept[j].1);
        j += 1;
    }
    swept.splice(i..j, [(a, e)]);
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // In a sweep, only what this thread has not asked about yet in it.
    let sweeping = j.sweeper == Some(std::thread::current().id());
    let asks = if sweeping {
        gaps(&j.swept, first, last)
    } else {
        vec![(first, last)]
    };
    let mut found: Vec<usize> = Vec::new();
    let mut failed = false;
    for &(a, e) in &asks {
        let mut count = (e - a) * PAGE / KERNEL_PAGE;
        j.found.resize(count.max(j.found.len()), 0);
        let mut granularity = 0;
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        // SAFETY: the answer buffer holds `count` entries.
        failed = unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (a * PAGE) as *const _,
                (e - a) * PAGE,
                j.found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::WatchUs, started.elapsed().as_micros() as u64);
        }
        if failed {
            break;
        }
        found.extend(j.found[..count].iter().map(|&at| at / PAGE));
    }
    let guarded = if failed {
        // Not write-watched memory (the kernel answers for a whole
        // allocation or not at all), so none of the range went in `swept`.
        found.clear();
        if !guarded {
            return None;
        }
        guard(first * PAGE, last * PAGE)?
    } else {
        if sweeping {
            for &(a, e) in &asks {
                sweep_in(&mut j.swept, a, e);
            }
        }
        Vec::new()
    };
    j.asked += 1;
    let now = j.asked;
    for page in found.into_iter().chain(guarded) {
        j.written.insert(page, now);
    }
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

struct Guards {
    /// Blocks we made read-only.
    protected: BTreeSet<usize>,
    /// Of those, the ones not written since a copy took them.
    clean: BTreeSet<usize>,
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BTreeSet::new(),
    clean: BTreeSet::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    use windows_sys::Win32::System::Memory::{MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(start as *const _, &mut info, size_of_val(&info)) };
    if n == 0 || info.Type != MEM_MAPPED {
        return None;
    }
    install();
    let mut g = guards();
    let mut written = Vec::new();
    let mut run: Option<usize> = None;
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    for block in first..=last {
        let open = block < last && !g.clean.contains(&block);
        if open {
            written.extend(block * BLOCK / PAGE..(block + 1) * BLOCK / PAGE);
            if g.protected.contains(&block) {
                // Written through the runtime, still read-only.
                g.clean.insert(block);
            } else {
                run.get_or_insert(block);
                continue;
            }
        }
        if let Some(from) = run.take() {
            protect(&mut g, from, block);
        }
    }
    Some(written)
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail (unmapped).
fn protect(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        // SAFETY: title view memory; only its protection changes.
        unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        }
    };
    let note = |g: &mut Guards, from: usize, to: usize| {
        g.protected.extend(from..to);
        g.clean.extend(from..to);
    };
    if set(from, to) {
        note(g, from, to);
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            note(g, block, block + 1);
        }
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let blocks: Vec<usize> = g
        .protected
        .range(start / BLOCK..end.div_ceil(BLOCK))
        .copied()
        .collect();
    for &block in &blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
    }
    !blocks.is_empty()
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut guards(), at, at.saturating_add(len));
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded.
pub fn title_memory_remapped(at: usize, len: usize) {
    let mut g = guards();
    let blocks: Vec<usize> = g
        .protected
        .range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK))
        .copied()
        .collect();
    for block in blocks {
        g.protected.remove(&block);
        g.clean.remove(&block);
    }
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn gaps_and_sweeps_keep_ranges_disjoint() {
        let mut s = Vec::new();
        assert_eq!(gaps(&s, 0, 10), [(0, 10)]);
        sweep_in(&mut s, 4, 6);
        sweep_in(&mut s, 10, 12);
        assert_eq!(gaps(&s, 0, 20), [(0, 4), (6, 10), (12, 20)]);
        assert_eq!(gaps(&s, 4, 6), []);
        assert_eq!(gaps(&s, 5, 11), [(6, 10)]);
        sweep_in(&mut s, 6, 10); // touching both: one range
        assert_eq!(s, [(4, 12)]);
        sweep_in(&mut s, 0, 2);
        sweep_in(&mut s, 1, 5);
        assert_eq!(s, [(0, 12)]);
    }

    /// A fresh write-watched allocation of `pages` pages. Never freed: the
    /// journal keeps its pages, and a later test given the same addresses
    /// would find them written.
    fn watched(pages: usize) -> usize {
        // SAFETY: a fresh allocation, kept for the rest of the run.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                pages * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        p
    }

    /// In a sweep a nested range is not asked about again, yet sees what
    /// the outer asking found; after the sweep the kernel is asked as before.
    #[test]
    fn a_sweep_shares_the_kernels_answer_with_nested_ranges() {
        let p = watched(4);
        // SAFETY: committed pages of ours.
        unsafe { ((p + 2 * PAGE) as *mut u8).write(1) };
        let sweep = Sweep::begin();
        let (outer, all) = writes(p, p + 4 * PAGE, false).expect("watched");
        assert_eq!(all, [(p / PAGE + 2, outer)]);
        // Written after the outer asking by another party, inside the
        // sweep: seen from the next submission's sweep on.
        // SAFETY: as above.
        unsafe { ((p + 3 * PAGE) as *mut u8).write(1) };
        let (inner, nested) = writes(p + 2 * PAGE, p + 4 * PAGE, false).expect("watched");
        assert!(inner > outer);
        assert_eq!(nested, [(p / PAGE + 2, outer)]);
        drop(sweep);
        let (later, after) = writes(p + 2 * PAGE, p + 4 * PAGE, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 2, outer), (p / PAGE + 3, later)]);
    }

    /// A write the sweeping thread makes itself (after `title_written`) is
    /// seen by the next asking in the same sweep.
    #[test]
    fn a_sweep_asks_again_after_its_own_write() {
        let p = watched(2);
        let sweep = Sweep::begin();
        let (_, before) = writes(p, p + 2 * PAGE, false).expect("watched");
        assert!(before.is_empty());
        Sweep::title_written();
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE) as *mut u8).write(1) };
        let (now, after) = writes(p, p + 2 * PAGE, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        drop(sweep);
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(!read(&mut file), "the kernel does not fault into our handler");
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        assert_eq!(unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) }, b"from the disk");
        assert_eq!(guard(view, view + 2 * BLOCK), Some((view / PAGE..view / PAGE + 1).collect()));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, 2 * BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }
}
