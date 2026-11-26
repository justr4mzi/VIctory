//! Which processors our own threads run on, where a processor has two kinds
//! of cores. `cpu_sets` and `keep_on_fast_cores` are fxceless/victory-fork's
//! (`framelog.rs`, commit 0b0074e0, GPL-3.0-or-later, as this).

/// The system's processor sets of group 0: each one's id, its efficiency
/// class (higher is faster) and its core (which two of them may share).
fn cpu_sets() -> Vec<(u32, u8, u8)> {
    use windows_sys::Win32::System::SystemInformation::{GetSystemCpuSetInformation, SYSTEM_CPU_SET_INFORMATION};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut bytes = 0u32;
    // SAFETY: a size query, then the call into a buffer of that size.
    unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut bytes, GetCurrentProcess(), 0) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: as above; the buffer holds `bytes` bytes, aligned for the records.
    let ok = unsafe { GetSystemCpuSetInformation(buffer.as_mut_ptr().cast(), bytes, &mut bytes, GetCurrentProcess(), 0) };
    let mut sets = Vec::new();
    let mut at = 0usize;
    while ok != 0 && at + size_of::<SYSTEM_CPU_SET_INFORMATION>() <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe { &*(buffer.as_ptr().cast::<u8>().add(at).cast::<SYSTEM_CPU_SET_INFORMATION>()) };
        if info.Size == 0 {
            break;
        }
        // SAFETY: the record's type is CpuSet (the only one there is).
        let set = unsafe { &info.Anonymous.CpuSet };
        if set.Group == 0 {
            sets.push((set.Id, set.EfficiencyClass, set.CoreIndex));
        }
        at += info.Size as usize;
    }
    sets
}

/// The bytes of the last-level cache each processor set of group 0 shares,
/// by set id (none known: empty).
fn l3_of_sets() -> Vec<(u32, u32)> {
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, GetSystemCpuSetInformation, RelationCache, SYSTEM_CPU_SET_INFORMATION,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut bytes = 0u32;
    // SAFETY: a size query, then the call into a buffer of that size.
    unsafe { GetLogicalProcessorInformationEx(RelationCache, std::ptr::null_mut(), &mut bytes) };
    let mut buffer = vec![0u8; (bytes as usize).max(1)];
    // SAFETY: as above; the buffer holds `bytes` bytes.
    let ok = unsafe { GetLogicalProcessorInformationEx(RelationCache, buffer.as_mut_ptr().cast(), &mut bytes) };
    let by_processor = if ok != 0 { largest_caches(&buffer[..(bytes as usize).min(buffer.len())]) } else { [0; 64] };
    // Processor indices to set ids.
    let mut bytes = 0u32;
    // SAFETY: as for `cpu_sets`.
    unsafe { GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut bytes, GetCurrentProcess(), 0) };
    let mut buffer = vec![0u64; (bytes as usize).div_ceil(8).max(1)];
    // SAFETY: as above.
    let ok = unsafe { GetSystemCpuSetInformation(buffer.as_mut_ptr().cast(), bytes, &mut bytes, GetCurrentProcess(), 0) };
    let mut sizes = Vec::new();
    let mut at = 0usize;
    while ok != 0 && at + size_of::<SYSTEM_CPU_SET_INFORMATION>() <= bytes as usize {
        // SAFETY: a record inside the buffer; each says its own size.
        let info = unsafe { &*(buffer.as_ptr().cast::<u8>().add(at).cast::<SYSTEM_CPU_SET_INFORMATION>()) };
        if info.Size == 0 {
            break;
        }
        // SAFETY: the record's type is CpuSet (the only one there is).
        let set = unsafe { &info.Anonymous.CpuSet };
        if set.Group == 0 && let Some(&size) = by_processor.get(set.LogicalProcessorIndex as usize) {
            sizes.push((set.Id, size));
        }
        at += info.Size as usize;
    }
    sizes
}

/// The largest cache of each logical processor of group 0, by its index, of
/// the cache records `GetLogicalProcessorInformationEx` wrote. Each says its
/// own size, less than the structure's (whose union is as large as its
/// largest member): the last one, left out where a whole structure was
/// asked for (an L3 listed last with it), is read too.
fn largest_caches(records: &[u8]) -> [u32; 64] {
    use windows_sys::Win32::System::SystemInformation::{RelationCache, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX};
    let mut by_processor = [0u32; 64];
    let mut at = 0usize;
    while let Some(rest) = records.get(at..).filter(|r| r.len() >= 8) {
        let size = u32::from_ne_bytes([rest[4], rest[5], rest[6], rest[7]]) as usize;
        if size == 0 || size > rest.len() {
            break;
        }
        // SAFETY: plain data, of which zeroes are a value.
        let mut info: SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX = unsafe { std::mem::zeroed() };
        // The record's own bytes; the rest of the structure stays zero.
        let n = size.min(size_of_val(&info));
        // SAFETY: `n` bytes inside both.
        unsafe { std::ptr::copy_nonoverlapping(rest.as_ptr(), (&raw mut info).cast::<u8>(), n) };
        if info.Relationship == RelationCache {
            // SAFETY: a cache record (asked for), with at least its first group mask.
            let cache = unsafe { &info.Anonymous.Cache };
            let mask = unsafe { cache.Anonymous.GroupMask };
            if mask.Group == 0 {
                for (p, size) in by_processor.iter_mut().enumerate() {
                    if mask.Mask >> p & 1 != 0 {
                        *size = (*size).max(cache.CacheSize);
                    }
                }
            }
        }
        at += size;
    }
    by_processor
}

/// Whether the cores of the largest last-level cache count as the fast
/// ones where all cores are of one class (`VICTORY_CACHE_CORES=0`: not).
/// On a processor with two core complexes of which one has the stacked cache
/// (Ryzen 7900X3D, 7950X3D, 9900X3D, 9950X3D) every core reports the same
/// efficiency class, so nothing was asked for: our threads ran on either
/// complex. Their work (compares, hashes, copies of tens of megabytes a
/// frame) is the kind the larger cache serves. Nothing changes on a processor
/// whose caches are all alike, or with two kinds of cores.
fn cache_cores() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_CACHE_CORES") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CACHE_CORES").is_none_or(|v| v != "0"))
}

/// The sets of the fastest class, if there are slower ones too.
fn fast_by_class(sets: &[(u32, u8, u8)]) -> Vec<u32> {
    let best = sets.iter().map(|c| c.1).max().unwrap_or(0);
    let fast: Vec<u32> = sets.iter().filter(|c| c.1 == best).map(|c| c.0).collect();
    if fast.len() == sets.len() { Vec::new() } else { fast }
}

/// The sets on the largest last-level cache (`l3`: bytes by set id), if
/// another one is smaller and the largest serves at least eight sets.
fn fast_by_cache(sets: &[(u32, u8, u8)], l3: &[(u32, u32)]) -> Vec<u32> {
    let size = |id: u32| l3.iter().find(|s| s.0 == id).map_or(0, |s| s.1);
    let sizes: Vec<u32> = sets.iter().map(|s| size(s.0)).collect();
    let (Some(&most), Some(&least)) = (sizes.iter().max(), sizes.iter().min()) else {
        return Vec::new();
    };
    if least == 0 || least == most {
        return Vec::new();
    }
    let fast: Vec<u32> = sets.iter().filter(|s| size(s.0) == most).map(|s| s.0).collect();
    if fast.len() < 8 { Vec::new() } else { fast }
}

/// The fast sets: those of the fastest class if there are slower ones, else
/// those of the largest cache ([`cache_cores`]). `now`: as things are at this
/// moment (else whether there could be any, for threads to ask at all).
fn fast_sets(sets: &[(u32, u8, u8)], now: bool) -> Vec<u32> {
    let by_class = fast_by_class(sets);
    if !by_class.is_empty() || (now && !cache_cores()) {
        return by_class;
    }
    static L3: std::sync::OnceLock<Vec<(u32, u32)>> = std::sync::OnceLock::new();
    fast_by_cache(sets, L3.get_or_init(l3_of_sets))
}

fn fast_of(sets: &[(u32, u8, u8)]) -> Vec<u32> {
    fast_sets(sets, true)
}

/// The sets of one fast core of the last, and all the others, for
/// [`lane_core`]; `None` with fewer than six fast cores.
fn one_core(sets: &[(u32, u8, u8)]) -> Option<(Vec<u32>, Vec<u32>)> {
    let best = sets.iter().map(|c| c.1).max()?;
    let mut cores: Vec<u8> = sets.iter().filter(|c| c.1 == best).map(|c| c.2).collect();
    cores.sort_unstable();
    cores.dedup();
    let core = *cores.last().filter(|_| cores.len() >= 6)?;
    let of = |own: bool| sets.iter().filter(|c| (c.2 == core) == own).map(|c| c.0).collect();
    Some((of(true), of(false)))
}

/// Whether the submitting thread has a fast core to itself, as far as this
/// process goes (`VICTORY_LANE_CORE=1`): nothing else of it runs on the
/// core's two threads. It was the one a frame waited for (94 % of it in a
/// street), and shared its core's units with whatever ran beside it.
fn lane_core() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_LANE_CORE") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LANE_CORE").is_some_and(|v| v == "1"))
}

/// Threads that asked: a handle of each, and whether it is the submitting one.
static ASKED: std::sync::Mutex<Vec<(usize, bool)>> = std::sync::Mutex::new(Vec::new());

/// What the threads of the process, and those that asked, are told now
/// ([`lane_core`], which may have changed).
pub(crate) fn settle() {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, SetProcessDefaultCpuSets, SetThreadSelectedCpuSets};
    if stage() == 0 {
        return;
    }
    let sets = cpu_sets();
    let fast = fast_of(&sets);
    let one = one_core(&sets).filter(|_| lane_core());
    let others: &[u32] = one.as_ref().map_or(&[], |o| &o.1);
    let null = std::ptr::null::<u32>();
    // SAFETY: our own process and the system's own set ids (none: all).
    unsafe {
        let at = if others.is_empty() { null } else { others.as_ptr() };
        SetProcessDefaultCpuSets(GetCurrentProcess(), at, others.len() as u32);
    }
    let helpers: Vec<u32> = match &one {
        Some((own, _)) => fast.iter().copied().filter(|id| !own.contains(id)).collect(),
        None => fast.clone(),
    };
    for &(thread, lane) in ASKED.lock().unwrap_or_else(|e| e.into_inner()).iter() {
        let sets = match &one {
            Some((own, _)) if lane => own,
            _ => &helpers,
        };
        // SAFETY: a handle kept open, and the system's own set ids.
        unsafe { SetThreadSelectedCpuSets(thread as _, sets.as_ptr(), sets.len() as u32) };
    }
    tracing::info!(on = one.is_some(), core = ?one.as_ref().map(|o| &o.0), "the submitting thread has a core of its own (VICTORY_LANE_CORE)");
    tracing::info!(
        fast = fast.len(),
        of = sets.len(),
        by_cache = fast_by_class(&sets).is_empty() && !fast.is_empty(),
        threads = ASKED.lock().unwrap_or_else(|e| e.into_inner()).len(),
        "our threads' processors (VICTORY_CACHE_CORES)"
    );
}

/// `VICTORY_FAST_CORES`: 0: no thread asks; 1: the threads a frame waits for
/// one by one; 2 (the default): the threads they share their work with too;
/// 3: those that work beside the rest as well (lists translated, descriptors
/// made): more threads by then than the fast cores run at a time, while the
/// slow ones stood idle.
fn stage() -> u32 {
    static STAGE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *STAGE.get_or_init(|| std::env::var("VICTORY_FAST_CORES").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(2))
}

/// Asks the system to run the calling thread, which a frame waits for, on
/// the fast cores only, where the processor has two kinds. That fork's submit
/// thread was found on a slow one in 1,529 looks of 24,021. (A selection of
/// processors is a preference the scheduler keeps unless nothing of it is
/// free.) The title's threads are not asked for.
pub(crate) fn keep_on_fast_cores(what: &str) {
    ask(what, 1);
}

/// [`keep_on_fast_cores`] for one of the threads a submission's preparing or
/// a landing shares its work with.
pub(crate) fn keep_helper_on_fast_cores(what: &str) {
    ask(what, 2);
}

/// [`keep_on_fast_cores`] for one of many threads whose work is ahead of what
/// a frame waits for.
pub(crate) fn keep_many_on_fast_cores(what: &str) {
    ask(what, 3);
}

fn ask(what: &str, from: u32) {
    use windows_sys::Win32::System::Threading::{
        GetCurrentThreadId, OpenThread, THREAD_SET_LIMITED_INFORMATION,
    };
    if stage() < from {
        return;
    }
    static FAST: std::sync::OnceLock<(Vec<u32>, usize)> = std::sync::OnceLock::new();
    let (fast, of) = FAST.get_or_init(|| {
        let sets = cpu_sets();
        // (whether there can be fast sets at all: `VICTORY_CACHE_CORES` may
        // change while the title runs, and `settle` follows it)
        (fast_sets(&sets, false), sets.len())
    });
    // All alike (or nothing known): nothing to ask for.
    if fast.is_empty() {
        return;
    }
    // SAFETY: our own thread; the handle stays open.
    let thread = unsafe { OpenThread(THREAD_SET_LIMITED_INFORMATION, 0, GetCurrentThreadId()) };
    if thread.is_null() {
        tracing::warn!(what, error = %std::io::Error::last_os_error(), "the thread could not be kept on the fast cores");
        return;
    }
    ASKED.lock().unwrap_or_else(|e| e.into_inner()).push((thread as usize, what == "victory-submit"));
    settle();
    tracing::info!(what, fast = fast.len(), of, "the thread is kept on the fast cores (VICTORY_FAST_CORES=0: not)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fast_cores_are_those_of_the_best_class_if_there_is_a_worse_one() {
        assert_eq!(fast_by_class(&[(256, 1, 0), (257, 1, 0), (258, 0, 2)]), [256, 257]);
        assert!(fast_by_class(&[(256, 0, 0), (257, 0, 1)]).is_empty());
        assert!(one_core(&[(256, 1, 0), (257, 1, 0), (258, 0, 2)]).is_none());
        let eight: Vec<(u32, u8, u8)> = (0..16).map(|n| (n, 1, n as u8 / 2 * 2)).chain((16..24).map(|n| (n, 0, n as u8))).collect();
        let (own, others) = one_core(&eight).unwrap();
        assert_eq!((own, others.len()), (vec![14, 15], 22));
        assert!(fast_by_class(&[]).is_empty());
        assert!(!cpu_sets().is_empty());
        std::thread::spawn(|| keep_on_fast_cores("test")).join().unwrap();
    }

    const MB: u32 = 1 << 20;

    /// A cache record as the system writes it: as many bytes as it says.
    fn cache_record(level: u8, bytes: u32, mask: usize) -> Vec<u8> {
        use windows_sys::Win32::System::SystemInformation::{
            CACHE_RELATIONSHIP, RelationCache, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
        };
        // SAFETY: plain data.
        let mut info: SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX = unsafe { std::mem::zeroed() };
        let size = std::mem::offset_of!(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Anonymous) + size_of::<CACHE_RELATIONSHIP>();
        info.Relationship = RelationCache;
        info.Size = size as u32;
        info.Anonymous.Cache.Level = level;
        info.Anonymous.Cache.CacheSize = bytes;
        info.Anonymous.Cache.Anonymous.GroupMask.Mask = mask;
        // SAFETY: the structure's first `size` bytes.
        unsafe { std::slice::from_raw_parts((&raw const info).cast::<u8>(), size) }.to_vec()
    }

    #[test]
    fn the_last_cache_record_is_read_too() {
        // Two complexes of eight processors, two to a core; the second
        // one's L3, the stacked one, listed last.
        let l2 = |core: usize| cache_record(2, MB, 0x3 << (2 * core));
        let records: Vec<u8> = (0..4)
            .map(l2)
            .chain([cache_record(3, 32 * MB, 0xff)])
            .chain((4..8).map(l2))
            .chain([cache_record(3, 96 * MB, 0xff00)])
            .collect::<Vec<_>>()
            .concat();
        // (Shorter than the structure: the last one holds no whole one.)
        let whole = size_of::<windows_sys::Win32::System::SystemInformation::SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>();
        assert!(cache_record(3, 0, 0).len() < whole);
        let by_processor = largest_caches(&records);
        assert_eq!(by_processor[..16], [[32 * MB; 8], [96 * MB; 8]].concat()[..]);
        assert!(by_processor[16..].iter().all(|&s| s == 0));
        // The stacked complex is the fast one (with the last record left
        // out, its processors had their L2s and the other one was).
        let sets: Vec<(u32, u8, u8)> = (0..16).map(|i| (i, 0, (i / 2) as u8)).collect();
        let l3: Vec<(u32, u32)> = (0..16).map(|i| (i, by_processor[i as usize])).collect();
        assert_eq!(fast_by_cache(&sets, &l3), (8..16).collect::<Vec<_>>());
        // A record cut short, or one of no size: nothing of it, nor after.
        assert_eq!(largest_caches(&records[..records.len() - 1])[8], MB);
        let mut none = cache_record(3, 64 * MB, 0xffff);
        none[4..8].fill(0);
        assert_eq!(largest_caches(&[none, records].concat()), [0; 64]);
        assert_eq!(largest_caches(&[]), [0; 64]);
    }

    /// `n` sets of one class, two to a core, from id 256.
    fn alike(n: u32) -> Vec<(u32, u8, u8)> {
        (0..n).map(|i| (256 + i, 0, (i / 2) as u8)).collect()
    }

    #[test]
    fn the_largest_cache_is_fast_only_where_the_caches_differ() {
        // Two complexes, one with the stacked cache (9950X3D: 16 sets each).
        let sets = alike(32);
        let l3: Vec<(u32, u32)> = (0..32).map(|i| (256 + i, if i < 16 { 96 * MB } else { 32 * MB })).collect();
        assert_eq!(fast_by_cache(&sets, &l3), (256..272).collect::<Vec<_>>());
        // The stacked cache on the second complex: that one.
        let flipped: Vec<(u32, u32)> = (0..32).map(|i| (256 + i, if i < 16 { 32 * MB } else { 96 * MB })).collect();
        assert_eq!(fast_by_cache(&sets, &flipped), (272..288).collect::<Vec<_>>());
        // Two complexes alike (9950X): nothing.
        let even: Vec<(u32, u32)> = (0..32).map(|i| (256 + i, 32 * MB)).collect();
        assert!(fast_by_cache(&sets, &even).is_empty());
        // One complex (7800X3D, 9800X3D): nothing.
        let one: Vec<(u32, u32)> = (0..16).map(|i| (256 + i, 96 * MB)).collect();
        assert!(fast_by_cache(&alike(16), &one).is_empty());
        // Caches not known: nothing.
        assert!(fast_by_cache(&sets, &[]).is_empty());
        assert!(fast_by_cache(&sets, &l3[..16]).is_empty());
        // A larger cache of under eight sets: nothing.
        let few: Vec<(u32, u32)> = (0..32).map(|i| (256 + i, if i < 4 { 64 * MB } else { 32 * MB })).collect();
        assert!(fast_by_cache(&sets, &few).is_empty());
    }

    #[test]
    fn two_kinds_of_cores_still_go_by_their_class() {
        // Eight fast cores (two sets each) and eight slow ones, the caches
        // told to differ too: the class decides, whatever the caches say.
        let sets: Vec<(u32, u8, u8)> =
            (0..16).map(|n| (n, 1, n as u8 / 2 * 2)).chain((16..24).map(|n| (n, 0, n as u8))).collect();
        assert_eq!(fast_sets(&sets, true), (0..16).collect::<Vec<_>>());
        assert_eq!(fast_sets(&sets, false), (0..16).collect::<Vec<_>>());
    }
}
