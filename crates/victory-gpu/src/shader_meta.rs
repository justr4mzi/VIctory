//! Persistent per-shader resource metadata cache (`shader-meta.bin`).
//!
//! Stores each translated container's `(stores, used)` metadata keyed by the
//! 32-byte BLAKE3 hash of the console shader container so warm launches skip
//! `Container::parse` and `native_programs` on cached shader hits without
//! changing `shader_cache.rs` keys.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};

use victory_shader::{ResourceType, RootType, UsedResource};

const FILE_NAME: &str = "shader-meta.bin";
const MAGIC: &[u8; 4] = b"VSM2";
const VERSION: u32 = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CachedMeta {
    pub(crate) stores: bool,
    pub(crate) used: Vec<UsedResource>,
}

type MetaMap = HashMap<[u8; 32], CachedMeta>;

struct MetaStore {
    map: MetaMap,
    dirty: bool,
}

impl MetaStore {
    fn empty() -> Self {
        Self {
            map: HashMap::new(),
            dirty: false,
        }
    }

    fn load(dir: &Path) -> Self {
        Self {
            map: load_from(&dir.join(FILE_NAME)).unwrap_or_default(),
            dirty: false,
        }
    }
}

struct Store {
    meta: RwLock<MetaStore>,
    save: Mutex<()>,
}

impl Store {
    fn empty() -> Self {
        Self {
            meta: RwLock::new(MetaStore::empty()),
            save: Mutex::new(()),
        }
    }

    fn load(dir: &Path) -> Self {
        Self {
            meta: RwLock::new(MetaStore::load(dir)),
            save: Mutex::new(()),
        }
    }
}

static STORE: OnceLock<Store> = OnceLock::new();
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

fn global_store() -> &'static Store {
    STORE.get_or_init(Store::empty)
}

/// Loads `dir/shader-meta.bin` into the global metadata store.
///
/// Missing or corrupt files silently start an empty store.
pub(crate) fn init(dir: &Path) {
    let loaded = Store::load(dir);
    if let Err(store) = STORE.set(loaded) {
        let mut guard = global_store().meta.write().unwrap_or_else(|e| e.into_inner());
        *guard = store.meta.into_inner().unwrap_or_else(|e| e.into_inner());
    }
}

/// Returns cached shader metadata for `hash`, or computes and stores it on a miss.
pub(crate) fn get_or_compute<F>(hash: &[u8; 32], compute: F) -> Result<CachedMeta, String>
where
    F: FnOnce() -> Result<CachedMeta, String>,
{
    get_or_compute_in(global_store(), hash, compute)
}

/// Atomically flushes new metadata entries to `dir/shader-meta.bin` if dirty.
pub(crate) fn save(dir: &Path) {
    save_in(global_store(), dir);
}

fn get_or_compute_in<F>(
    store: &Store,
    hash: &[u8; 32],
    compute: F,
) -> Result<CachedMeta, String>
where
    F: FnOnce() -> Result<CachedMeta, String>,
{
    {
        let guard = store.meta.read().unwrap_or_else(|e| e.into_inner());
        if let Some(meta) = guard.map.get(hash) {
            return Ok(meta.clone());
        }
    }

    let meta = compute()?;

    {
        let mut guard = store.meta.write().unwrap_or_else(|e| e.into_inner());
        let mut inserted = false;
        guard.map.entry(*hash).or_insert_with(|| {
            inserted = true;
            meta.clone()
        });
        if inserted {
            guard.dirty = true;
        }
    }

    Ok(meta)
}

fn save_in(store: &Store, dir: &Path) {
    save_in_with_hook(store, dir, || {});
}

fn save_in_with_hook(store: &Store, dir: &Path, before_rename: impl FnOnce()) {
    let _save_guard = store.save.lock().unwrap_or_else(|e| e.into_inner());

    {
        let guard = store.meta.read().unwrap_or_else(|e| e.into_inner());
        if !guard.dirty {
            return;
        }
    }

    let bytes = {
        let mut guard = store.meta.write().unwrap_or_else(|e| e.into_inner());
        if !guard.dirty {
            return;
        }
        guard.dirty = false;
        encode(&guard.map)
    };

    let n = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    let tmp = dir.join(format!("{FILE_NAME}.tmp-{}-{n}", std::process::id()));
    let target = dir.join(FILE_NAME);
    if let Err(e) = write_atomic(&tmp, &target, &bytes, before_rename) {
        let _ = std::fs::remove_file(&tmp);
        let mut guard = store.meta.write().unwrap_or_else(|e| e.into_inner());
        guard.dirty = true;
        tracing::warn!(path = %target.display(), error = %e, "failed to save shader metadata cache");
    }
}

fn write_atomic(
    tmp: &Path,
    target: &Path,
    bytes: &[u8],
    before_rename: impl FnOnce(),
) -> std::io::Result<()> {
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(tmp, bytes)?;
    before_rename();
    std::fs::rename(tmp, target)?;
    Ok(())
}

fn load_from(path: &Path) -> Option<MetaMap> {
    let bytes = std::fs::read(path).ok()?;
    decode(&bytes)
}

fn encode_resource_type(out: &mut Vec<u8>, kind: ResourceType) {
    match kind {
        ResourceType::Srv => out.push(0),
        ResourceType::Sampler => out.push(1),
        ResourceType::Cbv => out.push(2),
        ResourceType::Uav => out.push(3),
        ResourceType::Other(x) => {
            out.push(4);
            out.push(x);
        }
    }
}

fn encode_root_type(out: &mut Vec<u8>, root: RootType) {
    match root {
        RootType::Constants => out.push(0),
        RootType::Descriptor => out.push(1),
        RootType::Table => out.push(2),
        RootType::ImmediateSampler => out.push(3),
        RootType::ResourceHeap => out.push(4),
        RootType::SamplerHeap => out.push(5),
        RootType::Other(x) => {
            out.push(6);
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
}

fn encode(map: &MetaMap) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + map.len() * 64);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(map.len() as u32).to_le_bytes());
    for (hash, meta) in map {
        out.extend_from_slice(hash);
        out.push(u8::from(meta.stores));
        out.extend_from_slice(&(meta.used.len() as u32).to_le_bytes());
        for res in &meta.used {
            encode_resource_type(&mut out, res.kind);
            encode_root_type(&mut out, res.root);
            out.extend_from_slice(&res.table_offset_dwords.to_le_bytes());
            out.extend_from_slice(&res.lower_bound.to_le_bytes());
            out.extend_from_slice(&res.stride.to_le_bytes());
            out.push(res.dwords);
            out.push(res.user_data);
            out.push(res.slot);
        }
    }
    out
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let (head, rest) = self.0.split_at_checked(n)?;
        self.0 = rest;
        Some(head)
    }

    fn u8(&mut self) -> Option<u8> {
        Some(self.take(1)?[0])
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn hash(&mut self) -> Option<[u8; 32]> {
        self.take(32)?.try_into().ok()
    }

    fn resource_type(&mut self) -> Option<ResourceType> {
        match self.u8()? {
            0 => Some(ResourceType::Srv),
            1 => Some(ResourceType::Sampler),
            2 => Some(ResourceType::Cbv),
            3 => Some(ResourceType::Uav),
            4 => Some(ResourceType::Other(self.u8()?)),
            _ => None,
        }
    }

    fn root_type(&mut self) -> Option<RootType> {
        match self.u8()? {
            0 => Some(RootType::Constants),
            1 => Some(RootType::Descriptor),
            2 => Some(RootType::Table),
            3 => Some(RootType::ImmediateSampler),
            4 => Some(RootType::ResourceHeap),
            5 => Some(RootType::SamplerHeap),
            6 => Some(RootType::Other(self.u32()?)),
            _ => None,
        }
    }
}

fn decode(bytes: &[u8]) -> Option<MetaMap> {
    const MIN_ENTRY_BYTES: usize = 32 + 1 + 4;
    const MIN_RESOURCE_BYTES: usize = 1 + 1 + 4 + 4 + 4 + 1 + 1 + 1;

    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    if r.u32()? != VERSION {
        return None;
    }

    let count = r.u32()? as usize;
    if count.checked_mul(MIN_ENTRY_BYTES)? > r.0.len() {
        return None;
    }

    let mut map = HashMap::with_capacity(count.min(256));
    for _ in 0..count {
        let hash = r.hash()?;
        let stores = match r.u8()? {
            0 => false,
            1 => true,
            _ => return None,
        };
        let used_len = r.u32()? as usize;
        if used_len.checked_mul(MIN_RESOURCE_BYTES)? > r.0.len() {
            return None;
        }
        let mut used = Vec::with_capacity(used_len.min(256));
        for _ in 0..used_len {
            used.push(UsedResource {
                kind: r.resource_type()?,
                root: r.root_type()?,
                table_offset_dwords: r.u32()?,
                lower_bound: r.u32()?,
                stride: r.u32()?,
                dwords: r.u8()?,
                user_data: r.u8()?,
                slot: r.u8()?,
            });
        }
        map.insert(hash, CachedMeta { stores, used });
    }

    r.0.is_empty().then_some(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, mpsc};
    use std::time::Duration;
    use tempfile::TempDir;

    fn sample_resources() -> Vec<UsedResource> {
        let kinds = [
            ResourceType::Srv,
            ResourceType::Sampler,
            ResourceType::Cbv,
            ResourceType::Uav,
            ResourceType::Other(0),
            ResourceType::Other(255),
        ];
        let roots = [
            RootType::Constants,
            RootType::Descriptor,
            RootType::Table,
            RootType::ImmediateSampler,
            RootType::ResourceHeap,
            RootType::SamplerHeap,
            RootType::Other(0),
            RootType::Other(u32::MAX),
        ];
        kinds
            .iter()
            .cycle()
            .zip(roots.iter().cycle())
            .take(8)
            .enumerate()
            .map(|(i, (&kind, &root))| UsedResource {
                kind,
                root,
                table_offset_dwords: i as u32 * 4,
                lower_bound: 10 + i as u32,
                stride: 32 + i as u32,
                dwords: (4 + i) as u8,
                user_data: i as u8,
                slot: (i + 1) as u8,
            })
            .collect()
    }

    #[test]
    fn encode_decode_roundtrip_and_corruption() {
        let mut map = HashMap::new();
        let h1 = [0x11u8; 32];
        let h2 = [0x22u8; 32];
        map.insert(
            h1,
            CachedMeta {
                stores: true,
                used: sample_resources(),
            },
        );
        map.insert(
            h2,
            CachedMeta {
                stores: false,
                used: Vec::new(),
            },
        );

        let bytes = encode(&map);
        let decoded = decode(&bytes).expect("valid encoded map must decode");
        assert_eq!(decoded, map);

        let mut bad_magic = bytes.clone();
        bad_magic[3] = b'0';
        assert_eq!(decode(&bad_magic), None);

        let mut bad_ver = bytes.clone();
        bad_ver[4..8].copy_from_slice(&99u32.to_le_bytes());
        assert_eq!(decode(&bad_ver), None);

        for cut in 0..bytes.len() {
            assert_eq!(decode(&bytes[..cut]), None, "truncated at {cut}");
        }
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert_eq!(decode(&trailing), None);
    }

    #[test]
    fn store_persists_to_disk_and_skips_recompute() {
        let dir = TempDir::new().unwrap();
        let store = Store::load(dir.path());
        assert!(!store.meta.read().unwrap().dirty);

        let hash = [0xABu8; 32];
        let expected = CachedMeta {
            stores: true,
            used: sample_resources(),
        };
        let runs = AtomicU64::new(0);

        let got = get_or_compute_in(&store, &hash, || {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(expected.clone())
        })
        .unwrap();
        assert_eq!(got, expected);
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert!(store.meta.read().unwrap().dirty);

        let got2 = get_or_compute_in(&store, &hash, || {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(CachedMeta {
                stores: false,
                used: Vec::new(),
            })
        })
        .unwrap();
        assert_eq!(got2, expected);
        assert_eq!(runs.load(Ordering::Relaxed), 1);

        let err_hash = [0xCDu8; 32];
        assert!(get_or_compute_in(&store, &err_hash, || Err("boom".to_owned())).is_err());

        save_in(&store, dir.path());
        assert!(dir.path().join(FILE_NAME).is_file());
        assert!(!store.meta.read().unwrap().dirty);

        save_in(&store, dir.path());
        assert!(!store.meta.read().unwrap().dirty);

        let reloaded = Store::load(dir.path());
        let got3 = get_or_compute_in(&reloaded, &hash, || {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(CachedMeta {
                stores: false,
                used: Vec::new(),
            })
        })
        .unwrap();
        assert_eq!(got3, expected);
        assert_eq!(runs.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn concurrent_saves_publish_in_order_and_never_overwrite_newer_entries() {
        let dir = TempDir::new().unwrap();
        let store = Arc::new(Store::empty());

        let hash_a = [0xAA; 32];
        let val_a = CachedMeta {
            stores: false,
            used: sample_resources(),
        };
        get_or_compute_in(&store, &hash_a, || Ok(val_a.clone())).unwrap();

        let (at_rename_tx, at_rename_rx) = mpsc::channel();
        let (allow_rename_tx, allow_rename_rx) = mpsc::channel();

        let store_worker = Arc::clone(&store);
        let worker_dir = dir.path().to_path_buf();
        let worker = std::thread::spawn(move || {
            save_in_with_hook(&store_worker, &worker_dir, || {
                at_rename_tx.send(()).unwrap();
                allow_rename_rx.recv().unwrap();
            });
        });

        at_rename_rx.recv().unwrap();

        let hash_b = [0xBB; 32];
        let val_b = CachedMeta {
            stores: true,
            used: sample_resources().into_iter().take(2).collect(),
        };
        get_or_compute_in(&store, &hash_b, || Ok(val_b.clone())).unwrap();

        let (shutdown_done_tx, shutdown_done_rx) = mpsc::channel();
        let store_shutdown = Arc::clone(&store);
        let shutdown_dir = dir.path().to_path_buf();
        let shutdown = std::thread::spawn(move || {
            save_in(&store_shutdown, &shutdown_dir);
            shutdown_done_tx.send(()).unwrap();
        });

        assert!(
            shutdown_done_rx
                .recv_timeout(Duration::from_millis(50))
                .is_err()
        );

        allow_rename_tx.send(()).unwrap();
        worker.join().unwrap();
        shutdown.join().unwrap();
        assert!(shutdown_done_rx.recv().is_ok());

        let disk = load_from(&dir.path().join(FILE_NAME)).unwrap();
        assert_eq!(disk.get(&hash_a), Some(&val_a));
        assert_eq!(disk.get(&hash_b), Some(&val_b));
    }
}
