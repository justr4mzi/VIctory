use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::Converter;

/// Identifies one conversion result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey(blake3::Hash);

impl CacheKey {
    pub fn new(converter: &dyn Converter, input: &[u8]) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(converter.id().as_bytes());
        h.update(&[0]);
        h.update(&converter.version().to_le_bytes());
        h.update(input);
        CacheKey(h.finalize())
    }

    pub fn to_hex(&self) -> String {
        self.0.to_hex().to_string()
    }
}

/// Content-addressed store under the user's cache directory.
#[derive(Debug, Clone)]
pub struct Cache {
    root: PathBuf,
}

impl Cache {
    /// Opens (creating) a cache at `root`. Refuses locations inside the repo.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        victory_core::ensure_outside_repo(&root)?;
        std::fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
        Ok(Cache { root })
    }

    pub fn path_for(&self, key: &CacheKey) -> PathBuf {
        let hex = key.to_hex();
        self.root.join(&hex[..2]).join(&hex[2..])
    }

    /// Returns the cached output for `input`, converting it first if needed.
    pub fn get_or_convert(&self, converter: &dyn Converter, input: &[u8]) -> Result<PathBuf> {
        let key = CacheKey::new(converter, input);
        let path = self.path_for(&key);
        if path.is_file() {
            tracing::trace!(converter = converter.id(), key = %key.to_hex(), "cache hit");
            return Ok(path);
        }
        let output = converter
            .convert(input)
            .with_context(|| format!("converter {} v{}", converter.id(), converter.version()))?;
        let dir = path.parent().expect("hashed path has a parent");
        std::fs::create_dir_all(dir)?;
        // Write-then-rename so readers never see partial output.
        let tmp = tempfile_in(dir);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        let written = file.write_all(&output);
        drop(file);
        let published = written.and_then(|()| std::fs::rename(&tmp, &path));
        if let Err(error) = published {
            let _ = std::fs::remove_file(&tmp);
            // A concurrent writer may have published while we converted.
            // Windows can refuse replacement while a reader holds that file.
            if !same_output(&path, &output).unwrap_or(false) {
                return Err(error.into());
            }
        }
        Ok(path)
    }
}

fn same_output(path: &Path, output: &[u8]) -> io::Result<bool> {
    let mut published = File::open(path)?;
    let metadata = published.metadata()?;
    if !metadata.is_file() || metadata.len() != output.len() as u64 {
        return Ok(false);
    }
    let mut buffer = [0; 8192];
    for expected in output.chunks(buffer.len()) {
        let actual = &mut buffer[..expected.len()];
        published.read_exact(actual)?;
        if actual != expected {
            return Ok(false);
        }
    }
    let mut extra = [0];
    Ok(published.read(&mut extra)? == 0)
}

fn tempfile_in(dir: &Path) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    dir.join(format!(".tmp-{}-{n}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Upper {
        version: u32,
        runs: AtomicUsize,
    }

    impl Converter for Upper {
        fn id(&self) -> &'static str {
            "test.upper"
        }
        fn version(&self) -> u32 {
            self.version
        }
        fn convert(&self, input: &[u8]) -> Result<Vec<u8>> {
            self.runs.fetch_add(1, Ordering::Relaxed);
            Ok(input.to_ascii_uppercase())
        }
    }

    #[test]
    fn converts_once_then_hits() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let a = cache.get_or_convert(&conv, b"abc").unwrap();
        let b = cache.get_or_convert(&conv, b"abc").unwrap();
        assert_eq!(a, b);
        assert_eq!(std::fs::read(&a).unwrap(), b"ABC");
        assert_eq!(conv.runs.load(Ordering::Relaxed), 1);
    }

    #[cfg(windows)]
    #[test]
    fn an_open_concurrent_cache_entry_wins_only_if_identical() {
        use std::os::windows::fs::OpenOptionsExt;
        use std::sync::{Mutex, mpsc};
        use std::time::Duration;

        struct Paused {
            inner: Upper,
            entered: mpsc::Sender<()>,
            resume: Mutex<mpsc::Receiver<()>>,
        }

        impl Converter for Paused {
            fn id(&self) -> &'static str {
                self.inner.id()
            }

            fn version(&self) -> u32 {
                self.inner.version()
            }

            fn convert(&self, input: &[u8]) -> Result<Vec<u8>> {
                self.entered.send(())?;
                self.resume
                    .lock()
                    .unwrap()
                    .recv_timeout(Duration::from_secs(10))?;
                self.inner.convert(input)
            }
        }

        let publish = |winning_bytes: &[u8]| {
            let tmp = tempfile::tempdir().unwrap();
            let cache = Cache::open(tmp.path().join("cache")).unwrap();
            let (entered, converting) = mpsc::channel();
            let (resume, resumed) = mpsc::channel();
            let delayed = Paused {
                inner: Upper {
                    version: 1,
                    runs: AtomicUsize::new(0),
                },
                entered,
                resume: Mutex::new(resumed),
            };
            let winner = Upper {
                version: 1,
                runs: AtomicUsize::new(0),
            };
            std::thread::scope(|scope| {
                let pending = scope.spawn(|| cache.get_or_convert(&delayed, b"abc"));
                converting.recv_timeout(Duration::from_secs(10)).unwrap();

                let published = cache.get_or_convert(&winner, b"abc").unwrap();
                std::fs::write(&published, winning_bytes).unwrap();
                let guard = std::fs::OpenOptions::new()
                    .read(true)
                    .share_mode(1)
                    .open(&published)
                    .unwrap();
                resume.send(()).unwrap();
                let result = pending.join().unwrap();
                assert_eq!(std::fs::read(&published).unwrap(), winning_bytes);
                assert_eq!(
                    std::fs::read_dir(published.parent().unwrap())
                        .unwrap()
                        .count(),
                    1
                );
                drop(guard);
                result.map(|path| assert_eq!(path, published))
            })
        };
        assert!(publish(b"XYZ").is_err(), "different bytes");
        assert!(publish(b"ABCD").is_err(), "extra bytes");
        publish(b"ABC").expect("identical output was published");
    }

    #[test]
    fn failed_publication_removes_the_staging_file() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let path = cache.path_for(&CacheKey::new(&conv, b"abc"));
        std::fs::create_dir_all(&path).unwrap();

        assert!(cache.get_or_convert(&conv, b"abc").is_err());
        assert!(path.is_dir());
        assert_eq!(
            std::fs::read_dir(path.parent().unwrap()).unwrap().count(),
            1
        );
    }

    #[test]
    fn version_bump_invalidates() {
        let v1 = Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let v2 = Upper {
            version: 2,
            runs: AtomicUsize::new(0),
        };
        assert_ne!(CacheKey::new(&v1, b"x"), CacheKey::new(&v2, b"x"));
    }

    #[test]
    fn refuses_cache_inside_repo() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("cache");
        assert!(Cache::open(inside).is_err());
    }
}
