# Platform support matrix

mangrove-search is **Linux-only** by design (relies on io_uring + OpenMP +
roaring CRoaring + xxhash). Tested combinations below — others may work
but aren't part of the supported set.

## Officially supported

| Distro              | Version  | Notes                              |
| ------------------- | -------- | ---------------------------------- |
| Debian              | 12 / 13  | Reference target (matches Docker)  |
| Ubuntu              | 22.04 LTS, 24.04 LTS | Most common in K8s nodes |
| Rocky Linux / RHEL  | 9        | Needs `dnf install` equivalents    |
| Arch Linux          | rolling  | Latest liburing + libroaring       |
| Alpine              | 3.18+    | Use `apk add` (musl libc)          |

## Architecture

- **x86_64** — primary target, tested on EC2 c5/c6i, GCP n2.
- **aarch64** — works (build server is ARM Linux). Same code path.

Windows, macOS : not supported. WSL2 will probably work but isn't tested.

## Dependencies

### Required at build time

| Library                | Min version | Notes                            |
| ---------------------- | ----------- | -------------------------------- |
| gcc / clang            | C11 capable | (gcc 9+, clang 12+)              |
| liburing               | 2.1+        | We polyfill `_data64` for 2.1.   |
| libroaring (CRoaring)  | 0.9+        | Debian: `libroaring-dev`         |
| libxxhash              | 0.8+        | Debian: `libxxhash-dev`          |
| libomp                 | any         | OpenMP runtime                   |
| python3                | 3.10+       | For SDK + tests                  |

### Apt one-liners

**Debian/Ubuntu** :
```bash
apt-get install -y --no-install-recommends \
    gcc make pkg-config \
    libroaring-dev liburing-dev libxxhash-dev libomp-dev \
    python3 python3-numpy
```

**Rocky / RHEL 9** (after `dnf install epel-release`) :
```bash
dnf install -y \
    gcc make pkgconf-pkg-config \
    croaring-devel liburing-devel xxhash-devel libomp-devel \
    python3 python3-numpy
```

**Alpine 3.18+** (musl) :
```bash
apk add --no-cache \
    gcc make pkgconf libc-dev \
    croaring-dev liburing-dev xxhash-dev libomp-dev \
    python3 py3-numpy
```

**Arch** :
```bash
pacman -S --noconfirm \
    gcc make pkgconf \
    croaring liburing xxhash openmp \
    python python-numpy
```

## Known compatibility quirks

### liburing < 2.2 missing `io_uring_sqe_set_data64`

Ubuntu 22.04 ships liburing 2.1, which doesn't expose `_data64` helpers.
We use `io_uring_sqe_set_data(sqe, (void*)(uintptr_t)val)` everywhere
instead — works on 2.1 through 2.5+. No action needed by the user.

### Alpine + musl + io_uring

io_uring works on musl from Alpine 3.18 (kernel 5.15+ in the host).
On older Alpine you'll see runtime EINVAL from `io_uring_queue_init` —
upgrade to current.

### RSS reporting on cgroup v1

Older kernels (RHEL 7 vintage) report VmRSS differently under cgroup v1
memory accounting. Our memleak test parses `/proc/<pid>/status:VmRSS`
which works on all current kernels. If you see weird numbers, check your
cgroup version with `mount | grep cgroup`.

## Validation procedure for a new distro

```bash
# 1. Build smoke
make clean && make
./rpforest 2>&1 | head -3   # should print usage

# 2. Unit tests
python3 tests/test_core.py

# 3. End-to-end on SIFT 1M (download once)
mkdir -p sift && cd sift
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar xf sift.tar.gz --strip-components=1
cd ..
python3 scripts/test_live_autostore.py
python3 scripts/test_crash_recovery.py
```

All three should pass without errors. If anything fails, file an issue
with the distro/version + error log.
