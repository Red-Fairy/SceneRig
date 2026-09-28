# CoACD 1.0.7+grase.2

This is the minimal source closure of CoACD 1.0.7 used by GRASE. Its Python/C
API and decomposition defaults are unchanged. The local patches replace the projected coplanar segment/containment arithmetic
in `src/intersection.h` with CDT's existing adaptive orientation predicate, and
restore all original plane distances when a noncoplanar pair falls entirely inside
the rounding tolerance. The latter keeps sign classification consistent with the
distances used for interval interpolation; restoring only one distance could select
a zero denominator. Input coordinates, the 3D
plane/coplanarity tolerances, OpenVDB preprocessing, and decomposition settings
retain their upstream behavior.

## Source provenance

| Source | Revision |
| --- | --- |
| [CoACD 1.0.7](https://github.com/SarahWeiii/CoACD/tree/6700816e59640b33bb6f0681cb6dd4fcf4846179) | `6700816e59640b33bb6f0681cb6dd4fcf4846179` |
| [CDT headers](https://github.com/artem-ogre/CDT/tree/ec03b309fd18102ab1da069f2edf3b37be5d1fb3) | `ec03b309fd18102ab1da069f2edf3b37be5d1fb3` |
| zlib, upstream tag v1.2.11 | `cacf7f1d4e3d44d871b605da3b647f07d718623f` |
| spdlog, upstream tag v1.8.2 | `de0dbfa3596a18cd70a4619b6a9766847a941276` |
| OpenVDB, upstream tag v8.2.0 | `89873d2bd29870cc9f176ed12b3f3a930ca38d1a` |
| oneTBB, upstream tag v2022.0.0 | `0c0ff192a2304e114bc9e6557582dfba101360ff` |

The four native Git revisions were read from the completed reference build's
FetchContent checkouts. Their CMake declarations now use those exact commits.
Boost 1.81.0 retains the upstream release URLs and MD5 checksums unchanged.
`UPSTREAM.json` records these pins, archive hashes, the patched header hash, and
the predicate test hash. CoACD's MIT license, CDT's license, and the adaptive
predicate's notice are retained and included in distribution metadata with distinct
basenames: `LICENSE`, `LICENSE.CDT`, and `LICENSE.predicates`.

The copied closure includes `src`, `public`, the used CMake modules/overrides,
`main.cpp`, the Python package/CLI, and CDT's header-only include directory.
`main.cpp` is needed during CMake configuration even when building only `_coacd`.
Unused `cmake/eigen.cmake`, models, examples, visualizer, CI, and Docker files are
omitted. No prebuilt native library is checked into this source package.

## Build and install

GRASE declares this directory as a noneditable `uv` path dependency, with both
the requirement and constraint pinned to `coacd==1.0.7+grase.2`. Use the project
root's normal `uv sync --locked` workflow. Do not copy a native library into an
existing environment by hand. Other installers must be given this local source
or its built wheel explicitly; `tool.uv.sources` is specific to uv.

The isolated Python build requirements are pinned in `pyproject.toml`. The Linux
build also requires a C/C++ toolchain, Git, and access to the pinned native source
repositories/releases. It retains `WITH_3RD_PARTY_LIBS=ON`, static OpenVDB and
native dependency settings, and upstream OpenMP discovery. CMake checkouts and
compilation use a disposable directory under `/tmp`; the completed library is
written into the wheel's package staging directory before scratch is removed.
Set `CMAKE_GENERATOR=Ninja` to use the pinned Ninja executable; compiler and system
libraries remain build-environment inputs. uv cache keys include native sources
and headers so source edits cannot silently reuse a stale wheel.

From this directory, verify an sdist build and then build a wheel from that archive:

```sh
uv build --sdist --out-dir /tmp/grase-coacd-dist .
CMAKE_GENERATOR=Ninja uv build --wheel --out-dir /tmp/grase-coacd-dist /tmp/grase-coacd-dist/coacd-1.0.7+grase.2.tar.gz
```

This generates a wheel for the local platform. It is not a claim of manylinux
portability or byte-for-byte reproducibility across compilers. Inspect the final
library with `readelf -d`/`ldd` and verify an installed-wheel import after build
scratch deletion. The reference Linux link still requires system `libgomp`,
`libgcc_s`, `libm`, and `libc`; OpenVDB and the other fetched libraries are static.

## Native predicate regression

The retained standalone test exercises the captured failure and synthetic contact,
overlap, disjoint, winding, projection, scale, and degenerate cases. From this
directory:

```sh
g++ -std=c++20 -O2 -I src -I 3rd/cdt/CDT tests/coplanar_predicate_test.cpp -o /tmp/grase-coacd-predicate-test
/tmp/grase-coacd-predicate-test
```

The expected result is `fixture_groups=24 assertions=17503 failures=0`. This test
does not run CoACD decomposition or certify complete scene physics.

The interval regression covers captured IMG and Bridge disjoint pairs, actual thin
intersections, and same-side separation with vertex permutations, argument swapping
and axis permutations:

```sh
g++ -std=c++20 -O2 -I src -I 3rd/cdt/CDT tests/interval_predicate_test.cpp -o /tmp/grase-coacd-interval-test
/tmp/grase-coacd-interval-test
```

Expected: `fixture_groups=6 assertions=1296 failures=0`. Assertions and existing
coplanarity/plane-distance tolerances remain enabled and unchanged.
