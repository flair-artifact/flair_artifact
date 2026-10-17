# Third-party components

The precompiled extension incorporates the components below. Versions follow the build inventory
provided with the binaries. License texts have been checked against the corresponding upstream
releases; their inclusion does not change the license of the original FLAIR implementation.

| Component | Recorded version | License | Versioned upstream source |
|---|---|---|---|
| async_simple | 1.3 | Apache-2.0, with component-specific terms below | [v1.3](https://github.com/alibaba/async_simple/tree/f49395f5e8cf30b4439ac395053e6c833074ed61) |
| Eigen | 5.0.1 | MPL-2.0, with additional compatible third-party notices | [5.0.1](https://gitlab.com/libeigen/eigen/-/tree/5.0.1) |
| CRoaring | 4.3.1 | MIT, elected from the Apache-2.0/MIT dual license | [v4.3.1](https://github.com/RoaringBitmap/CRoaring/tree/v4.3.1) |
| RapidJSON | 1.1.0 | MIT; upstream file also describes third-party components | [v1.1.0](https://github.com/Tencent/rapidjson/tree/v1.1.0) |
| pybind11 | 2.13.6 | BSD-3-Clause | [v2.13.6](https://github.com/pybind/pybind11/tree/v2.13.6) |

NumPy is installed separately for running the drivers (the supplied build inventory records testing
with 2.4.4). Regenerating data also uses separately installed h5py. Neither dependency's distribution
is bundled here; its own installation provides its license notices.

## async_simple

The Apache license and NOTICE are retained unchanged from upstream v1.3:

- `licenses/async_simple-Apache-2.0.txt`: [upstream LICENSE](https://raw.githubusercontent.com/alibaba/async_simple/f49395f5e8cf30b4439ac395053e6c833074ed61/LICENSE).
- `licenses/async_simple-NOTICE.txt`: [upstream NOTICE](https://raw.githubusercontent.com/alibaba/async_simple/f49395f5e8cf30b4439ac395053e6c833074ed61/NOTICE).

The NOTICE names the upstream copyright holder; it is a third-party attribution, not a statement
of the paper authors' affiliation. The following component-specific material is also included:

- `licenses/async_simple-LLVM-LICENSE.txt`: the complete [LLVM license file](https://llvm.org/LICENSE.txt),
  including the Apache-2.0 LLVM exceptions referenced by
  [experimental/coroutine.h](https://raw.githubusercontent.com/alibaba/async_simple/f49395f5e8cf30b4439ac395053e6c833074ed61/async_simple/experimental/coroutine.h).
- `licenses/async_simple-Mutex-MIT.txt`: the copyright and MIT permission notice from
  [coro/Mutex.h](https://raw.githubusercontent.com/alibaba/async_simple/f49395f5e8cf30b4439ac395053e6c833074ed61/async_simple/coro/Mutex.h),
  credited to Lewis Baker. Only the C++ comment markers were removed when copying this notice.
- `licenses/async_simple-Boost-1.0.txt`: the [Boost Software License 1.0](https://www.boost.org/LICENSE_1_0.txt)
  referenced by the upstream NOTICE for its uthread and example components. It is retained for
  completeness; this does not imply that example applications or uthread assembly are linked
  into the supplied extension.

## Eigen

Eigen's version header reports `EIGEN_VERSION_STRING "5.0.1"`. Its legacy WORLD macro remains 3;
the recorded macro sequence 3/5/0/1 is not a separate release named 3.5.0.1.

The Source Form of the incorporated, unmodified Eigen headers can be obtained from the
[5.0.1 source tree](https://gitlab.com/libeigen/eigen/-/tree/5.0.1) or downloaded as a
[source archive](https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.gz).
The Eigen portion remains governed by its upstream licenses. This source-availability statement
concerns Eigen and does not provide FLAIR native implementation source.

The following files are exact copies from that release:

- `licenses/Eigen-MPL-2.0.txt`: [COPYING.MPL2](https://gitlab.com/libeigen/eigen/-/raw/5.0.1/COPYING.MPL2).
- `licenses/Eigen-COPYING.README.txt`: [COPYING.README](https://gitlab.com/libeigen/eigen/-/raw/5.0.1/COPYING.README).
- `licenses/Eigen-BSD.txt`: [COPYING.BSD](https://gitlab.com/libeigen/eigen/-/raw/5.0.1/COPYING.BSD).
- `licenses/Eigen-MINPACK.txt`: [COPYING.MINPACK](https://gitlab.com/libeigen/eigen/-/raw/5.0.1/COPYING.MINPACK).

The additional compatible notices are retained as supplied by upstream; this does not imply
that every optional Eigen module is used by FLAIR.

## CRoaring, RapidJSON and pybind11

- `licenses/CRoaring-Apache-or-MIT.txt` is unchanged from the
  [v4.3.1 LICENSE](https://raw.githubusercontent.com/RoaringBitmap/CRoaring/v4.3.1/LICENSE).
  The MIT alternative is elected for this distribution.
- `licenses/RapidJSON-copyright.txt` is the exact
  [v1.1.0 license.txt](https://raw.githubusercontent.com/Tencent/rapidjson/v1.1.0/license.txt),
  including its copyright and third-party notices. The upstream text mentions `bin/jsonchecker/`;
  that directory is not distributed or linked by the headers-only RapidJSON target used here.
- `licenses/pybind11-BSD-3-Clause.txt` is unchanged from the
  [v2.13.6 LICENSE](https://raw.githubusercontent.com/pybind/pybind11/v2.13.6/LICENSE).

`licenses/SHA256SUMS` records the included notice files. `MANIFEST.sha256` covers the public tree.
Third-party rights are preserved by section 4 of the repository LICENSE.
