# GraphSwitch — Complete Roadmap: CLI to Adaptive Engine

This is the consolidated, start-to-finish version of the plan. It folds in the CPU
deep-dive, the three-way router, structured logging, and PETSc as a fallback for
genuinely hard matrices. It replaces the earlier roadmap file — this is the only
document you need going forward.

**Who this is actually for:** not HPC veterans, not an interview panel — people
like you, a year or two into learning this, who have a sparse matrix and no real
idea yet what GPU or parallel programming involves. Every phase below should
leave the tool a little more transparent and a little less intimidating to that
person, including future-you. If a phase starts to feel like a wall, the "done
when" line is your real minimum bar — hit that, move on, even if it isn't
perfect. A tool that explains itself is the whole point here, more than a tool
that's merely fast.

**What you're building, in one breath:** matrix file → CSR → split into chunks →
check structure (skew and symmetry) → route each chunk to CPU-hybrid, Triton, or
cuSPARSE — or, if the whole matrix is structurally hard, hand the solve to PETSc
→ narrate every decision live in the terminal and save it to a log file → stitch
results back together → report what happened and how much faster it was.

---

## Phase 0 — Environment & Smoke Tests
*Roughly 3–7 days*

**Goal:** prove every tool in the stack works on a trivial example before writing
a single line of project code.

**Why this phase exists:** broken tooling — wrong CUDA version, a missing
compiler, a permissions error — is the single most common way a solo GPU project
stalls for weeks. Find out now, on "hello world," not three weeks in.

Steps:
1. If you're on Windows: install WSL2 with Ubuntu. Triton has no official Windows
   wheels — WSL2 with GPU passthrough is the standard path. Already on Linux?
   Skip ahead.
2. Confirm driver and CUDA toolkit: `nvidia-smi` should show your GPU; `nvcc
   --version` should work.
3. Set up a Python environment, install PyTorch with CUDA support, then `pip
   install triton`.
4. **Smoke test 1:** `torch.cuda.is_available()` returns True.
5. **Smoke test 2:** run Triton's official vector-add tutorial. If it compiles
   and gives correct output, your compiler toolchain works.
6. **Smoke test 3:** install Nsight Compute and run `ncu` against that same
   script. A permissions error on first try is a known, one-time fix, not a sign
   anything's broken.
7. **Smoke test 4:** hand-download one tiny matrix (under ~100K nonzeros) from
   the SuiteSparse Matrix Collection, load it with `scipy.io.mmread`, convert to
   CSR.

**Done when:** all four smoke tests pass in the same environment.

**Pitfall:** don't grab a big real-world graph yet — just prove the pipe is open.

---

## Phase 1 — The Slicing Framework
*Roughly 1.5–2 weeks*

**Goal:** a loader that reads a sparse matrix, splits it into row-chunks, and
computes a skew score per chunk.

**Why this phase exists:** this builds your independent variable. Every later
phase needs a trustworthy number for "how hub-heavy is this chunk." Chunking
also solves any VRAM limit directly — it lets you stream matrices bigger than
your GPU's memory through one row-block at a time.

Steps:
1. Get comfortable with CSR by hand: values, column indices, row pointers.
   Convert one tiny matrix into CSR yourself before letting scipy do it.
2. Write a function that slices a CSR matrix into row-chunks (start with 1024
   rows per chunk).
3. For each chunk, compute the Gini coefficient of its row degrees.
4. Build a synthetic generator: draw a degree sequence from a Zipf distribution
   with a tunable exponent, build a graph matching it with networkx's
   configuration model, clean up self-loops, convert to CSR. This is what lets
   you dial skew precisely, rather than relying on whatever real graphs hand you.
5. Sanity check: one near-uniform matrix, one aggressively skewed matrix, same
   total nonzero count — confirm your Gini function reports near-0 and high
   respectively.

**Done when:** any matrix in gives you chunks plus a per-chunk Gini score, and
your synthetic extremes score where you'd expect.

**Pitfall:** don't skip the synthetic generator — it's what makes this a
controlled experiment instead of "I measured some graphs I found."

---

## Phase 1.5 — The CPU Deep Dive (C++)
*Roughly 2.5–4 weeks — open-ended by design*

**Goal:** build and compare CSR, ELLPACK / ELLPACK-R, and hybrid CSR-ELL SpMV
kernels by hand in C++, vectorize with AVX, parallelize with OpenMP, measure
cache behavior, and test graph reordering — all on CPU, before touching cuSPARSE
or Triton.

**Why this phase exists:** numpy/scipy hide the memory layout and access pattern
from you — you call a function and a pre-compiled routine runs somewhere you
can't see. To feel what a cache miss costs, what AVX and OpenMP each buy you,
and why a format choice matters, you need to control the loop and the memory
yourself.

Steps:
1. **Plain CSR in C++:** re-implement the scalar CSR SpMV in C++, over the three
   raw arrays. Validate against your Python/scipy ground truth.
2. **ELLPACK:** pad every row to the matrix's longest row, stored column-major.
   Run it on your synthetic matrices at increasing skew and watch memory use and
   runtime blow up as one hub row gets longer — the most direct, hands-on way to
   feel this project's central problem, no GPU involved.
3. **ELLPACK-R:** add a per-row length array so the kernel skips padded zeros.
   Confirm it's faster on skewed matrices, and make sure you can explain why.
4. **Hybrid CSR-ELL:** split each matrix by a degree threshold — regular rows
   into ELLPACK, hub rows into plain CSR, computed separately. This is a small,
   CPU-side version of the exact idea behind your GPU router later.
5. **AVX vectorization:** hand-vectorize the ELLPACK half with AVX2/AVX-512
   intrinsics (`_mm256_loadu_pd`, `_mm256_fmadd_pd`). ELLPACK's fixed-stride
   layout is what makes this tractable. Compile with `-O3 -march=native` first
   and check whether the compiler already auto-vectorized your scalar loop.
6. **OpenMP:** parallelize across chunks (or row-blocks within a large chunk)
   with `#pragma omp parallel for`. This stacks with AVX rather than competing
   with it — AVX gives SIMD width inside one core, OpenMP gives you more cores.
   Measure speedup as you scale thread count, and notice where it stops scaling
   — for SpMV, that ceiling is usually memory bandwidth, not core count, which is
   itself a useful thing to observe directly.
7. **Cache measurement:** use `perf stat` (or `valgrind --tool=cachegrind` if
   `perf` access is restricted) to measure L1/L2 cache miss rates for CSR vs.
   ELLPACK vs. hybrid. Good practice for trusting hardware counters before doing
   the same with Nsight Compute in Phase 3.
8. **Graph reordering:** implement Reverse Cuthill-McKee to reduce matrix
   bandwidth, and re-run your cache measurements before and after.

**Done when:** for a handful of matrices across the skew spectrum, you have a
table of (format × AVX × OpenMP threads × reordered or not) against runtime and
cache miss rate, and you can explain in your own words why each change helped or
didn't.

**Where this goes next:** wrap the finished hybrid + AVX + OpenMP kernel with
pybind11 (or ctypes, to see the raw FFI boundary) so Python can call it. That
turns a serious, multi-threaded, vectorized CPU kernel into a real third backend
for the Phase 4 router — not just a correctness oracle.

**Pitfall:** don't make all of this production-quality. The goal is
understanding. Once you can explain why ELLPACK blows up on skew, why the hybrid
format and AVX and OpenMP each helped, move to Phase 2.

---

## Phase 2 — The Kernel Duelists
*Roughly 2–2.5 weeks*

**Goal:** one correct SpMV computation on a chunk, on cuSPARSE and on your own
Triton kernel — both checked against the C++ ground truth from Phase 1.5.

**Why this phase exists:** you can't claim Triton is faster or slower than
anything until you trust your reference. This phase also produces the central
phenomenon: does a timing gap between Triton and cuSPARSE actually appear as
skew rises?

Steps:
1. CPU ground truth: already covered in Phase 1.5.
2. cuSPARSE baseline: load a chunk as a `torch.sparse_csr_tensor` on GPU, call
   `torch.sparse.mm` — this routes into cuSPARSE under the hood, no C++ required.
3. Work through Triton's official tutorials in order — vector add, fused
   softmax, matrix multiply — before writing your own kernel.
4. Write your own Triton SpMV kernel over a chunk's CSR arrays. Validate against
   the CPU baseline on every test chunk, including your synthetic skew extremes.
5. Generalize to SpMM (sparse adjacency × dense feature matrix).
6. Run the sweep: fixed nonzero count, skew dialed from flat to aggressively
   Zipfian, timing both backends.

**Done when:** Triton and cuSPARSE both match the CPU baseline numerically, and
you have a first table of time vs. skew for both.

**Pitfall:** don't trust a timing number from a kernel you haven't checked for
correctness — a fast-but-wrong kernel will quietly poison every plot downstream.

---

## Phase 3 — Hardware Introspection
*Roughly 1.5–2 weeks — protect this phase above all others*

**Goal:** explain the Phase 2 timing gap using real hardware counters.

**Why this phase exists:** "Triton is slower on skewed graphs" is an observation
anyone could make. "Triton is slower because of measured warp divergence and
memory stalls, and the effect crosses a threshold around Gini ≈ G" is a mechanism
with a number attached — the actual contribution.

Steps:
1. Run `ncu` manually on one kernel launch first. Find two metrics: warp
   execution efficiency (divergence) and the long-scoreboard-stall metric
   (memory latency). Understand what each means before scripting anything.
2. Write a wrapper that calls `ncu` as a subprocess per chunk/backend, parsing
   out just those two metrics.
3. Run it across your synthetic skew sweep for both backends.
4. Plot skew against both metrics, one line per backend, and find where
   Triton's curve breaks away from cuSPARSE's.

**Done when:** you can state, in one sentence, the specific hardware mechanism
and the rough skew threshold where it kicks in — backed by counters.

**Pitfall:** profiling has real overhead — sample broadly first, then zoom in
near the threshold, rather than profiling every chunk blindly.

---

## Phase 4 — The Router
*Roughly 1–1.5 weeks*

**Goal:** turn the Phase 3 threshold into a rule that picks a backend for each
chunk, before that chunk is executed.

**A framing note:** this isn't "hot-swapping mid-kernel." A GPU kernel runs to
completion as the code it was compiled with — there's no swapping the running
binary partway through one launch. What you're building is a decision made fresh
*before each chunk's launch*, so one matrix can route some chunks to Triton and
others to cuSPARSE. That's dynamic in the way that matters, and a more honest,
achievable claim than literal mid-kernel switching.

**Three backends, not two:** your Phase 1.5 hybrid CPU kernel, once bound via
pybind11, is a real third option — useful especially for chunks small enough
that GPU launch overhead dominates regardless of skew.

Steps:
1. Start with the simplest rule: a single Gini threshold ("if Gini > G, route to
   cuSPARSE, else Triton"). Get this working end to end before any ML model.
   Add the CPU-hybrid path afterward, likely gated on chunk size first.
2. Confirm low overhead: time the feature-extraction + routing decision itself,
   and check it's a small fraction of actual chunk execution time.
3. Optional refinement: replace the single threshold with a small, interpretable
   decision tree over a few structural features, only once the simple rule
   already works.
4. Wire it into one function: matrix in → sliced, routed, executed per chunk →
   result reassembled.

**Looking ahead:** Phase 6 adds one more check upstream of this router — whether
the *whole* matrix is structurally hard enough (non-symmetric, ill-conditioned)
that it shouldn't go through chunk-wise SpMV routing at all, and should be handed
to PETSc instead.

**Done when:** a mixed matrix runs through this function, produces a correct
result, and shows a measurable speedup over always-cuSPARSE, always-Triton, and
always-CPU.

**Pitfall:** don't chase a slightly better number by adding features you can't
explain. The router's only job is faithfully applying the Phase 3 mechanism.

---

## Phase 5 — Observability: Terminal Logs & Log Files
*Roughly 1–1.5 weeks*

**Goal:** make every run narrate itself — live, in the terminal, and afterward,
in a saved file — so nothing the tool does is a black box.

**Why this phase exists:** this is the single biggest thing that makes a tool
approachable for someone who's just starting out. A silent tool that returns a
number forces you to trust it blindly. A tool that prints what it's doing, chunk
by chunk, and why, turns every run into a small lesson — which is exactly the
audience this is for.

Steps:
1. Design the live terminal output on paper first. Something like:
   `[chunk 003] rows 2048-4095 | Gini 0.81 | routed -> cuSPARSE (skew threshold
   exceeded) | 3.2ms`. Decide what belongs live (readable, not a wall of text)
   versus what only goes in the saved file.
2. Build this as one small logging module that every backend reports through —
   not scattered `print()` calls — so the format stays consistent regardless of
   which backend (or PETSc, once Phase 6 exists) handled a given chunk.
3. Write a structured log file at the end of every run — plain JSON is the
   simplest honest choice — recording every chunk's routing decision and reason,
   the timing breakdown, total wall-clock time, and overall speedup versus a
   fixed-backend baseline. This is what makes a run reviewable a week later,
   including by you, after you've forgotten what happened.
4. Add a `--verbose` / `--quiet` flag so live terminal detail can be dialed up or
   down without affecting what the log file captures — the file should always
   record everything.

**Done when:** you can run the tool, watch a human-readable account of every
decision scroll past, and afterward open the log file and reconstruct exactly
what happened without re-running anything.

**Pitfall:** don't let logging slow down the actual computation. Buffer in
memory and flush once at the end, not chunk-by-chunk.

---


## Phase 6 — Packaging
*Roughly 1–1.5 weeks*

**Goal:** turn the working pipeline into something a stranger — or a future,
forgetful version of you — could pick up and use.

**Why this phase exists:** an unpackaged result helps nobody, including you in
six months. This is the only phase that's pure communication, not new technical
risk.

Steps:
1. Wrap everything in a small CLI (e.g. `graphswitch solve matrix.mtx`), making
   sure the Phase 5 logging output and the Phase 6 PETSc hand-off are both
   visible through it, not hidden side paths.
2. Write unit tests: backend correctness against the CPU baseline, Gini
   coefficient and symmetry detection on known inputs, router decisions at known
   thresholds, and at least one test matrix that correctly triggers the PETSc
   path.
3. Set up a minimal CI workflow (e.g. GitHub Actions) running those tests on
   push.
4. Write the README: quickstart, a real example of the terminal log output, the
   measured speedup.
5. Write a short technical note: your phases as the method, the Phase 3 plot as
   the central figure, an honest limitations section.

**Done when:** a stranger can clone the repo, run the CLI on a sample matrix,
and understand exactly what happened and why — without asking you anything.

---

## Closing note

This still works fine as the project you talk about in research or engineering
interviews — nothing about that earlier framing goes away. But the actual point
of building it, from here, is the one you stated: a small, honest, narrated
engine that lets someone like you — or a PhD student who's never touched CUDA —
get a real, working feel for heterogeneous sparse computation quickly, without
either lying to them about what's happening. That's worth building whether or not anyone else ever sees it.
