# EulerEasel — Complete Roadmap: CLI to Adaptive Sparse Execution Engine

This is the single document covering everything — from environment setup through
the final versioned product. It replaces all earlier roadmap files.

---

## What You Are Building

EulerEasel is a runtime sparse execution switchboard. It takes a CSR matrix,
extracts cheap structural features from it, decides which memory layout and which
execution backend will handle each chunk most efficiently, runs the computation,
and explains every decision it made. It exposes a clean Python API so it can slot
directly into a SciPy or CuPy workflow as a custom matvec — meaning existing
iterative solvers (CG, GMRES, BiCGSTAB) can call your engine instead of their
own default sparse multiply, without any other changes.

In one breath: matrix file → CSR → features extracted → layout chosen →
chunks routed to CPU-hybrid, Triton, or cuSPARSE → decisions narrated in
terminal and saved to log → result assembled → speedup reported → pluggable as
a LinearOperator into any solver that accepts one.

**Who this is for:** people like you — a year or two into CS, no prior GPU or
parallel programming background, who have a sparse matrix and want it to run
fast without becoming an HPC engineer first. Every phase should leave the tool
a little more transparent and a little less intimidating. The "done when" line
at the end of each phase is your real minimum bar — hit it and move on.

---

## Full Tech Stack

- **Python 3.10+** — orchestration, CLI, feature extraction, SciPy/CuPy binding
- **C++ (C++17)** — hand-written CSR, ELLPACK-R, hybrid CSR-ELL kernels
- **AVX2 / AVX-512 intrinsics** — SIMD vectorization inside the ELLPACK half
- **OpenMP** — multi-core parallelism across row blocks
- **pybind11** — binding C++ kernels into Python as the CPU backend
- **Triton DSL** — compiler-generated GPU SpMV/SpMM kernel
- **CUDA / cuSPARSE** — vendor-tuned GPU baseline via `torch.sparse.mm`
- **PyTorch** — tensor management, GPU memory, cuSPARSE access
- **CuPy** — GPU-resident LinearOperator for solver integration
- **NVIDIA Nsight Compute (ncu)** — hardware counter profiling
- **SciPy** — CSR loading, ground-truth SpMV, LinearOperator interface
- **NetworkX** — synthetic Zipf graph generation
- **SuiteSparse Matrix Collection** — real-world test matrices
- **GitHub Actions** — CI for correctness tests across backends

---

## Research Positioning

### The problem this solves

No single sparse format is best across matrices or hardware. CSR is the universal
default but leaves large performance on the table for structured or skewed
matrices. Format-switching systems exist (SMAT, SMATER, Morpheus-Oracle,
Auto-SpMV, DyLaClass) but each solves only a slice of the problem: some switch
formats, some switch kernels, some model overhead, almost none do all three
together at chunk-local granularity, and none combine this with an explainable
routing log that tells a non-expert what happened and why.

### Your research gaps and edge

| Claim | What already exists | Your edge |
|---|---|---|
| Whole-matrix format switching | SMAT, SMATER, Morpheus-Oracle | You do it with explainability, low overhead, and chunk-local granularity |
| Overhead-aware selection | Zhou et al. 2020 (one paper) | You make overhead a first-class runtime metric, not an afterthought |
| Stable "most suitable" labels | DyLaClass (active gap as of 2024) | You predict the most robust backend, not the noisiest "fastest" label |
| Local irregularity handling | Implied by block-partitioning work | You explicitly route different chunks of the same matrix differently |
| Triton on graph-skewed SpMV | Completely absent from literature | Your Phase 3 is the only direct measurement of this |
| SciPy/CuPy LinearOperator hook | Not in any format-switching library | You plug directly into existing solver pipelines with no user changes |

### What you are not competing with

You are not replacing PETSc, Ginkgo, or any solver stack. They solve Ax = b.
You make the SpMV that any such solver calls thousands of times per solve run as
fast as structurally possible, and you explain each decision in plain language.
These are complementary positions, not competing ones.

### Novelty statement (one sentence for interviews or a paper abstract)

EulerEasel is a chunk-local, overhead-aware sparse execution switchboard that
jointly selects memory layout and execution backend per matrix block, exposes
every routing decision with a human-readable reason, and plugs into standard
Python solver interfaces — a combination no existing format-switching or
auto-tuning library currently provides.

---

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

***This is a strategic turning point based on the size of the matrix graph reordering and graph partitioning will be done.***

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

**Looking ahead:** Phase 6 uses PETSc separately, only to generate additional
realistic test matrices for this router to be evaluated against — it does not
change what this router does or when it runs.

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
   which backend handled a given chunk.
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

## Phase 6 — Layout Switching and the Decision Layer (Version 1 Extension)
*Roughly 2–3 weeks*

**Goal:** extend Phase 4's backend router with a full layout-switching layer —
so the tool doesn't just pick a backend, it also picks the best memory format
for each chunk before execution, using a lightweight, explainable model.

**Why this phase exists and why it's novel:** Phase 4 routes based on one signal
(Gini skew) to pick an execution backend. But backend and format are two
separate decisions and the best format often depends on structural properties
beyond skew — row-length variance, diagonal density, block structure, matrix
bandwidth. Prior work proves that no single format wins across matrices
(CSR, ELL, SELL, DIA, hybrid each dominate on different inputs), but most
existing auto-tuners either pick format OR kernel, rarely both together, and
almost none expose a human-readable reason for their choice. That explainability
gap is your edge.

The decision layer added here is deliberately not a neural network or XGBoost
stack. It is a small, interpretable decision tree or threshold-rule ensemble —
something where you can trace exactly which feature triggered which branch and
explain it out loud. Explainability is a deliberate design goal, not a
simplification forced by time.

Steps:
1. Add four or five cheap structural features to your Phase 1 loader, computed
   once per chunk alongside Gini: row-length variance (how uneven the nonzero
   distribution is beyond skew alone), matrix bandwidth (how far off-diagonal
   the nonzeros reach), diagonal density (fraction of diagonal entries that are
   nonzero, relevant for DIA format), and a simple blockiness check (are
   nonzeros clustered in dense sub-blocks). All of these are O(nnz) to compute
   and should take negligible time relative to the SpMV itself.
2. Add two internal layout formats beyond what Phase 1.5 already built: SELL
   (sliced ELLPACK, which groups rows into slices and pads only within a slice,
   reducing waste for moderately skewed input) and DIA (diagonal format, which
   wins for structured PDE matrices where most nonzeros sit on a small number of
   diagonals). Implement conversion from CSR to each, measure conversion time
   explicitly, and log it.
3. Build a timing database: for each (matrix chunk, format, backend) triple you
   test, log the measured time into a lightweight JSON store. This becomes your
   training data for the decision model.
4. Train a small decision tree (scikit-learn, max depth 4-5) on your timing
   database, predicting "most suitable format" from the five features. Use
   "most suitable" (the format that is consistently fast) as the label, not
   "fastest on this one run" — the latter is noisy and produces brittle models.
   This specific distinction is called out as an active research gap in DyLaClass
   (2024) and is part of your research positioning.
5. Add conversion-cost awareness to the routing policy: the tool should only
   convert from CSR to a different format if the predicted speedup from that
   format over multiple repeated SpMV calls exceeds the one-time conversion
   cost. Track this explicitly — if a matrix will only be used once, CSR wins
   by default even if SELL would be faster per-multiply.
6. Update the Phase 5 log to include the chosen format and the specific feature
   values that triggered the choice, in plain language.

**Done when:** for a mixed test set, the tool picks a (format, backend) pair per
chunk, logs the reason, and the end-to-end time (including conversion cost) beats
always-CSR-with-best-backend by a measurable margin on matrices that get repeated
SpMV calls.

**Pitfall:** don't treat the decision model as the contribution — the combination
of format selection + backend selection + overhead accounting + explainability in
one lightweight tool is the contribution. The model itself is a means, not the
point.

---

## Phase 7 — GPU Backend Extension and Solver Integration (Version 2)
*Roughly 2–3 weeks*

**Goal:** add runtime confidence scoring to the router, add lazy conversion
(defer expensive format conversions until the tool is sure they'll pay off), and
expose the whole engine as a SciPy and CuPy LinearOperator so existing iterative
solvers can use your SpMV as their inner loop with no other changes.

**Why this phase exists:** Phase 6 picks a format and executes. But in a real
solver context, the same matrix is multiplied thousands of times. The tool
should get smarter over repeated calls — starting with a safe default, tracking
whether its initial prediction is confirmed by actual hardware timings, and
switching format lazily only once it's certain the conversion cost has amortized.
That feedback loop between prediction confidence and observed timing is what
"overhead-aware" really means in practice.

The LinearOperator integration is the payoff that turns EulerEasel from a
standalone benchmarking tool into something that sits usefully inside real
scientific software — someone running iterative CG or GMRES gets your adaptive
engine as their SpMV kernel by changing one line of their code.

Steps:
1. Add a confidence score to the routing decision: the decision tree from
   Phase 6 outputs a class prediction, but most implementations also output a
   confidence probability. Log this. If confidence is below a threshold (say
   0.7), default to cuSPARSE rather than making a risky format conversion.
2. Implement lazy conversion: on the first call, use CSR. On the second call,
   if the feature profile suggests a different format would win, convert and
   cache the result. On subsequent calls, use the cached converted matrix. Track
   how many calls it took before the conversion cost was recovered.
3. Implement the SciPy binding: wrap your router as a
   `scipy.sparse.linalg.LinearOperator` subclass where `_matvec` calls your
   adaptive engine. Test it by passing this operator to `scipy.sparse.linalg.cg`
   and timing a full solve against the same solve using scipy's default SpMV.
4. Implement the CuPy binding the same way using
   `cupyx.scipy.sparse.linalg.LinearOperator`. This version keeps vectors
   GPU-resident across the entire iterative solve — your matvec, plus the
   solver's vector arithmetic (dot products, AXPY), never leaves the GPU. This
   is a more honest performance test than the SciPy version, and a more
   meaningful real-world number.
5. Run an end-to-end timing comparison: same matrix, same right-hand side,
   same solver (CG for symmetric, GMRES for non-symmetric), same convergence
   tolerance — once with scipy's default SpMV, once with your engine via the
   LinearOperator hook. Report total solve time, not just per-iteration time.

**Done when:** a real matrix, put through a real iterative solver using your
LinearOperator, converges to the same answer in measurably less wall-clock time
than the same solver using scipy's own SpMV — and the log file explains every
routing decision that happened during the solve.

**Pitfall:** don't report per-iteration speedup if total solve time doesn't
improve — convergence rate and iteration count are controlled by the solver's
math, not your engine, so it's theoretically possible to make each iteration
faster while not reducing the number of iterations needed.

---

## Phase 8 — Chunk-Local Heterogeneous Routing and SpMM (Version 3)
*Roughly 2–3 weeks — the research contribution that prior work hasn't done*

**Goal:** route different chunks of the same matrix to different (format,
backend) pairs based on each chunk's local structure, rather than applying one
global decision to the whole matrix. Add an SpMM path for graph-neural-network
style workloads where the operation is sparse adjacency times a dense feature
matrix rather than sparse times a vector.

**Why this phase exists and why it's novel:** every format-switching system
in the literature makes one decision per matrix. Whole-matrix granularity is
the safe, obvious choice. But real matrices — especially large ones — are
structurally heterogeneous: one region might be dense and nearly diagonal
(suited for DIA), another might have extreme hub rows (better in CSR with
load-balancing), another might be regular enough to vectorize cleanly in SELL.
Treating the whole matrix as one object forces a compromise that nobody wins.
Chunk-local heterogeneous routing — letting different parts of the same matrix
take different paths — is your cleanest, most defensible research novelty claim.
Prior work implies it could help (GPU block-partitioning papers, Im et al.'s
Sparsity framework from 2004) but nobody has built it as a first-class runtime
feature in a user-facing tool.

SpMM (sparse matrix times dense matrix, not a single vector) is also the
operation that makes this relevant to GNN training: the message-passing step
in a GCN layer is exactly SpMM between the adjacency matrix and a dense node-
feature matrix. Adding an SpMM path makes the tool directly usable for GNN
inference workloads, connecting your project to an active, highly-cited domain
without you having to build a GNN framework yourself.

Steps:
1. Verify empirically that your test matrices are actually structurally
   heterogeneous chunk to chunk: plot per-chunk Gini scores for a few large
   real SuiteSparse matrices and check whether different chunks genuinely have
   different structural profiles. If they don't vary, the whole premise of this
   phase is wrong for those matrices — document that finding honestly rather
   than hiding it.
2. Implement the chunk-local decision: each chunk gets its own feature
   extraction and its own (format, backend) choice, independently of every
   other chunk. This requires that format conversions are per-chunk, not
   per-matrix — make sure converted chunks are cached efficiently so you're
   not converting the same chunk on every call.
3. Measure the overhead of per-chunk decisions against per-matrix decisions:
   the feature extraction and routing decision are cheap, but multiplied by
   many chunks they could add up. Report this explicitly — if the overhead
   is negligible relative to computation time, that's a strong result. If it
   isn't, report the breakeven chunk size.
4. Add the SpMM Triton kernel: extend your Phase 2 Triton SpMV to handle a
   dense feature matrix X of shape (N × F) rather than a single vector. The
   routing logic (check skew, pick backend) applies per chunk of the adjacency
   matrix, with X passed as the dense side.
5. Run your full skew-vs-hardware-counter sweep from Phase 3 again, but for
   SpMM, and check whether the Triton breakdown threshold is different for
   SpMM than for SpMV. If it shifts, that's an additional, specific finding.

**Done when:** a single large heterogeneous matrix runs through your tool with
demonstrably different (format, backend) choices on different chunks, each
logged with its reason, and the end-to-end time beats both always-one-format
and always-one-backend alternatives. SpMM produces numerically correct output
validated against the CPU baseline.

**Pitfall:** if real matrices turn out to be structurally homogeneous chunk to
chunk (step 1 above), chunk-local routing produces no gain and your honest
finding is "whole-matrix granularity is sufficient for these inputs." That's a
real result, not a failure — it tells you something true about when local
routing matters. Don't manufacture a gain that isn't there.

---

## Phase 9 — Packaging and Final Report
*Roughly 1.5–2 weeks*

**Goal:** turn the complete pipeline into two deliverables: a clean, installable
tool and a short technical write-up that can be read as a standalone document.

Steps:
1. Clean CLI covering all three version capabilities: `EulerEasel bench`,
   `EulerEasel solve`, `EulerEasel analyze` (the analysis command prints the
   per-chunk structural profile before running anything, so a user can inspect
   their matrix before committing to a solve).
2. Unit tests: correctness of every backend against the CPU baseline, feature
   extraction on known inputs, router decisions at known thresholds, format
   conversion round-trips, LinearOperator integration against scipy's own CG
   on a known-answer test problem.
3. CI on GitHub Actions.
4. README with a quickstart, a real terminal log example, and the end-to-end
   solver timing comparison as the headline result.
5. Technical write-up: frame the three versions as V1/V2/V3 with a clear
   statement of what each adds. The central figure is the Phase 3 skew-vs-
   hardware-counter plot. The secondary figure is the chunk-local routing
   result from Phase 8. The limitations section is honest about matrix size,
   GPU model, and the fact that your decision model was trained on synthetic
   and SuiteSparse matrices, not arbitrary real-world inputs.

**Done when:** a stranger clones the repo, runs the CLI on a sample matrix,
and gets back a correct result plus a human-readable log and a speedup number,
without asking you anything. The write-up stands on its own without you in the
room to explain it.

---

## Version Summary

- **Version 1 (Phases 0–6):** CPU-only, whole-matrix routing across three
  backends and multiple formats, explainable decisions, SciPy integration.
  This is a complete, presentable, self-contained product.

- **Version 2 (Phase 7):** GPU backend, runtime confidence, lazy conversion,
  overhead accounting, CuPy LinearOperator, end-to-end iterative solver timing.
  This is the research and engineering extension that puts the tool inside real
  solver workflows.

- **Version 3 (Phase 8):** chunk-local heterogeneous routing, SpMM path for
  GNN-style workloads. This is the novel research contribution that has no
  direct precedent in the format-switching literature.

Each version is independently shippable. If you finish Version 1 and time runs
out, you have a complete project. If you reach Version 3, you have a credible
systems research contribution built on top of it.
