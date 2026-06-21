# GraphSwitch — Build Roadmap

A phase-by-phase guide for building this solo, with a clear "done" checkpoint at the
end of each phase so you always know whether you're ready to move on. Time estimates
are ballparks for someone learning Triton and GPU profiling from scratch — treat the
checkpoints as the real gate, not the calendar.

**The core question driving every phase:** as a graph's degree distribution gets more
skewed (a few "hub" nodes with huge degree, most nodes with tiny degree), does a
compiler-generated GPU kernel (Triton) lose to a hand-tuned vendor library (cuSPARSE),
and can you measure exactly why and where?

**A scope note up front:** the original 4-backend plan (serial CPU, OpenMP+AVX,
hand-written CUDA, Triton) is a lot of separate skills to learn solo from zero. This
roadmap drops hand-written CUDA C++ as a requirement — cuSPARSE (called through
PyTorch) already gives you a real, legitimate "hand-tuned vendor" baseline without you
needing to write raw CUDA kernels yourself. The core comparison (compiler-generated vs.
hand-tuned) stays fully intact. Hand-written CUDA is listed as an optional stretch goal
in Phase 2 if you have spare time later — not a blocker.

---

## Phase 0 — Environment & Smoke Tests
*Roughly 3–7 days*

**Goal:** prove every tool in the stack works on a trivial example before writing a
single line of project code.

**Why this phase exists:** broken tooling — wrong CUDA version, a missing compiler, a
permissions error — is the single most common way a solo GPU project stalls for weeks.
Find out now, on "hello world," not three weeks in when you can't tell if it's your
code or your environment.

Steps:
1. If you're on Windows: install WSL2 with an Ubuntu distribution. Triton doesn't ship
   official Windows wheels — WSL2 with GPU passthrough is the standard, well-supported
   path, and simpler than the community Windows forks, which need manual MSVC/PATH
   setup. Already on Linux? Skip ahead.
2. Confirm your driver and CUDA toolkit: `nvidia-smi` should show your RTX 3050;
   `nvcc --version` should work after installing the CUDA toolkit.
3. Set up a Python environment (conda or venv), install PyTorch with CUDA support,
   then `pip install triton`.
4. **Smoke test 1:** `torch.cuda.is_available()` returns True and shows your GPU.
5. **Smoke test 2:** run Triton's own official vector-add tutorial from their docs. If
   it compiles and gives correct output, your whole compiler toolchain works.
6. **Smoke test 3:** install Nsight Compute (bundled with the CUDA toolkit) and run
   `ncu` against that same vector-add script. If you hit a permissions error, it's a
   known, well-documented one-time fix — not a sign your build is broken.
7. **Smoke test 4:** hand-download one tiny matrix (under ~100K nonzeros) from the
   SuiteSparse Matrix Collection as a `.mtx` file, load it with `scipy.io.mmread`,
   convert to CSR.

**Done when:** all four smoke tests pass, in the same environment, on your machine.

**Pitfall:** don't grab a big real-world graph yet. The only goal here is proving the
pipe is open end to end.

---

## Phase 1 — The Slicing Framework
*Roughly 1.5–2 weeks*

**Goal:** a loader that reads a sparse matrix, splits it into row-chunks, and computes
a skew score per chunk.

**Why this phase exists:** this builds your independent variable. Every later phase —
the backend comparison, the mechanism, the router — needs a trustworthy number for
"how hub-heavy is this chunk." Chunking also solves your VRAM limit directly: it lets
you stream matrices bigger than 4GB through the GPU one row-block at a time, which is
a real engineering reason for chunking, not just structure for its own sake.

Steps:
1. Get comfortable with CSR (compressed sparse row) format by hand: three arrays —
   values, column indices, row pointers. Convert one tiny matrix into CSR yourself in
   a notebook before letting scipy do it for you.
2. Write a function that slices a CSR matrix into row-chunks (start with chunks of
   1024 rows).
3. For each chunk, compute the Gini coefficient of its row degrees (nonzeros per row)
   — look up the formula and implement it on a numpy array.
4. Build a synthetic generator: draw a degree sequence from a Zipf distribution with a
   tunable exponent (numpy), build a graph matching that sequence with networkx's
   configuration model, clean up resulting self-loops/multi-edges, convert to CSR.
   This is what lets you dial skew up and down precisely later, rather than relying
   only on whatever real graphs happen to give you.
5. Sanity check: generate one near-uniform matrix and one aggressively skewed matrix
   with the same total nonzero count, and confirm your Gini function reports near-0
   for one and a high value for the other.

**Done when:** any matrix in gives you back a list of chunks plus a per-chunk Gini
score, and your two synthetic sanity-check matrices score where you'd expect.

**Pitfall:** don't skip the synthetic generator to save time — it's what turns this
from "I measured some graphs I found" into "I controlled one variable and measured its
effect," which is the actual experimental design.

---

## Phase 2 — The Kernel Duelists
*Roughly 2.5–3 weeks (the biggest single jump — budget extra time here)*

**Goal:** one correct SpMV computation on a chunk, in two trusted forms: a
CPU/cuSPARSE baseline, and your own Triton kernel.

**Why this phase exists:** you can't claim Triton is faster or slower than anything
until you have a reference you actually trust. This phase also produces the central
phenomenon: does a timing gap between Triton and cuSPARSE really show up as skew
rises?

Steps:
1. CPU ground truth: use scipy.sparse for a plain CSR SpMV, and confirm it matches a
   dense numpy multiply on a tiny hand-checkable example.
2. cuSPARSE baseline: load a chunk as a `torch.sparse_csr_tensor` on GPU and call
   `torch.sparse.mm` against a dense vector — this routes into cuSPARSE under the
   hood, giving you a real vendor-tuned baseline with no C++ required.
3. Work through Triton's official tutorials in order before writing your own kernel:
   vector add, then fused softmax, then matrix multiply. Each teaches a primitive
   (block pointers, masking, reductions) you'll need for SpMV.
4. Write your own Triton SpMV kernel over a chunk's CSR arrays. Validate against the
   CPU baseline on every test chunk — including your two synthetic skew extremes —
   before trusting any timing number.
5. Once SpMV is solid, generalize to SpMM (sparse adjacency × dense feature matrix —
   the GNN-relevant operation).
6. Run the sweep: fixed total nonzero count, skew dialed from flat to aggressively
   Zipfian, timing both backends.

**Done when:** Triton and cuSPARSE both match the CPU baseline numerically on every
test chunk, and you have a first table of time vs. skew for both.

**Optional stretch (not required):** a hand-written CUDA kernel with manual
load-balancing across hub rows, as a third comparison point, if you have time left
after Phase 5.

**Pitfall:** don't trust a timing number from a kernel you haven't checked for
correctness yet — a "fast but wrong" kernel is worse than no result at all, because
it'll quietly poison every plot downstream.

---

## Phase 3 — Hardware Introspection
*Roughly 1.5–2 weeks — protect this phase above all others*

**Goal:** explain the Phase 2 timing gap using real hardware counters.

**Why this phase exists:** "Triton is slower on skewed graphs" is an observation
anyone could make. "Triton is slower because of measured warp divergence and memory
stalls, and the effect crosses a threshold around Gini ≈ G" is a mechanism with a
number attached — that's the actual contribution. If time runs short anywhere, cut
from another phase before this one.

Steps:
1. Run `ncu` manually on one kernel launch first, before automating anything. Find
   two specific metrics in the report: the warp-execution-efficiency metric
   (divergence) and the long-scoreboard-stall metric (memory latency waits). Make
   sure you understand what each number means before scripting around it.
2. Write a small wrapper that calls `ncu` as a subprocess for a given chunk + backend,
   and parses out just those two metrics.
3. Run it across your synthetic skew sweep for both backends.
4. Plot or tabulate skew (x-axis) against both metrics (y-axis), one line per backend,
   and look for where Triton's curve visibly breaks away from cuSPARSE's.

**Done when:** you can state, in one sentence, the specific hardware mechanism and the
rough skew threshold where it kicks in — backed by counters, not impressions.

**Pitfall:** profiling has real overhead. Don't profile every chunk from Phase 2
blindly — sample broadly first to find the rough threshold, then zoom in with denser
samples near it.

---

## Phase 4 — The Router
*Roughly 1–1.5 weeks*

**Goal:** turn the Phase 3 threshold into a rule that picks a backend for each chunk,
before that chunk is sent to the GPU.

**A framing correction, worth internalizing before you build this:** this isn't really
"hot-swapping mid-kernel." A GPU kernel, once launched, runs to completion as the code
it was compiled with — there's no swapping the running binary partway through one
launch. What you're actually building — and it's still genuinely interesting — is a
decision made fresh *before each chunk's launch*, so one matrix can have some chunks
routed to Triton and others to cuSPARSE. That's dynamic and adaptive in the way that
matters; it's just resolved between launches, not inside one. It's also a more honest
claim to make to an interviewer who knows GPU execution models, and an easier, more
achievable target than literal mid-kernel switching.

Steps:
1. Start with the simplest rule possible: a single threshold on the Gini coefficient
   from Phase 3 ("if Gini > G, route to cuSPARSE, else Triton"). Get this working end
   to end before reaching for any ML model.
2. Confirm low overhead: time the feature-extraction + routing decision itself, and
   check it's a small fraction of the actual chunk execution time.
3. Optional refinement: replace the single threshold with a small, interpretable
   decision tree over a few structural features, only once the simple rule already
   works — treat it as a refinement of something you understand, not a black box
   you're hoping helps.
4. Wire it into one function: matrix in → sliced, routed, executed per chunk → result
   reassembled, with a log of what was routed where and why.

**Done when:** a mixed matrix (some uniform regions, some hub-heavy regions) runs
through this function, produces a correct result, and shows a measurable speedup over
always-cuSPARSE and always-Triton.

**Pitfall:** don't chase a slightly better accuracy number by adding features you
can't explain. The router's only job is to faithfully apply the mechanism from Phase
3 — anything more is the "boring generic auto-tuner" trap.

---

## Phase 5 — Packaging
*Roughly 1–1.5 weeks*

**Goal:** turn the working pipeline into the two things your two audiences will
actually read.

**Why this phase exists:** an unpackaged result doesn't help anyone evaluate you. This
is the only phase that's pure communication, not new technical risk.

Steps:
1. Wrap the Phase 4 function in a small CLI (e.g. `graphswitch bench --matrix
   file.mtx`).
2. Write unit tests: backend correctness against the CPU baseline, Gini coefficient on
   known inputs, router decisions at known thresholds.
3. Set up a minimal CI workflow (e.g. GitHub Actions) running those tests on push.
4. Write the tool's README: quickstart, example output, the speedup number.
5. Separately, write a short technical report: your phases as the method section, the
   Phase 3 plot as the central figure, an honest limitations section (one GPU, modest
   matrix sizes, mostly synthetic data plus a handful of real ones).

**Done when:** a stranger can clone the repo and run the CLI without asking you
anything, and a second stranger can read the report and understand the finding
without running any code.

---

## Two audiences, one pipeline

You're not building two projects. The repo from Phase 5 is your software/data-science
deliverable — talk about the engineering decisions (why cuSPARSE instead of
hand-rolled CUDA, why a threshold rule before an ML model, what the tests cover). The
report from Phase 5 is your research deliverable — talk about the mechanism in Phase 3
and the threshold you found. Same evidence, two conversations.
