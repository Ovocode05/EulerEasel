#pragma once

#include <iostream>
using namespace std;

enum class ExecutionMode
{
    OneShot,
    Batched,
    IterativeSolver,
    Streaming
};

struct ExplorationContext
{
    bool exploration_allowed;
    int32_t max_trials;
    double acceptable_overhead;
};

struct WorkloadContext
{
    bool result_needed_on_host = true;

    ExecutionMode ex_mode;
    ExplorationContext exp_ctx;
};