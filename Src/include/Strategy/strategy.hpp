#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include "./../../include/magic_enum.hpp"
#include "./../WorkLoadCTX/state.hpp"
#include "./../WorkLoadCTX/workload.hpp"
#include "./../Hardware/hardware_context.hpp"

using namespace std;
class Hardware_filter
{
public:
    inline void apply(
        vector<strategy> &strategies,
        const HardwareContext &hardware)
    {
        if (!hardware.has_gpu)
        {
            for (auto it = strategies.begin(); it != strategies.end(); ++it)
            {
                if (it->device == Device::GPU)
                    it->is_available = false;
            }
        }

        if (!hardware.has_AVX || !hardware.has_openMP)
        {
            for (auto it = strategies.begin(); it != strategies.end(); ++it)
            {
                if (it->arch == Architecture::AVX)
                    it->is_available = false;
            }
        }
    }
};

class StrategyRegister
{
public:
    inline static vector<strategy> create()
    {
        return {
            {Device::CPU, Format::CSR, Architecture::Proc, Kernel::CPU_CSR},
            {Device::CPU, Format::ELL, Architecture::Proc, Kernel::CPU_ELL},
            {Device::CPU, Format::HYB, Architecture::Proc, Kernel::CPU_HYB},
            {Device::CPU, Format::CSR, Architecture::AVX, Kernel::CPU_CSR_AVX},
            {Device::CPU, Format::ELL, Architecture::AVX, Kernel::CPU_ELL_AVX_x4},
            {Device::CPU, Format::ELL, Architecture::AVX, Kernel::CPU_ELL_AVX_x16},
            {Device::CPU, Format::HYB, Architecture::AVX, Kernel::CPU_HYB_AVX},

            {Device::GPU, Format::CSR, Architecture::CUDA, Kernel::GPU_CSR},
            {Device::GPU, Format::ELL, Architecture::CUDA, Kernel::GPU_ELL},
            {Device::GPU, Format::HYB, Architecture::CUDA, Kernel::GPU_HYB}};
    }

    inline void generate(
        vector<strategy> &str,
        const HardwareContext &hardware)
    {
        auto strategies = StrategyRegister::create();
        Hardware_filter().apply(str, hardware);
    }

    inline vector<strategy> get_strategies(HardwareContext &hrd)
    {
        query_hardware_context(hrd);
        checkAVX_support(hrd);
        check_openmp_support(hrd);
        print_hardware_context(hrd);
        StrategyRegister str;
        static vector<strategy> stat_vec = str.create();
        str.generate(stat_vec, hrd);

        return stat_vec;
    }

    inline vector<string> get_strategy_names(vector<strategy> &strategies)
    {
        vector<string> names;
        for (const auto &s : strategies)
        {
            if (!s.is_available)
                continue;
            string name = string("Device: ") + string(magic_enum::enum_name(s.device)) +
                          ", Format: " + string(magic_enum::enum_name(s.format)) +
                          ", Architecture: " + string(magic_enum::enum_name(s.arch)) +
                          ", Kernel: " + string(magic_enum::enum_name(s.kernel));
            names.push_back(name);
        }

        return names;
    }
};