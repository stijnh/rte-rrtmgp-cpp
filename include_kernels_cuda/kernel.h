#include <cstdint>

#include "kernel_launcher.h"
#include "kernel_launcher/pragma.h"
#include "Array.h"

namespace kernel_launcher {
    template <typename T, int N>
    struct IntoKernelArg<Array_gpu<T, N>> {
    static KernelArg convert(Array_gpu<T, N>& arg) {
        return KernelArg::from_array<T>(arg.ptr(), arg.size());
    }

    static KernelArg convert(const Array_gpu<T, N>& arg) {
        return KernelArg::from_array<const T>(arg.ptr(), arg.size());
    }
};
}

static std::string find_base_dir() {
    auto this_file = std::string(__FILE__);
    auto last_slash = this_file.rfind("/include_kernels_cuda/");

    if (last_slash != std::string::npos) {
        return std::string(this_file).substr(0, last_slash);
    } else {
        return ".";
    }
}

struct Kernel: kernel_launcher::PragmaKernel {
    Kernel(std::string name, std::string filename, std::vector<kernel_launcher::Value> args = {}):
            kernel_launcher::PragmaKernel(name, find_base_dir() + "/" + filename, args) {}

    kernel_launcher::KernelBuilder build() const override {
        auto builder = kernel_launcher::PragmaKernel::build();
        builder.compiler_flag("-std=c++17");

        auto this_dir = find_base_dir();
        if (!this_dir.empty()) {
            builder.compiler_flag("-I" + this_dir + "/include_kernels_cuda/");
            builder.compiler_flag("-I" + this_dir + "/include/");
        }

        builder.define("USECUDA", "1");
        builder.define("RESTRICTKEYWORD", "__restrict__");

#ifdef RTE_USE_CBOOL
        builder.define("RTE_USE_CBOOL", "1");
#endif

#ifdef RTE_USE_SP
        builder.define("RTE_USE_SP", "1");
#endif

        return builder;
    }

};