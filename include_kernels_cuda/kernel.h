#include <cstdint>

#include "Array.h"

#define RTE_STRINGIFY2(x) #x
#define RTE_STRINGIFY(x) RTE_STRINGIFY2(x)

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

static std::string find_cuda_dir() {
    for (const char* s: std::array<const char*, 3> {
            std::getenv("CUDA_HOME"),
            std::getenv("CUDA_ROOT"),
            std::getenv("CUDA_PATH")
    }) {
        if (s != nullptr && s[0] != '\0') {
            return s;
        }
    }

    return "/usr/";
}

struct Kernel: kernel_launcher::PragmaKernel {
    Kernel(std::string name, std::string filename, std::vector<kernel_launcher::Value> args = {}):
            kernel_launcher::PragmaKernel(name, find_base_dir() + "/" + filename, args) {}

    kernel_launcher::KernelBuilder build() const override {
        auto builder = kernel_launcher::PragmaKernel::build();

        auto this_dir = find_base_dir();
        if (!this_dir.empty()) {
            builder.compiler_flag("-I" + this_dir + "/include_kernels_cuda/");
            builder.compiler_flag("-I" + this_dir + "/include/");
            builder.compiler_flag("-I" + this_dir + "/external/kernel_float/single_include/");
        }

        builder.compiler_flag("-I" + find_cuda_dir() + "/include/");
        builder.compiler_flag("--std=c++17");

#if USECUDA
        builder.define("USECUDA", "1");
#elif USEHIP
        builder.define("USEHIP", "1");
#endif

        builder.define("RESTRICTKEYWORD", "__restrict__");

#ifdef RTE_USE_CBOOL
        builder.define("RTE_USE_CBOOL", RTE_STRINGIFY(RTE_USE_CBOOL));
#endif

#ifdef RTE_USE_SP
        builder.define("RTE_USE_CBOOL", RTE_STRINGIFY(RTE_USE_SP));
#endif

#ifdef RTE_ACCURACY
        builder.define("RTE_ACCURACY", RTE_STRINGIFY(RTE_ACCURACY));
#endif

        return builder;
    }

};
