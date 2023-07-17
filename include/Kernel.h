#include "kernel_launcher.h"

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

struct Kernel: kernel_launcher::PragmaKernel {
    Kernel(std::string name, std::string filename, std::vector<kernel_launcher::Value> args = {}):
        kernel_launcher::PragmaKernel(name, filename, args) {}

    virtual kernel_launcher::KernelBuilder build() const {
        auto builder = kernel_launcher::PragmaKernel::build();
        builder.define("USECUDA", "1");
        builder.define("RESTRICTKEYWORD", "__restrict__");

#ifdef RTE_RRTMGP_USE_CBOOL
        builder.define("RTE_RRTMGP_USE_CBOOL", "1");
#endif

#ifdef RTE_RRTMGP_SINGLE_PRECISION
        builder.define("RTE_RRTMGP_SINGLE_PRECISION", "1");
#endif

        return builder;
    }

};