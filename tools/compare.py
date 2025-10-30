import h5py, numpy as np
import sys
import argparse
import json

def compare(reference, result):
    reference = np.array(reference)
    result = np.array(result)

    diff = reference - result
    rel_diff = np.abs(diff) / np.maximum(np.abs(reference), 1e-7)

    return {
            "shape": reference.shape,
            "mean": np.mean(reference),
            "mean_bias": np.mean(diff),
            "root_mean_sq_error": np.sqrt(np.mean(diff**2)),
            "mean_rel_error": np.mean(rel_diff),
            "mean_abs_error": np.mean(np.abs(diff)),
            "p99_abs_error": np.percentile(np.abs(diff), 99),
    }


def main():
    parser = argparse.ArgumentParser(prog="Compare output")
    parser.add_argument("reference-file")
    parser.add_argument("result-file")
    args = parser.parse_args()

    reference_file = getattr(args, 'reference-file')
    result_file = getattr(args, 'result-file')

    print(f"comparing {reference_file} with {result_file}")

    reference = h5py.File(reference_file)
    result = h5py.File(result_file)

    toa_index = np.argmin(reference["p_lev"][:,0,0])
    sfc_index = np.argmax(reference["p_lev"][:,0,0])

    output = dict()
    datasets = {
            "LW flux": (reference["lw_flux_net"], result["lw_flux_net"]),
            "SW flux": (reference["sw_flux_net"], result["sw_flux_net"]),
            "net flux": (
                reference["lw_flux_net"][...] + reference["sw_flux_net"][...],
                result["lw_flux_net"][...] + result["sw_flux_net"][...],
            )
    }

    for name, (A, B) in datasets.items():
        output[name] = compare(A, B)
        output[name + " (TOA)"] = compare(A[toa_index], B[toa_index])
        output[name + " (surface)"] = compare(A[sfc_index], B[sfc_index])

    print(json.dumps(output, indent=4))

if __name__ == "__main__":
    main()

