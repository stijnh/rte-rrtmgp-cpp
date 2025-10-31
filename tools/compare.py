import h5py, numpy as np
import sys
import argparse
import json

def heating_rate(lw_flux_net, sw_flux_net, p_lev):
    # Constants
    g = 9.8    # m/s^2
    cp = 1004  # J/kg/K

    # Determine the number of layers and levels
    num_levels = p_lev.shape[0]
    num_layers = num_levels - 1

    # Initialize arrays for heating rates in each layer
    lw_heating_rate = np.zeros_like(lw_flux_net[:-1])
    sw_heating_rate = np.zeros_like(sw_flux_net[:-1])

    # Calculate the heating rate for each layer
    for i in range(num_layers):
        # Change in net flux across the layer
        delta_lw_flux_net = lw_flux_net[i+1] - lw_flux_net[i]
        delta_sw_flux_net = sw_flux_net[i+1] - sw_flux_net[i]

        # Change in pressure across the layer
        delta_p = p_lev[i+1] - p_lev[i]

        # Calculate heating rate for the layer using the formula
        lw_heating_rate[i] = (g / cp) * (delta_lw_flux_net / delta_p)
        sw_heating_rate[i] = (g / cp) * (delta_sw_flux_net / delta_p)


    # Total heating rate is the sum of longwave and shortwave
    total_heating_rate = lw_heating_rate + sw_heating_rate

    # The heating rates are in K/s. To convert to K/day, multiply by the number of seconds in a day.
    seconds_per_day = 86400
    lw_heating_rate_k_per_day = lw_heating_rate * seconds_per_day
    sw_heating_rate_k_per_day = sw_heating_rate * seconds_per_day
    total_heating_rate_k_per_day = total_heating_rate * seconds_per_day

    return total_heating_rate_k_per_day

def compare(reference, result):
    reference = np.array(reference)
    result = np.array(result)

    diff = reference - result
    abs_diff = np.abs(diff)
    rel_diff = np.abs(diff) / np.maximum(np.abs(reference), 1e-7)

    return {
            "shape": reference.shape,
            "mean": np.mean(reference),
            "mean_bias": np.mean(diff),
            "root_mean_sq_error": np.sqrt(np.mean(diff**2)),
            "mean_rel_error": np.mean(rel_diff),
            "mean_abs_error": np.mean(abs_diff),
            "max_abs_error": np.amax(abs_diff),
            "max_rel_error": np.amax(rel_diff),
            "p99_abs_error": np.percentile(abs_diff, 99),
            "p99_rel_error": np.percentile(rel_diff, 99),
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
    p_lev = np.array(reference["p_lev"])

    if p_lev[0,0,0] < p_lev[-1,0,0]:
        toa_index, sfc_index = 0, -1
    else:
        toa_index, sfc_index = -1, 0

    hr_reference = heating_rate(reference["lw_flux_net"], reference["sw_flux_net"], reference["p_lev"])
    hr_result = heating_rate(result["lw_flux_net"], result["sw_flux_net"], reference["p_lev"])

    output = dict()
    datasets = {
            "LW up flux": (reference["lw_flux_up"], result["lw_flux_up"]),
            "LW down flux": (reference["lw_flux_dn"], result["lw_flux_dn"]),
            "LW flux": (reference["lw_flux_net"], result["lw_flux_net"]),
            "SW flux": (reference["sw_flux_net"], result["sw_flux_net"]),
            "SW up flux": (reference["sw_flux_up"], result["sw_flux_up"]),
            "SW down flux": (reference["sw_flux_dn"], result["sw_flux_dn"]),
            "net flux": (
                reference["lw_flux_net"][...] + reference["sw_flux_net"][...],
                result["lw_flux_net"][...] + result["sw_flux_net"][...],
            ),
            "heating rate": (hr_reference, hr_result),
    }

    for name, (A, B) in datasets.items():
        output[name] = compare(A, B)
        output[name + " (TOA)"] = compare(A[toa_index], B[toa_index])
        output[name + " (surface)"] = compare(A[sfc_index], B[sfc_index])

    print(json.dumps(output, indent=4))

if __name__ == "__main__":
    main()
