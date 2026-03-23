import h5py, numpy as np
import sys
import argparse
import json
from tqdm import tqdm

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

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
    for i in tqdm(range(num_layers)):
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
    return total_heating_rate * seconds_per_day

def summarize(reference, result):
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
            "median_rel_error": np.median(rel_diff),
            "median_abs_error": np.median(abs_diff),
            "max_abs_error": np.amax(abs_diff),
            "max_rel_error": np.amax(rel_diff),
            "p99_abs_error": np.percentile(abs_diff, 99),
            "p99_rel_error": np.percentile(rel_diff, 99),
    }


def main():
    parser = argparse.ArgumentParser(prog="Compare output")
    parser.add_argument("reference-file")
    parser.add_argument("result-file")
    parser.add_argument("--output-file", "--output", "-o", default="", required=False)
    args = parser.parse_args()

    reference_file = getattr(args, 'reference-file')
    result_file = getattr(args, 'result-file')
    output_file = getattr(args, "output_file")

    eprint(f"comparing {reference_file} with {result_file}")

    eprint(f"opening {reference_file}")
    reference = h5py.File(reference_file)
    flux_reference = np.array(reference["lw_flux_net"]) + reference["sw_flux_net"]
    hr_reference = heating_rate(reference["lw_flux_net"], reference["sw_flux_net"], reference["p_lev"])

    eprint(f"opening {result_file}")
    result = h5py.File(result_file)
    flux_result = np.array(result["lw_flux_net"]) + result["sw_flux_net"]
    hr_result = heating_rate(result["lw_flux_net"], result["sw_flux_net"], reference["p_lev"])

    p_lev = np.array(reference["p_lev"])
    n = len(p_lev)

    eprint("computing statistics")
    output = dict()
    stats = [
        ("lw flux", reference["lw_flux_net"], result["lw_flux_net"]),
        ("sw flux", reference["sw_flux_net"], result["sw_flux_net"]),
        ("net flux", flux_reference, flux_result),
        ("heating rate", hr_reference, hr_result),
    ]

    for key, ref, res in tqdm(stats):
        output[key] = summarize(ref, res)
        output[key + " layers"] = [summarize(ref[i], res[i]) for i in tqdm(range(len(ref)))]

    body = json.dumps(output, indent=4)

    if output_file:
        eprint(f"writing to {output_file}")
        with open(output_file) as f:
            f.write(body)
    else:
        print(body)

if __name__ == "__main__":
    main()

