import h5py
import sys
import numpy as np


def main(output_file, reference_file):
    print(f"opening {output_file=}")
    output = h5py.File(output_file)

    print(f"opening {reference_file=}")
    reference = h5py.File(reference_file)

    columns = [
        'lw_flux_up',
        'lw_flux_dn',
        'lw_flux_net',
        'sw_flux_up',
        'sw_flux_dn',
        'sw_flux_dn_dir',
        'sw_flux_net'
    ]

    for column in columns:
        print(f"field {column}")
        print(f" - shape: {output[column].shape} vs {reference[column].shape}")

        A = np.array(output[column]).flatten()
        B = np.array(reference[column]).flatten()
        diff = A - B
        mask = B != 0

        print(" - mean abs diff:", np.mean(np.abs(diff)))
        print(" - mean rel diff:", np.mean(np.abs(diff[mask])/np.abs(B[mask])))
        print(" - first ten values (output - reference = diff)")

        for a, b, d in zip(A, B, diff[:10]):
            print(f"   * {a} - {b} = {d}")

        print()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"usage: python {sys.argv[0]} [output file] [reference file]")
        sys.exit(0)

    output_file = sys.argv[1]
    reference_file = sys.argv[2]
    main(output_file, reference_file)
