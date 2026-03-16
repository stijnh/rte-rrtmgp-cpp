import h5py
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import json

plt.style.use(["science", "grid"])
matplotlib.rcParams.update({
    "text.usetex": True,
    "axes.grid" : True,
    "grid.color": ".8"
})

figsize = (4, 3)
i,j = 0,0
plt.figure(figsize=figsize)

input_file = sys.argv[1]
results_file = sys.argv[2]

inputs = h5py.File(input_file)
results = h5py.File(results_file)

height_per_layer = 25
height_lim = 2500

plt.subplot(121)
plt.title("Input gas ratios")

gasses = {
        "vmr_h2o": "H$_2$O",
        "vmr_co2": "CO$_2$",
        "vmr_o3": "O$_3$",
        "vmr_ch4": "CH$_4$",
}

for key in gasses:
    data = inputs[key]
    heights = np.arange(len(data)) * height_per_layer
    ys = np.array(data[:,i,j])
    plt.plot(ys, heights, label=gasses[key])

plt.legend(loc="upper left")
plt.xscale("log")
plt.ylim(0, height_lim)
plt.xlabel("Mixing ratio")
plt.ylabel("Height from surface (meter)")

plt.subplot(122)
plt.title("Output fluxes")
y0 = np.array(results["lw_flux_net"][:,i,j])
y1 = np.array(results["sw_flux_net"][:,i,j])
ys = y0 + y1
heights = np.arange(len(ys)) * height_per_layer
plt.plot(ys, heights, label="Net flux", c=".3")
plt.yticks([500 * i for i in range(10)], [""] * 10)
plt.ylim(0, height_lim)
plt.legend(loc="upper left")
plt.xlabel("Wm$^{-2}$")

out_file = "example_fluxes.pdf"
plt.tight_layout()
plt.savefig(out_file, pad_inches=0)
print(f"saved as {out_file}")

