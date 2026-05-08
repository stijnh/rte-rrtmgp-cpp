import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import json
import re

plt.style.use(["science", "grid"])
matplotlib.rcParams.update({
    "text.usetex": True,
    "axes.grid" : True,
    "grid.color": ".8"
})

figsize = (8, 3)

def load_energy(filename):
    with open(filename) as f:
        content = f.read()

    lw_matches = re.findall(r"Duration longwave solver: ([0-9.]+) \(ms\), ([0-9.]+) J", content)
    sw_matches = re.findall(r"Duration shortwave solver: ([0-9.]+) \(ms\), ([0-9.]+) J", content)

    lw_eff = [float(e)/float(t) for t,e in lw_matches]
    sw_eff = [float(e)/float(t) for t,e in sw_matches]

    lw_times, sw_times = load_times(filename)

    return (
            np.median(lw_eff) * lw_times,
            np.median(sw_eff) * sw_times,
    )

def load_times(filename):
    with open(filename) as f:
        content = f.read()

    lw_times = re.findall("Duration longwave solver: ([0-9.]+)", content)
    sw_times = re.findall("Duration shortwave solver: ([0-9.]+)", content)

    return (
            np.median([float(x) for x in lw_times]),
            np.median([float(x) for x in sw_times]),
    )

def load_json(filename):
    with open(filename) as f:
        return json.load(f)

levels = {
        "double": 0,
        "float": 1,
        "A": 2,
        "B": 3,
        "C": 4,
        "D": 5,
        "E": 6,
}

labels = "ABCDE"
n = len(labels)
xs = np.arange(n)

def savefig(name):
    filename = f"{name}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"saved as {filename}")


def annotate(label, x, y, ha="left"):
    plt.annotate(
            label,
        (x, y),
        (x*2 if ha == "left" else x/2, y*0.995),
        va="center",
        ha=ha,
        bbox=dict(
            boxstyle="round,pad=0.3",
            fc="white",
            ec="none",
            alpha=0.7
        ),
        fontsize=10
    )

def plot_performance(name, results, times, unit="time"):
    assert unit in ("time", "energy")
    plt.figure(figsize=(figsize[0], figsize[1]*0.8))

    baseline = sum(times["double"])
    time = {k: sum(times[k]) for k in times}
    speedup = {k: baseline / t for k, t in time.items()}

    errors = {k: results[k]["net flux"]["mean_rel_error"] for k in times}
    errors = {k: results[k]["net flux"]["root_mean_sq_error"] / results[k]["net flux"]["mean"] for k in times}

    print(f"timing results {name}")
    for label in times:
        x = errors[label]
        y = sum(times[label])
        print(f" * {label}: time={y} speedup={baseline/y} error={x} logerror={np.log10(x)}")

    plt.subplot(121)
    plt.text(.05, .95, name.capitalize(), ha="left", va="top", transform=plt.gca().transAxes, fontsize=14, backgroundcolor="1")

    plt.bar([-2], [baseline], color=".3")
    plt.bar([-1], [sum(times["float"])], color=".5")

    for x, l in enumerate(labels):
        plt.bar(x, time[l])

    xticks = [-2, -1] + list(xs)
    xlabels = ["FP64", "FP32"] + list(labels)
    plt.xticks(xticks, xlabels)

    if unit == "energy":
        ymultiple = 250
    else:
        ymultiple = 1000

    ylim = np.ceil(baseline / ymultiple + 1) * ymultiple
    plt.grid(False, axis="x")
    plt.xlim(-2.5, len(labels)-.5)
    plt.ylim(0, ylim)

    if unit == "energy":
        plt.ylabel("Energy usage (J)")
    else:
        plt.ylabel("Execution time (ms)")

    plt.subplot(122)
    plt.text(.05, .95, name.capitalize(), ha="left", va="top", transform=plt.gca().transAxes, fontsize=14, backgroundcolor="1")

    for label in labels:
        x = errors[label]
        y = speedup[label]
        plt.scatter(x, y, s=50, marker="x")

        if label != "A":
            ha="right"
        else:
            ha="left"

        annotate(f"Version {label}", x, y, ha)

    x = errors["float"]
    y = speedup["float"]
    plt.scatter(x, y, s=50, marker=".", c=".5")
    annotate("Version FP32", x, y, "left")

    plt.xscale("log")
    plt.xlabel("Normalized RMSE on net flux (W m$^{-2}$)")
    plt.xlim(10**-9, 1)
    plt.ylim(1.0, 2.3)

    if unit == "energy":
        plt.ylabel("Energy efficiency vs FP64")
    else:
        plt.ylabel("Speedup over FP64")

    if unit == "energy":
        savefig(f"energy_{name}")
    else:
        savefig(f"times_{name}")

def plot_heating_rate(results):
    plt.figure(figsize=(figsize[0], figsize[1]))
    marker_step = 5 # add marker every 5 points
    layer_height = 25 # meter per layer
    nz = 1500 // layer_height
    heights = np.arange(nz) * layer_height
    zorder = 100

    markers = ["^", "v", "x", "|", "."]
    xlim = (-5, 30)

    plt.subplot(131)

    layers = results["double"]["heating rate layers"]
    ys = [l["mean"] for l in layers][:nz]
    plt.plot(ys, heights, c=".3", label="FP64")

    plt.legend(loc="upper right")
    plt.title("Reference output")
    plt.xlim(*xlim)
    plt.ylim(0, np.amax(heights))
    plt.xlabel("Mean heating rate per layer (K/day)")
    plt.ylabel("Height from surface (meters)")

    plt.subplot(132)
    ax = plt.gca()
    #axi = ax.inset_axes([.5, .5, .25, .25])
    #axi.set_xlim(1000, 1500)
    #axi.set_ylim(1-eps, 1+eps)
    #ax.indicate_inset_zoom(axi)

    for key, marker in zip(labels, markers):
        # Get the data
        layers = results[key]["heating rate layers"]
        ys = [l["mean"] + l["mean_bias"] for l in layers][:nz]

        # Get the color
        [l] = plt.plot([], [], label=key, marker=marker)
        color = l.get_color()

        # Plot the real results
        ax.plot(ys, heights, zorder=zorder, color=color)
        ax.plot(ys[::marker_step], heights[::marker_step], c=color, marker=marker, ls="")

        #axi.plot(heights, ys, zorder=zorder, color=color)

    plt.title("Mixed-precision output")
    plt.xlim(*xlim)
    plt.ylim(0, np.amax(heights))
    plt.xlabel("Mean heating rate per layer (K/day)")
    plt.legend(loc="upper right").set_zorder(1000)
    plt.tick_params('y', labelleft=False)

    plt.subplot(133)

    print("heating rate error:")

    for key, marker in zip(labels, markers):
        # Get the data
        layers = results[key]["heating rate layers"]
        ys = [np.abs(l["mean_bias"]) for l in layers][:nz]
        #ys = [np.abs(l["root_mean_sq_error"]) / np.abs(l["mean"]) for l in layers][:nz]
        print(f" * {key}: heating rate error=10**{np.log10(np.average(ys))}")

        # Get the color
        [l] = plt.plot([], [], label=key, marker=marker)
        color = l.get_color()

        # Plot the real results
        plt.plot(ys, heights, ys, color=color)
        plt.plot(ys[::marker_step], heights[::marker_step], c=color, marker=marker, ls="")

    plt.title("Difference reference vs output")
    plt.xlim(1e-9, 100)
    plt.ylim(0, np.amax(heights))
    plt.xscale("log")
    plt.xlabel("Mean heating rate per layer (K/day)")
    plt.tick_params('y', labelleft=False)

    savefig("heating_rate")

def plot_energy_efficiency(name, times, energy):
    plt.figure(figsize=figsize)
    colors = {"double": ".3", "float": ".5"}

    plt.subplot(121)
    xs = [sum(times[l]) for l in labels]
    ys = [sum(energy[l]) for l in labels]

    for x, y in zip(xs, ys):
        plt.scatter([x], [y], marker="x")

    tdp = 300
    xlim = (1500, 3500)
    ylim = tuple(np.array(xlim) * tdp * 1e-3)
    plt.plot(xlim, ylim, c=".1")
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("Execution time (ms)")
    plt.ylabel("Energy usage (J)")
    plt.text(
            xlim[0] + (xlim[1]-xlim[0])*0.48,
            ylim[0] + (ylim[1]-ylim[0])*0.52,
            "TDP 300W",
            va="center", ha="center", rotation=35)

    plt.subplot(122)
    xticks = ["double", "float"] + list(labels)
    for i, l in enumerate(xticks):
        plt.bar(i, sum(energy[l]) / sum(times[l]) * 1000, color=colors.get(l))

    plt.xticks(range(len(xticks)), xticks)
    plt.ylim(0, 350)
    plt.text(-.5, tdp * 1.02, "TDP 300W", ha="left", va="bottom", backgroundcolor="1")
    plt.axhline(tdp, c=".1", zorder=100)
    plt.ylabel("Power (W)")
    savefig(f"efficiency_{name}")


times_lumi = {
        k: load_times(f"results/lumi/output_{v}.txt") for k,v in levels.items()
}

times_leonardo = {
        k: load_times(f"results/leonardo/output_{v}.txt") for k,v in levels.items()
}

times_snellius = {
        k: load_times(f"results/a100/output_{v}.txt") for k,v in levels.items()
}

energy_snellius = {
        k: load_energy(f"results/a100/output_{v}.txt") for k,v in levels.items()
}

errors_lumi = {
        k: load_json(f"results/lumi/result_{v}.json") for k,v in levels.items()
}

errors_a100 = {
        k: load_json(f"results/a100/result_{v}.json") for k,v in levels.items()
}


plot_performance("snellius", errors_a100, energy_snellius, unit="energy")
plot_performance("snellius", errors_a100, times_snellius, unit="time")
plot_energy_efficiency("snellius", times_snellius, energy_snellius)
#plot_performance("leonardo", errors_a100, times_leonardo)
#plot_performance("lumi", errors_lumi, times_lumi)
#plot_heating_rate(errors_a100)
