import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import json

sbn.set("paper")
figsize = (8, 3)
n = 5
xs = np.arange(n)

times = [
        (2862.284424, 1930.439331),
        (2355.549316, 1759.989014),
        (1649.722168, 1105.866699),
        (1437.307861, 1036.919922),
        (1406.610474, 937.159607),
]

labels = ["A", "B", "C", "D", "E"]

def load_datasets():
    results = [
        json.load(open(f"result_{i}.json")) for i in range(n)
    ]

    datasets = dict()

    for a in results[0]:
        for b in results[0][a]:
            datasets[a,b] = [results[i][a][b] for i in range(n)]

    return datasets

def savefig(name):
    filename = f"{name}.eps"
    plt.tight_layout()
    plt.savefig(filename)
    print(f"saved as {filename}")

results = load_datasets()

def plot_times():
    plt.figure(figsize=figsize)

    plt.subplot(121)
    plt.bar(np.arange(n), [p[0] for p in times], label="LW solver")
    plt.bar(np.arange(n), [p[1] for p in times], bottom=[p[0] for p in times], label="SW solver")
    plt.ylabel("Execution time (ms)")
    plt.legend()
    plt.grid(False, axis="x")
    plt.xticks(xs, labels)

    plt.subplot(122)
    baseline = sum(times[0])
    plt.bar(np.arange(n), [baseline / sum(p) for p in times])
    plt.ylabel("Speedup over FP64")
    plt.xticks(xs, labels)
    plt.grid(False, axis="x")

    savefig("times")


def plot_rel_error():
    plt.figure(figsize=figsize)

    ys = [sum(times[0]) / sum(p) for p in times]
    xs = results["net flux","mean_rel_error"]
    plt.xscale("log")

    for x, y, label in zip(xs, ys, labels):
        plt.annotate(
                f"Version {label}",
                (x, y),
                (x*3, y),
                va="center",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    fc="white",
                    ec="none",
                    alpha=0.7
                ),
                fontsize=10
        )

    plt.xlim(np.amin(xs) / 10, np.amax(xs) * 200)
    plt.ylim(.9, 2.12)
    plt.scatter(xs, ys, s=100)
    plt.title("Error on net flux (LW + SW)")
    plt.xlabel("Mean relative error on net flux")
    plt.ylabel("Speedup over FP64")
    plt.xscale("log")
    savefig("rel_error")


def plot_abs_error():
    plt.figure(figsize=figsize)
    for index in range(2):
        w = .7 / 3
        key = ["mean_bias", "root_mean_sq_error"][index]

        plt.subplot(121 + index)

        kwargs = dict(s=50, marker="d")
        plt.scatter(xs - w, np.abs(results["LW flux", key]), label="LW flux", **kwargs)
        plt.scatter(xs, np.abs(results["SW flux", key]), label="SW flux", **kwargs)
        plt.scatter(xs + w, np.abs(results["net flux", key]), label="Net flux", **kwargs)

        for x in range(n - 1):
            plt.axvline(x+.5, c="1")

        plt.title(["Flux, mean bias", "Flux, RMSE"][index])
        plt.ylabel(["Mean bias (W m⁻²)", "RMSE (W m⁻²)"][index])
        plt.yscale("log")
        plt.legend()
        plt.grid(False, axis="x")
        plt.xticks(xs, labels)

    savefig("abs_error")

def plot_heating_error():
    plt.figure(figsize=figsize)
    plt.suptitle("Error on heating rate at surface and top of atmosphere (TOA)")

    for index in [0, 1]:
        plt.subplot(121 + index)
        w = .4
        plt.bar(xs - .5*w, np.abs(results["heating rate (surface)","mean_abs_error"]), w, label="Surface")
        plt.bar(xs + .5*w, np.abs(results["heating rate (TOA)","mean_abs_error"]), w, label="TOA")
        plt.ylabel("Mean error of heating rate (K/day)")
        plt.legend()
        plt.ylim(0, [0.005, 5][index])
        plt.xticks(xs, labels)

    savefig("heating_error")

#plot_times()
#plot_rel_error()
#plot_abs_error()
plot_heating_error()
