import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import json

sbn.set("paper")
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

results = [
    json.load(open(f"result_{i}.json")) for i in range(n)
]

plt.subplot(321)
plt.bar(np.arange(n), [p[0] for p in times], label="LW solver")
plt.bar(np.arange(n), [p[1] for p in times], bottom=[p[0] for p in times], label="SW solver")
plt.ylabel("Execution time (ms)")
plt.legend()
plt.grid(False, axis="x")
plt.xticks(xs, labels)

plt.subplot(322)
baseline = sum(times[0])
plt.bar(np.arange(n), [baseline / sum(p) for p in times])
plt.ylabel("Speedup over FP64")
plt.xticks(xs, labels)
plt.grid(False, axis="x")

plt.subplot(323)
w = .8/3
plt.bar(xs - w, [results[i]["LW flux"]["mean_rel_error"] for i in range(n)], w, label="LW")
plt.bar(xs, [results[i]["SW flux"]["mean_rel_error"] for i in range(n)], w, label="SW")
plt.bar(xs + w, [results[i]["net flux"]["mean_rel_error"] for i in range(n)], w, label="Net")
plt.yscale("log")
plt.ylim(1e-9, 1e-1)
plt.ylabel("mean relative error")
plt.title("SW flux, mean relative error")
plt.legend()
plt.xticks(xs, labels)
plt.grid(False, axis="x")


plt.subplot(325)
w = .4
plt.bar(xs - .5*w, np.abs([results[i]["net flux (TOA)"]["mean_bias"] for i in range(n)]), w, label="TOA")
plt.bar(xs + .5*w, np.abs([results[i]["net flux (surface)"]["mean_bias"] for i in range(n)]), w, label="Surface")
plt.yscale("log")
plt.ylim(1e-6, 1e2)
plt.ylabel("Mean bias (W m⁻²)")
plt.legend()
plt.title("Net flux, mean bias")
plt.grid(False, axis="x")
plt.xticks(xs, labels)

plt.subplot(326)
w = .4
plt.bar(xs - .5*w, np.abs([results[i]["net flux (TOA)"]["root_mean_sq_error"] for i in range(n)]), w, label="TOA")
plt.bar(xs + .5*w, np.abs([results[i]["net flux (surface)"]["root_mean_sq_error"] for i in range(n)]), w, label="Surface")
plt.yscale("log")
plt.ylim(1e-6, 1e2)
plt.ylabel("RMSE (W m⁻²)")
plt.legend()
plt.title("Net flux, RMSE")
plt.grid(False, axis="x")
plt.xticks(xs, labels)

plt.tight_layout()
plt.show()
