import argparse
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import numpy as np


def show_name(name):
    convert = {
        # backend
        "srt": "SGLang",
        "vllm": "Outlines + vLLM",
        "guidance": "Guidance + llama.cpp",
        "srt_no_fst_fwd": "SGLang (No Jump-Forward)",
        # bench_mode
        "character": "Character Generation",
        "city": "Long Document Retrieval",
        "character_bs1": "Character Generation\n(batch size = 1)",
        "city_bs1": "Long Document Retrieval\n(batch size = 1)",
    }
    return convert.get(name, name)


def get_hatch(backend):
    convert = {
        "srt": "",
        "srt_no_fst_fwd": "*",
        "vllm": "",
        "guidance": "",
    }
    return convert.get(backend, "")


def get_color(backend):
    convert = {
        "srt": "C0",
        "srt_no_fst_fwd": "lightblue",
        "vllm": "C1",
        "guidance": "C2",
    }
    return convert.get(backend, "k")


backend_list = ["srt", "srt_no_fst_fwd", "vllm", "guidance"]


def plot_group_bar_chart(
    ax, data, tasks, methods, width=0.1, y_label="", baseline=None
):
    # Organize data
    group_labels = [show_name(t) for t in tasks]
    groups = []
    for method in methods:
        tmp = []
        for task in tasks:
            value = data[task].get(method, 0)
            if baseline is not None:
                value /= data[task][baseline]
            tmp.append(value)
        groups.append(tmp)

    # Draw bar
    x = np.arange(len(group_labels))
    for i, g in enumerate(groups):
        ax.bar(
            x + i * width,
            g,
            width,
            label=show_name(methods[i]),
            # hatch=get_hatch(methods[i]),
            color=get_color(methods[i]),
        )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label, fontsize=15)
    if baseline is not None:
        ax.set_ylim(ymax=1.1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticks(x + (len(groups) - 1) / 2 * width)
    ax.set_xticklabels(group_labels, fontsize=15)
    # x tick fontsize
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(15)
    ax.legend(
        handlelength=0.7,
        ncols=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.45),
        fontsize=13,
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # build the data
    results = []
    with open("result.jsonl", "r") as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except:
                pass

    max_parallel = defaultdict(lambda: 0)
    for res in results:
        max_parallel[res["backend"]] = max(
            max_parallel[res["backend"]], res["parallel"]
        )

    data = defaultdict(lambda: defaultdict(lambda: 0))
    data_ct = defaultdict(lambda: defaultdict(lambda: 0))
    for res in results:
        if res["parallel"] == max_parallel[res["backend"]]:
            data[res["mode"]][res["backend"]] += res["latency"]
            data_ct[res["mode"]][res["backend"]] += 1
        if res["parallel"] == 1:
            data[res["mode"] + "_bs1"][res["backend"]] += res["latency"]
            data_ct[res["mode"] + "_bs1"][res["backend"]] += 1

    for bm in data:
        for backend in data[bm]:
            data[bm][backend] /= data_ct[bm][backend]
            print(bm, backend, data[bm][backend])
            data[bm][backend] = 1 / data[bm][backend]

    # get one figure
    fig, ax = plt.subplots(1, 1, figsize=(13, 4))
    plot_group_bar_chart(
        ax,
        data,
        ["character", "city", "character_bs1", "city_bs1"],
        backend_list,
        y_label="Normalized Throughput",
        baseline="srt",
    )
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)

    fig.savefig("result.png")


if __name__ == "__main__":
    main()
