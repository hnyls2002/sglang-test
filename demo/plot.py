import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt
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


def get_color(backend):
    convert = {
        "srt": "orange",
        "srt_no_fst_fwd": "lightcoral",
        "vllm": "lightgreen",
        "guidance": "lightblue",
    }
    return convert.get(backend, "k")


backend_list = ["srt", "srt_no_fst_fwd", "vllm", "guidance"]


def plot_one_chat(ax, x_names, data, title, width=0.7, y_lable=""):
    x_pos = np.arange(len(x_names))
    vals = [data[x] for x in x_names]
    ax.bar(
        x_pos,
        vals,
        width,
        label=[show_name(x) for x in x_names],
        color=[get_color(x) for x in x_names],
    )

    # remove the xticks
    ax.set_xticks([])
    ax.set_ylabel(y_lable, fontsize=15)
    ax.set_title(title, fontsize=15)
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
            data[res["mode"]][res["backend"]] += res["num_jsons"] / res["latency"]
            data_ct[res["mode"]][res["backend"]] += 1
        if res["parallel"] == 1:
            data[res["mode"] + "_bs1"][res["backend"]] += (
                res["latency"] / res["num_jsons"]
            )
            data_ct[res["mode"] + "_bs1"][res["backend"]] += 1

    for bm in data:
        for backend in data[bm]:
            data[bm][backend] /= data_ct[bm][backend]
            print(bm, backend, data[bm][backend])

    # get one figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    plot_one_chat(
        axs[0][0],
        ["srt", "srt_no_fst_fwd", "vllm", "guidance"],
        data["character"],
        "Character Generation",
        y_lable="Throughput (Req / s)",
    )
    plot_one_chat(
        axs[0][1],
        ["srt", "srt_no_fst_fwd", "vllm", "guidance"],
        data["city"],
        "Long Document Retrieval",
        y_lable="Throughput (Req / s)",
    )
    plot_one_chat(
        axs[1][0],
        ["srt", "srt_no_fst_fwd", "vllm", "guidance"],
        data["character_bs1"],
        "Character Generation\n(Batch Size = 1)",
        y_lable="Latency (s / Req)",
    )
    plot_one_chat(
        axs[1][1],
        ["srt", "srt_no_fst_fwd", "vllm", "guidance"],
        data["city_bs1"],
        "Long Document Retrieval\n(Batch Size = 1)",
        y_lable="Latency (s / Req)",
    )

    plt.subplots_adjust(wspace=0.3, hspace=0.25, bottom=0.12)
    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0),
        fontsize=16,
    )

    fig.savefig("result.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
