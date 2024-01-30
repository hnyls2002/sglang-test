import argparse
import json
import time
from functools import partial

import requests
from sglang.test.test_utils import add_common_other_args_and_parse

character_regex = (
    r"""\{\n"""
    + r"""    "name": "[\w\d\s]{1,16}",\n"""
    + r"""    "house": "(Gryffindor|Slytherin|Ravenclaw|Hufflepuff)",\n"""
    + r"""    "blood status": "(Pure-blood|Half-blood|Muggle-born)",\n"""
    + r"""    "occupation": "(student|teacher|auror|ministry of magic|death eater|order of the phoenix)",\n"""
    + r"""    "wand": \{\n"""
    + r"""        "wood": "[\w\d\s]{1,16}",\n"""
    + r"""        "core": "[\w\d\s]{1,16}",\n"""
    + r"""        "length": [0-9]{1,2}\.[0-9]{0,2}\n"""
    + r"""    \},\n"""
    + r"""    "alive": "(Alive|Deceased)"\n"""
    + r"""\}"""
)


def call_generate_outlines(
    prompt, temperature, max_tokens, url, stop=[], regex=None, stream=False
):
    data = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
        "regex": regex,
        "n": 1,
        "stream": stream,
    }
    res = requests.post(url, json=data, stream=stream)
    return res


def main(args):
    names = []
    with open(args.data_path, "r") as f:
        for line in f:
            names.append(line.strip())
    names = names[: args.num_jsons]

    url = f"{args.host}:{args.port}/generate"
    generate = partial(call_generate_outlines, url=url)

    time.sleep(3)
    tic = time.time()
    for name in names:
        prompt = (
            name
            + " is a character in Harry Potter.\nPlease fill in the following information about the character.\n"
        )
        res = generate(
            prompt,
            temperature=0,
            max_tokens=256,
            regex=character_regex,
            stream=True,
        )
        last_len = 0
        for chunk in res.iter_content(chunk_size=None):
            chunk_str = chunk.decode().strip()
            chunk_str = chunk_str[: chunk_str.rfind("}") + 1]
            chunk_json = json.loads(chunk_str)
            out = chunk_json["text"][0][last_len:]
            last_len = len(chunk_json["text"][0])
            print(out, end="", flush=True)
        print()
        if name != names[-1]:
            print("=" * 50)

    latency = time.time() - tic

    # Compute accuracy
    # highlight the latency print
    print("=" * 20, end="")
    print(f"\x1b[44mLatency: {latency:.3f}\x1b[0m", end="")
    print("=" * 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="dataset.txt")
    parser.add_argument("--num-jsons", type=int, default=50)
    args = add_common_other_args_and_parse(parser)
    main(args)
