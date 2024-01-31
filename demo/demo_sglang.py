import argparse
import time

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)

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

# fmt: off
@sgl.function
def character_gen(s, name):
    s += name+ " is a character in Harry Potter.\nPlease fill in the following information about this character.\n"
    s += sgl.gen("json_output", max_tokens=256, regex=character_regex)
# fmt: on


def main(args):
    names = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            names.append(line.strip())
    names = names[: args.num_jsons]

    # Select backend
    backend = select_sglang_backend(args)
    sgl.set_default_backend(backend)

    # Run requests
    time.sleep(3)
    tic = time.time()
    for name in names:
        state = character_gen.run(
            name,
            temperature=0,
            stream=True,
        )
        for out in state.text_iter():
            print(out, end="", flush=True)
        print()
        if name != names[-1]:
            print("=" * 50)

    latency = time.time() - tic

    # Compute accuracy
    print("=" * 20, end="")
    print(f"\x1b[44mLatency: {latency:.3f}\x1b[0m", end="")
    print("=" * 20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="dataset.txt")
    parser.add_argument("--num-jsons", type=int, default=50)
    args = add_common_sglang_args_and_parse(parser)
    main(args)
