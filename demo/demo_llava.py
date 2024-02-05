import argparse
import os
import time

import sglang as sgl
from sglang.backend.runtime_endpoint import RuntimeEndpoint

character_regex = (
    r"""\{\n"""
    + r"""    "name": "[\w\d\s]{1,16}",\n"""
    + r"""    "occupation": "[\w\d\s]{1,16}",\n"""
    + r"""    "weapon": \{\n"""
    + r"""        "name": "[\w\d\s]{1,16}",\n"""
    + r"""        "type": "(physical|magical)",\n"""
    + r"""        "power": "[0-9]{1,3}"\n"""
    + r"""    \},\n"""
    + r"""    "wear": "[\w\d\s]{1,16}",\n"""
    + r"""    "ablities": \["[\w\d\s]{1,16}", "[\w\d\s]{1,16}", "[\w\d\s]{1,16}"\],\n"""
    + r"""    "motto": "[\w\d\s]{1,32}"\n"""
    + r"""\}"""
)

answer_regex = f"""\
The leftmost character:
{character_regex}
The middle character:
{character_regex}
The rightmost character:
{character_regex}
"""


@sgl.function
def image_qa(s, image_path, question):
    s += sgl.user(sgl.image(image_path))
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=512, regex=answer_regex))


def stream_demo(args, question):
    print()
    os.system(f"imgcat -W 40% {args.image_path}")
    print(question)
    time.sleep(2)

    state = image_qa.run(
        image_path=args.image_path,
        question=question,
        stream=True,
    )

    for out in state.text_iter("answer"):
        print(out, end="", flush=True)
    print()


def main():
    paser = argparse.ArgumentParser()
    paser.add_argument("--image-path", type=str, default="characters.png")
    args = paser.parse_args()

    sgl.set_default_backend(RuntimeEndpoint("http://localhost:30000"))

    question = "The image is about three different characters.\nPlease fill the following JSONs about each of them."

    stream_demo(args, question)


if __name__ == "__main__":
    main()
