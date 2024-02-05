# Single Bench Streaming Demo

### Dependencies

```
vllm                      0.2.5
outlines                  0.0.25
```

### Run sglang

```bash
python3 -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

```bash
python demo_sglang.py --num-jsons 5
```

### Run Outlines

```bash
python3 -m outlines.serve.serve --tokenizer-mode auto --model meta-llama/Llama-2-7b-chat-hf  --disable-log-requests --port 21000
```

```bash
python demo_outlines.py --backend vllm --num-jsons 5
```


