## https://docs.vllm.ai/en/latest/getting_started/quickstart.html

from vllm import LLM, SamplingParams
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="facebook/opt-125m")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

## ---- run 
# (base) xlisp@xlisp:~/jimw-vllm$ python quickstart.py
# config.json: 100%|███████████████████████████████████████████████████████████████| 651/651 [00:00<00:00, 2.90MB/s]
# INFO 12-06 14:54:20 llm_engine.py:249] Initializing an LLM engine (v0.6.4.post1) with config: model='facebook/opt-125m', speculative_config=None, tokenizer='facebook/opt-125m', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=2048, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=facebook/opt-125m, num_scheduler_steps=1, chunked_prefill_enabled=False multi_step_stream_outputs=True, enable_prefix_caching=False, use_async_output_proc=True, use_cached_outputs=False, chat_template_text_format=string, mm_processor_kwargs=None, pooler_config=None)
# tokenizer_config.json: 100%|█████████████████████████████████████████████████████| 685/685 [00:00<00:00, 2.52MB/s]
# vocab.json: 100%|███████████████████████████████████████████████████████████████| 899k/899k [00:01<00:00, 788kB/s]
# merges.txt: 100%|██████████████████████████████████████████████████████████████| 456k/456k [00:00<00:00, 4.62MB/s]
# special_tokens_map.json: 100%|███████████████████████████████████████████████████| 441/441 [00:00<00:00, 1.91MB/s]
# generation_config.json: 100%|█████████████████████████████████████████████████████| 137/137 [00:00<00:00, 604kB/s]
# INFO 12-06 14:54:27 selector.py:261] Cannot use FlashAttention-2 backend for Volta and Turing GPUs.
# INFO 12-06 14:54:27 selector.py:144] Using XFormers backend.
# INFO 12-06 14:54:28 model_runner.py:1072] Starting to load model facebook/opt-125m...
# INFO 12-06 14:54:29 weight_utils.py:243] Using model weights format ['*.bin']
# pytorch_model.bin: 100%|███████████████████████████████████████████████████████| 251M/251M [00:42<00:00, 5.86MB/s]
# Loading pt checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
# /home/xlisp/anaconda3/lib/python3.12/site-packages/vllm/model_executor/model_loader/weight_utils.py:425: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   state = torch.load(bin_file, map_location="cpu")
# Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  3.18it/s]
# Loading pt checkpoint shards: 100% Completed | 1/1 [00:00<00:00,  3.18it/s]
# 
# INFO 12-06 14:55:15 model_runner.py:1077] Loading model weights took 0.2389 GB
# INFO 12-06 14:55:15 worker.py:232] Memory profiling results: total_gpu_memory=7.92GiB initial_memory_usage=0.37GiB peak_torch_memory=0.71GiB memory_usage_post_profile=0.39GiB non_torch_memory=0.15GiB kv_cache_size=6.27GiB gpu_memory_utilization=0.90
# INFO 12-06 14:55:15 gpu_executor.py:113] # GPU blocks: 11422, # CPU blocks: 7281
# INFO 12-06 14:55:15 gpu_executor.py:117] Maximum concurrency for 2048 tokens per request: 89.23x
# INFO 12-06 14:55:19 model_runner.py:1400] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
# INFO 12-06 14:55:19 model_runner.py:1404] If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
# INFO 12-06 14:55:37 model_runner.py:1518] Graph capturing finished in 18 secs, took 0.39 GiB
# Processed prompts: 100%|████| 4/4 [00:00<00:00, 21.11it/s, est. speed input: 137.29 toks/s, output: 337.92 toks/s]
# Prompt: 'Hello, my name is', Generated text: ' Joel<s><s><s><s><s><s><s><s><s><s><s><s><s><s><s>'
# Prompt: 'The president of the United States is', Generated text: ' giving<s><s><s><s><s><s><s><s><s><s><s><s><s><s><s>'
# Prompt: 'The capital of France is', Generated text: ' now<s><s><s><s><s><s><s><s><s><s><s><s><s><s><s>'
# Prompt: 'The future of AI is', Generated text: ' now<s><s><s><s><s><s><s><s><s><s><s><s><s><s><s>'
# [rank0]:[W1206 14:55:38.624395404 ProcessGroupNCCL.cpp:1250] Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present,  but this warning has only been added since PyTorch 2.4 (function operator())
# (base) xlisp@xlisp:~/jimw-vllm$
# 
