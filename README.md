# learn vllm
## ubuntu 20.04 install gpu driver 530 & cuda 12.2

```sh
(base) xlisp@xlisp:~/jimw-vllm$ uname -a
Linux xlisp 5.15.0-126-generic #136~20.04.1-Ubuntu SMP Thu Nov 14 16:38:05 UTC 2024 x86_64 x86_64 x86_64 GNU/Linux
(base) xlisp@xlisp:~/jimw-vllm$

$ sudo apt install  nvidia-dkms-530  nvidia-driver-530

## ---- install nvcc
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2004-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-2-local_12.2.0-535.54.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
## ----

(base) xlisp@xlisp:~/jimw-vllm$ nvidia-smi
Fri Dec  6 12:31:23 2024
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce GTX 1080        Off | 00000000:03:00.0  On |                  N/A |
|  0%   36C    P8              11W / 198W |    308MiB /  8192MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      1115      G   /usr/lib/xorg/Xorg                           66MiB |
|    0   N/A  N/A      1494      G   /usr/lib/xorg/Xorg                          155MiB |
|    0   N/A  N/A      1639      G   /usr/bin/gnome-shell                         40MiB |
|    0   N/A  N/A      9859      G   ...90,262144 --variations-seed-version       35MiB |
+---------------------------------------------------------------------------------------+
(base) xlisp@xlisp:~/jimw-vllm$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Jun_13_19:16:58_PDT_2023
Cuda compilation tools, release 12.2, V12.2.91
Build cuda_12.2.r12.2/compiler.32965470_0
(base) xlisp@xlisp:~/jimw-vllm$
```

## test vllm
```py
(base) xlisp@xlisp:~$ proxy
(base) xlisp@xlisp:~$ ipython
Python 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 13:27:36) [GCC 11.2.0]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.27.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: from vllm import LLM, SamplingParams

In [2]: llm = LLM(model="microsoft/phi-1_5")
INFO 12-06 11:39:30 llm_engine.py:249] Initializing an LLM engine (v0.6.4.post1) with config: model='microsoft/phi-1_5', speculative_config=None, tokenizer='microsoft/phi-1_5', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=2048, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=microsoft/phi-1_5, num_scheduler_steps=1, chunked_prefill_enabled=False multi_step_stream_outputs=True, enable_prefix_caching=False, use_async_output_proc=True, use_cached_outputs=False, chat_template_text_format=string, mm_processor_kwargs=None, pooler_config=None)
INFO 12-06 11:39:31 selector.py:261] Cannot use FlashAttention-2 backend for Volta and Turing GPUs.
INFO 12-06 11:39:31 selector.py:144] Using XFormers backend.
INFO 12-06 11:39:31 model_runner.py:1072] Starting to load model microsoft/phi-1_5...
INFO 12-06 11:39:32 weight_utils.py:243] Using model weights format ['*.safetensors']
INFO 12-06 11:39:33 weight_utils.py:288] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.15s/it]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.15s/it]

INFO 12-06 11:39:34 model_runner.py:1077] Loading model weights took 2.6419 GB
INFO 12-06 11:39:36 worker.py:232] Memory profiling results: total_gpu_memory=7.92GiB initial_memory_usage=3.09GiB peak_torch_memory=3.12GiB memory_usage_post_profile=3.12GiB non_torch_memory=0.47GiB kv_cache_size=3.54GiB gpu_memory_utilization=0.90
INFO 12-06 11:39:36 gpu_executor.py:113] # GPU blocks: 1207, # CPU blocks: 1365
INFO 12-06 11:39:36 gpu_executor.py:117] Maximum concurrency for 2048 tokens per request: 9.43x
INFO 12-06 11:39:39 model_runner.py:1400] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 12-06 11:39:39 model_runner.py:1404] If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 12-06 11:40:01 model_runner.py:1518] Graph capturing finished in 21 secs, took 0.54 GiB

In [3]: llm.generate("who are you?")
Processed prompts: 100%|███████████████| 1/1 [00:00<00:00,  3.00it/s, est. speed input: 12.00 toks/s, output: 47.98 toks/s]
Out[3]: [RequestOutput(request_id=0, prompt='who are you?', prompt_token_ids=[8727, 389, 345, 30], encoder_prompt=None, encoder_prompt_token_ids=None, prompt_logprobs=None, outputs=[CompletionOutput(index=0, text=' � disagreement!!!!!!!!!!!!!!', token_ids=(564, 25800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), cumulative_logprob=None, logprobs=None, finish_reason=length, stop_reason=None)], finished=True, metrics=RequestMetrics(arrival_time=1733456759.553771, last_token_time=1733456759.553771, first_scheduled_time=1733456759.5569673, first_token_time=1733456759.6604152, time_in_queue=0.003196239471435547, finished_time=1733456759.8888953, scheduler_time=0.0018781600001602783, model_forward_time=None, model_execute_time=None), lora_request=None, num_cached_tokens=0)]

```

## test llama3.2

* https://huggingface.co/meta-llama/Llama-3.2-1B?local-app=vllm

```bash

huggingface-cli login  Install from pip  Copy  # Install vLLM from pip:

pip install vllm   Copy  # Load and run the model:

vllm serve "meta-llama/Llama-3.2-1B"   Copy  # Call the server using curl:

## ValueError: Bfloat16 is only supported on GPUs with compute capability of at least 8.0. Your NVIDIA GeForce GTX 1080 GPU has compute capability 6.1. You can use float16 instead by explicitly setting the`dtype` flag in CLI, for example: --dtype=half.

curl -X POST "http://localhost:8000/v1/completions" \
	-H "Content-Type: application/json" \
	--data '{
		"model": "meta-llama/Llama-3.2-1B",
		"prompt": "Once upon a time,",
		"max_tokens": 512,
		"temperature": 0.5
	}'

```

* other 

```
In [5]: llm = LLM(model="Qwen/Qwen-7B", trust_remote_code=True)
        - OutOfMemoryError: CUDA out of memory. Tried to allocate 172.00 MiB.

https://huggingface.co/google/gemma-2-9b

ValueError: Bfloat16 is only supported on GPUs with compute capability of at least 8.0. Your NVIDIA GeForce GTX 1080 GPU has compute capability 6.1.

```

* add arguments, allow cpu

```
(base) xlisp@xlisp:~/jimw-vllm$ vllm serve "meta-llama/Llama-3.2-1B" --dtype=half --gpu-memory-utilization=0.95 --max-model-len=105088
INFO 12-06 14:07:37 api_server.py:585] vLLM API server version 0.6.4.post1
INFO 12-06 14:07:37 api_server.py:586] args: Namespace(subparser='serve', model_tag='meta-llama/Llama-3.2-1B', config='', host=None, port=8000, uvicorn_log_level='info', allow_credentials=False, allowed_origins=['*'], allowed_methods=['*'], allowed_headers=['*'], api_key=None, lora_modules=None, prompt_adapters=None, chat_template=None, response_role='assistant', ssl_keyfile=None, ssl_certfile=None, ssl_ca_certs=None, ssl_cert_reqs=0, root_path=None, middleware=[], return_tokens_as_token_ids=False, disable_frontend_multiprocessing=False, enable_auto_tool_choice=False, tool_call_parser=None, tool_parser_plugin='', model='meta-llama/Llama-3.2-1B', task='auto', tokenizer=None, skip_tokenizer_init=False, revision=None, code_revision=None, tokenizer_revision=None, tokenizer_mode='auto', chat_template_text_format='string', trust_remote_code=False, allowed_local_media_path=None, download_dir=None, load_format='auto', config_format=<ConfigFormat.AUTO: 'auto'>, dtype='half', kv_cache_dtype='auto', quantization_param_path=None, max_model_len=105088, guided_decoding_backend='outlines', distributed_executor_backend=None, worker_use_ray=False, pipeline_parallel_size=1, tensor_parallel_size=1, max_parallel_loading_workers=None, ray_workers_use_nsight=False, block_size=16, enable_prefix_caching=False, disable_sliding_window=False, use_v2_block_manager=False, num_lookahead_slots=0, seed=0, swap_space=4, cpu_offload_gb=0, gpu_memory_utilization=0.95, num_gpu_blocks_override=None, max_num_batched_tokens=None, max_num_seqs=256, max_logprobs=20, disable_log_stats=False, quantization=None, rope_scaling=None, rope_theta=None, hf_overrides=None, enforce_eager=False, max_seq_len_to_capture=8192, disable_custom_all_reduce=False, tokenizer_pool_size=0, tokenizer_pool_type='ray', tokenizer_pool_extra_config=None, limit_mm_per_prompt=None, mm_processor_kwargs=None, enable_lora=False, enable_lora_bias=False, max_loras=1, max_lora_rank=16, lora_extra_vocab_size=256, lora_dtype='auto', long_lora_scaling_factors=None, max_cpu_loras=None, fully_sharded_loras=False, enable_prompt_adapter=False, max_prompt_adapters=1, max_prompt_adapter_token=0, device='auto', num_scheduler_steps=1, multi_step_stream_outputs=True, scheduler_delay_factor=0.0, enable_chunked_prefill=None, speculative_model=None, speculative_model_quantization=None, num_speculative_tokens=None, speculative_disable_mqa_scorer=False, speculative_draft_tensor_parallel_size=None, speculative_max_model_len=None, speculative_disable_by_batch_size=None, ngram_prompt_lookup_max=None, ngram_prompt_lookup_min=None, spec_decoding_acceptance_method='rejection_sampler', typical_acceptance_sampler_posterior_threshold=None, typical_acceptance_sampler_posterior_alpha=None, disable_logprobs_during_spec_decoding=None, model_loader_extra_config=None, ignore_patterns=[], preemption_mode=None, served_model_name=None, qlora_adapter_name_or_path=None, otlp_traces_endpoint=None, collect_detailed_traces=None, disable_async_output_proc=False, scheduling_policy='fcfs', override_neuron_config=None, override_pooler_config=None, disable_log_requests=False, max_log_len=None, disable_fastapi_docs=False, enable_prompt_tokens_details=False, dispatch_function=<function serve at 0x7ff687c68220>)
INFO 12-06 14:07:37 api_server.py:175] Multiprocessing frontend to use ipc:///tmp/7212975a-926e-46aa-9787-6759cc64c0a1 for IPC Path.
INFO 12-06 14:07:37 api_server.py:194] Started engine process with PID 16470
WARNING 12-06 14:07:41 config.py:1865] Casting torch.bfloat16 to torch.float16.
INFO 12-06 14:07:47 config.py:350] This model supports multiple tasks: {'embedding', 'generate'}. Defaulting to 'generate'.
WARNING 12-06 14:07:47 arg_utils.py:1013] Chunked prefill is enabled by default for models with max_model_len > 32K. Currently, chunked prefill might not work with some features or models. If you encounter any issues, please disable chunked prefill by setting --enable-chunked-prefill=False.
WARNING 12-06 14:07:47 arg_utils.py:1075] [DEPRECATED] Block manager v1 has been removed, and setting --use-v2-block-manager to True or False has no effect on vLLM behavior. Please remove --use-v2-block-manager in your engine argument. If your use case is not supported by SelfAttnBlockSpaceManager (i.e. block manager v2), please file an issue with detailed information.
INFO 12-06 14:07:47 config.py:1136] Chunked prefill is enabled with max_num_batched_tokens=512.
WARNING 12-06 14:07:48 config.py:1865] Casting torch.bfloat16 to torch.float16.
INFO 12-06 14:07:55 config.py:350] This model supports multiple tasks: {'embedding', 'generate'}. Defaulting to 'generate'.
WARNING 12-06 14:07:55 arg_utils.py:1013] Chunked prefill is enabled by default for models with max_model_len > 32K. Currently, chunked prefill might not work with some features or models. If you encounter any issues, please disable chunked prefill by setting --enable-chunked-prefill=False.
WARNING 12-06 14:07:55 arg_utils.py:1075] [DEPRECATED] Block manager v1 has been removed, and setting --use-v2-block-manager to True or False has no effect on vLLM behavior. Please remove --use-v2-block-manager in your engine argument. If your use case is not supported by SelfAttnBlockSpaceManager (i.e. block manager v2), please file an issue with detailed information.
INFO 12-06 14:07:55 config.py:1136] Chunked prefill is enabled with max_num_batched_tokens=512.
INFO 12-06 14:07:55 llm_engine.py:249] Initializing an LLM engine (v0.6.4.post1) with config: model='meta-llama/Llama-3.2-1B', speculative_config=None, tokenizer='meta-llama/Llama-3.2-1B', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=105088, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None, collect_model_forward_time=False, collect_model_execute_time=False), seed=0, served_model_name=meta-llama/Llama-3.2-1B, num_scheduler_steps=1, chunked_prefill_enabled=True multi_step_stream_outputs=True, enable_prefix_caching=False, use_async_output_proc=True, use_cached_outputs=True, chat_template_text_format=string, mm_processor_kwargs=None, pooler_config=None)
INFO 12-06 14:07:56 selector.py:261] Cannot use FlashAttention-2 backend for Volta and Turing GPUs.
INFO 12-06 14:07:56 selector.py:144] Using XFormers backend.
INFO 12-06 14:07:56 model_runner.py:1072] Starting to load model meta-llama/Llama-3.2-1B...
INFO 12-06 14:07:57 weight_utils.py:243] Using model weights format ['*.safetensors']
INFO 12-06 14:07:58 weight_utils.py:288] No model.safetensors.index.json found in remote.
Loading safetensors checkpoint shards:   0% Completed | 0/1 [00:00<?, ?it/s]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.56s/it]
Loading safetensors checkpoint shards: 100% Completed | 1/1 [00:01<00:00,  1.56s/it]

INFO 12-06 14:08:00 model_runner.py:1077] Loading model weights took 2.3185 GB
INFO 12-06 14:08:01 worker.py:232] Memory profiling results: total_gpu_memory=7.92GiB initial_memory_usage=2.73GiB peak_torch_memory=3.49GiB memory_usage_post_profile=2.76GiB non_torch_memory=0.44GiB kv_cache_size=3.60GiB gpu_memory_utilization=0.95
INFO 12-06 14:08:01 gpu_executor.py:113] # GPU blocks: 7366, # CPU blocks: 8192
INFO 12-06 14:08:01 gpu_executor.py:117] Maximum concurrency for 105088 tokens per request: 1.12x
INFO 12-06 14:08:04 model_runner.py:1400] Capturing cudagraphs for decoding. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 12-06 14:08:04 model_runner.py:1404] If out-of-memory error occurs during cudagraph capture, consider decreasing `gpu_memory_utilization` or switching to eager mode. You can also reduce the `max_num_seqs` as needed to decrease memory usage.
INFO 12-06 14:08:27 model_runner.py:1518] Graph capturing finished in 23 secs, took 0.24 GiB
INFO 12-06 14:08:27 api_server.py:249] vLLM to use /tmp/tmp9mjfahmu as PROMETHEUS_MULTIPROC_DIR
INFO 12-06 14:08:27 launcher.py:19] Available routes are:
INFO 12-06 14:08:27 launcher.py:27] Route: /openapi.json, Methods: GET, HEAD
INFO 12-06 14:08:27 launcher.py:27] Route: /docs, Methods: GET, HEAD
INFO 12-06 14:08:27 launcher.py:27] Route: /docs/oauth2-redirect, Methods: GET, HEAD
INFO 12-06 14:08:27 launcher.py:27] Route: /redoc, Methods: GET, HEAD
INFO 12-06 14:08:27 launcher.py:27] Route: /health, Methods: GET
INFO 12-06 14:08:27 launcher.py:27] Route: /tokenize, Methods: POST
INFO 12-06 14:08:27 launcher.py:27] Route: /detokenize, Methods: POST
INFO 12-06 14:08:27 launcher.py:27] Route: /v1/models, Methods: GET
INFO 12-06 14:08:27 launcher.py:27] Route: /version, Methods: GET
INFO 12-06 14:08:27 launcher.py:27] Route: /v1/chat/completions, Methods: POST
INFO 12-06 14:08:27 launcher.py:27] Route: /v1/completions, Methods: POST
INFO 12-06 14:08:27 launcher.py:27] Route: /v1/embeddings, Methods: POST
INFO:     Started server process [16458]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO 12-06 14:08:37 metrics.py:449] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
INFO 12-06 14:08:47 metrics.py:449] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
INFO 12-06 14:08:57 metrics.py:449] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
INFO 12-06 14:09:07 metrics.py:449] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
INFO 12-06 14:09:17 metrics.py:449] Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 0 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.0%, CPU KV cache usage: 0.0%.
INFO 12-06 14:09:27 logger.py:37] Received request cmpl-24f077ae65df4dc4baa7c6cec7b29240-0: prompt: 'Once upon a time,', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=0.5, top_p=1.0, top_k=-1, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=512, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None), prompt_token_ids: [128000, 12805, 5304, 264, 892, 11], lora_request: None, prompt_adapter_request: None.
INFO 12-06 14:09:27 engine.py:267] Added request cmpl-24f077ae65df4dc4baa7c6cec7b29240-0.
Unsupported conversion from f16 to f16
LLVM ERROR: Unsupported rounding mode for conversion.
ERROR 12-06 14:09:37 client.py:282] RuntimeError('Engine process (pid 16470) died.')
ERROR 12-06 14:09:37 client.py:282] NoneType: None
CRITICAL 12-06 14:09:39 launcher.py:99] MQLLMEngine is already dead, terminating server process
INFO:     Shutting down
INFO:     Waiting for application shutdown.
INFO:     Application shutdown complete.
INFO:     Finished server process [16458]
(base) xlisp@xlisp:~/jimw-vllm$

## can run
(base) xlisp@xlisp:~/jimw-vllm$ vllm serve "meta-llama/Llama-3.2-1B" --dtype=float32 --gpu-memory-utilization=0.95 --max-model-len=4096

vllm serve "meta-llama/Llama-3.2-1B" --dtype=float32 --gpu-memory-utilization=1.0 --max-model-len=8192

## test 
(base) xlisp@xlisp:~/jimw-vllm$ vllm serve "google/gemma-2-9b" --dtype=float32 --gpu-memory-utilization=1.0 --max-model-len=4096

ERROR 12-06 14:41:08 engine.py:366] ValueError: XFormers does not support attention logits soft capping.
INFO 12-06 14:41:06 selector.py:261] Cannot use FlashAttention-2 backend for Volta and Turing GPUs.

```
