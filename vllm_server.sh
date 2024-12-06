(base) xlisp@xlisp:~/jimw-vllm$ vllm serve Qwen/Qwen2.5-1.5B-Instruct --dtype=half

(base) xlisp@xlisp:~/jimw-vllm$ curl http://localhost:8000/v1/completions \
>     -H "Content-Type: application/json" \
>     -d '{
>         "model": "Qwen/Qwen2.5-1.5B-Instruct",
>         "prompt": "hi",
>         "max_tokens": 7,
>         "temperature": 0
>     }'
{"id":"cmpl-4c55bf66b51a4903aec3335ca064ac15","object":"text_completion","created":1733470360,"model":"Qwen/Qwen2.5-1.5B-Instruct","choices":[{"index":0,"text":"!!!!!!!","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":1,"total_tokens":8,"completion_tokens":7,"prompt_tokens_details":null}}(base) xlisp@xlisp:~/jimw-vllm$
(base) xlisp@xlisp:~/jimw-vllm$ curl http://localhost:8000/v1/models
{"object":"list","data":[{"id":"Qwen/Qwen2.5-1.5B-Instruct","object":"model","created":1733470369,"owned_by":"vllm","root":"Qwen/Qwen2.5-1.5B-Instruct","parent":null,"max_model_len":32768,"permission":[{"id":"modelperm-251a31d7a6824bf19fbf76ddef577a30","object":"model_permission","created":1733470369,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}(base) xlisp@xlisp:~/jimw-vllm$
(base) xlisp@xlisp:~/jimw-vllm$

