## ! if no login , will get this erro: requests.exceptions.HTTPError: The request model: google/gemma-2-9b does not exist!

(base) ➜  learn-vllm git:(master) pip install modelscope  # TODO: learn

(base) ➜  learn-vllm git:(master) git config --global credential.helper store

(base) ➜  learn-vllm git:(master) huggingface-cli login                      

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    A token is already saved on your machine. Run `huggingface-cli whoami` to get more information or `huggingface-cli logout` if you want to log out.
    Setting a new token will erase the existing one.
    To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Enter your token (input will not be visible): 
Add token as git credential? (Y/n) y
Token is valid (permission: fineGrained).
The token `stevevllm` has been saved to /home/xlisp/.cache/huggingface/stored_tokens
Your token has been saved in your configured git credential helpers (store).
Your token has been saved to /home/xlisp/.cache/huggingface/token
Login successful.
The current active token is: `stevevllm`
(base) ➜  learn-vllm git:(master)

(base) ➜  learn-vllm git:(master) ✗ vllm serve "meta-llama/Llama-3.2-1B" --dtype=float32 --gpu-memory-utilization=1.0 --max-model-len=8192

