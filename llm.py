from langchain_community.llms import LlamaCpp
import warnings
warnings.filterwarnings("ignore")

class LLM():
    def __init__(self):
        n_gpu_layers = -1  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
        n_batch = 4000  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
        model_path = "/home/gabe/.cache/huggingface/hub/models--mistralai--Mistral-7B-Instruct-v0.2/gguf/mistral-7b-instruct-v0.2.Q8_0.gguf"

        config = {
            "model_path": model_path,
            "n_gpu_layers": n_gpu_layers,
            "n_batch": n_batch,
            "f16_kv": True,
            "grammar_path": "./data/json.gbnf",
            "n_ctx": 32000,
            "rope_freq_base": 1e6,
            "verbose": False,
            "echo": False,
            "task": "text-generation",
            "do_sample": False,
            "max_tokens": 4000,
            "repetition_penalty": 1.15,
            "attention_dropout": 0.0,
            "eos_token_id": 2,
            "bos_token_id": 1,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 14336,
            "max_position_embeddings": 32768,
            "model_type": "mistral",
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.36.0",
            "use_cache": True,
            "vocab_size": 32000
        }

        # Make sure the model path is correct for your system!
        self.llm_instance = LlamaCpp(**config)

