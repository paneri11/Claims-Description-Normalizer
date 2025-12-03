from typing import List, Optional
from llama_cpp import Llama

# Path to your GGUF model (update this path as per your system)
MODEL_PATH = "models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Singleton-style global LLM instance
_llm: Optional[Llama] = None


def get_llm() -> Llama:
    """
    Lazily load and return a singleton Llama instance.
    This avoids reloading the model on every import/call.
    """
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,
            n_gpu_layers=35,  # adjust based on your GPU
            verbose=False,
        )
    return _llm


def generate_completion(
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.1,
    stop: Optional[List[str]] = None,
) -> str:
    """
    Generic helper to call the local LLM and return the text output.
    """
    llm = get_llm()
    response = llm.create_completion(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop or ["</s>", "\n\n\n"],
    )
    return response["choices"][0]["text"]
