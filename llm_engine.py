from typing import List, Optional
from llama_cpp import Llama


MODEL_PATH = "models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"

_llm: Optional[Llama] = None


def get_llm() -> Llama:
    """
    Lazily load a single global Llama instance, configured to use GPU
    like in your original working script.
    """
    global _llm
    if _llm is None:
        print("[LLM] Loading model from:", MODEL_PATH)
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=4096,       # same as your original
            n_gpu_layers=35,  # ✅ use GPU (like original app (1).py)
            verbose=False,
        )
        print("[LLM] Model loaded successfully.")
    return _llm


def generate_completion(
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.1,
    stop: Optional[List[str]] = None,
) -> str:
    """
    Wrapper around llama_cpp Llama.create_completion, returning only text.
    """
    llm = get_llm()
    response = llm.create_completion(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop or ["</s>", "\n\n"],
    )
    return response["choices"][0]["text"]
