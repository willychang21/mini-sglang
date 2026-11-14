"""mini-sglang: a small LLM inference engine built by following zero-to-sglang."""

from .core import SamplingParams

__all__ = ["SamplingParams", "LLM"]


def __getattr__(name):  # lazy import keeps `import minisgl` light for workers
    if name == "LLM":
        from .llm import LLM

        return LLM
    raise AttributeError(name)
