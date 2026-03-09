"""Inference backend abstractions.

Each backend (Ollama, llama.cpp, MLX, …) implements the ``InferenceBackend``
protocol from ``protocol.py``.  Application code interacts with backends
exclusively through this protocol.
"""
