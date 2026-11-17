# -*- coding: utf-8 -*-
"""Octo — small offline GPT-style language model built from scratch (PyTorch).

Train:    python -m octo.train --corpus data/octo/corpus.txt --steps 2000
Generate: python -m octo.chat --checkpoint octo_checkpoints/octo.pt --prompt "hello"
"""
from .model import OctoConfig, OctoModel
from .tokenizer import CharTokenizer

__version__ = "0.1.0"
__all__ = ["OctoConfig", "OctoModel", "CharTokenizer", "__version__"]
