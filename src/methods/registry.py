from __future__ import annotations

"""
Author: Gourav Singla
Date: 2025-12-18
Description: Simple registry for chunking methods to enable CLI selection and discovery.
Paper Inspiration: Internal utility (not from a paper); supports comparing multiple chunkers.
"""

from collections.abc import Callable
from dataclasses import dataclass


Chunker = Callable[[str, int], str]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    chunker: Chunker
    description: str


METHOD_REGISTRY: dict[str, MethodSpec] = {}


def register_method(name: str, *, description: str) -> Callable[[Chunker], Chunker]:
    """Decorator to register a chunking function by name."""

    def decorator(func: Chunker) -> Chunker:
        if name in METHOD_REGISTRY:
            raise ValueError(f"Method already registered: {name}")
        METHOD_REGISTRY[name] = MethodSpec(name=name, chunker=func, description=description)
        return func

    return decorator


def list_methods() -> list[MethodSpec]:
    return [METHOD_REGISTRY[name] for name in sorted(METHOD_REGISTRY.keys())]


def get_method(name: str) -> MethodSpec:
    try:
        return METHOD_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(METHOD_REGISTRY.keys()))
        raise KeyError(f"Unknown method '{name}'. Available: {available}") from exc
