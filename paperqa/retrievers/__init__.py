"""Concrete Retriever backends.

WHY: Mirrors paperqa.backends — keeps the core import path cheap and
network-free. Each retriever module owns its own optional dependency;
none are imported by paperqa.__init__.
"""
