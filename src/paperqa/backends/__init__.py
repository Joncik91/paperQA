"""Concrete Answerer backends.

WHY: Kept in a subpackage so the core grounding contract (paperqa.answering)
stays import-cheap and network-free. Each backend module owns its own
optional dependency; none of them are imported by paperqa.__init__.
"""
