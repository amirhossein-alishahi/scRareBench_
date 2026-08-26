"""Backward-compatible shim for the runtime helper now shipped inside scRareBench.

New notebooks should use ``from scrarebench.runtime import setup_notebook``.
"""
from scrarebench.runtime import (  # noqa: F401
    CORE_SMOKE_IMPORTS,
    DEFAULT_ANCHORS,
    METHOD_RUNTIME_PROFILES,
    MethodRuntimeProfile,
    RuntimeInstallReport,
    available_methods,
    get_method_profile,
    install_notebook_runtime,
    print_install_report,
    setup_notebook,
)

__all__ = [
    "CORE_SMOKE_IMPORTS",
    "DEFAULT_ANCHORS",
    "METHOD_RUNTIME_PROFILES",
    "MethodRuntimeProfile",
    "RuntimeInstallReport",
    "available_methods",
    "get_method_profile",
    "install_notebook_runtime",
    "print_install_report",
    "setup_notebook",
]
