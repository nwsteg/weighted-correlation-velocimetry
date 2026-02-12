"""Backward-compatible notebook entrypoint for the PLIF interactive explorer.

Prefer importing from `wcv` directly:

    from wcv import make_plif_interactive_widget
"""

from wcv.interactive import load_hardcoded_plif, make_plif_interactive_widget

__all__ = ["load_hardcoded_plif", "make_plif_interactive_widget"]
