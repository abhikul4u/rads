"""Register CBAM with Ultralytics' parse_model.

Approach: monkey-patch parse_model by re-defining it with our module added
to the channel-aware set. We do this by reading the original source, adding
"CBAM" to the set literal, and exec'ing the modified source.

This sounds invasive but is actually the most robust path: we inherit all of
Ultralytics' channel-scaling, depth-scaling, and verbose-printing logic
unchanged, just with CBAM joining Conv/C2f/SPPF/... in the standard handling.

YAML convention (matches Conv exactly):
    [-1, 1, CBAM, [c2, reduction, kernel_size]]
where c2 is the desired output channels (== input channels for CBAM). The
parser scales c2 by width_multiple just like it does for Conv's c2.

CBAM ctor accepts (c1, c2, reduction, kernel_size). c1 == c2 always; we keep
c2 in the signature so the parser's args-rewriting just works.
"""
from __future__ import annotations

import inspect
import textwrap

import ultralytics.nn.tasks as tasks_mod

from src.modules.cbam import CBAM

_REGISTERED = False


def register_all() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    setattr(tasks_mod, "CBAM", CBAM)
    _patch_parse_model_source()
    _REGISTERED = True
    print("[register] CBAM module registered with Ultralytics parser")


def _patch_parse_model_source() -> None:
    """Re-define parse_model by source-rewriting: add CBAM to its channel-set."""
    if getattr(tasks_mod, "_RADS_PARSER_PATCHED", False):
        return

    src = inspect.getsource(tasks_mod.parse_model)
    src = textwrap.dedent(src)

    # Inject CBAM into the FIRST `if m in {` set (the channel-aware one).
    # We find the exact string `Conv,` (with the comma + newline) inside that
    # set literal and insert `CBAM,` right after it.
    marker = "Conv,"
    idx = src.find(marker)
    if idx < 0:
        raise RuntimeError("Could not find Conv marker in parse_model source")
    # Make sure we're inside the first set, not later. The first `if m in {`
    # occurs once; we splice just past the Conv line.
    line_end = src.find("\n", idx)
    new_src = src[:line_end + 1] + "            CBAM,\n" + src[line_end + 1:]

    # Build a namespace that mirrors what `tasks` module sees.
    ns = dict(vars(tasks_mod))
    ns["CBAM"] = CBAM  # so the rewritten source can resolve it

    exec(compile(new_src, "<rads_parse_model_patched>", "exec"), ns)
    new_parse = ns["parse_model"]

    tasks_mod.parse_model = new_parse
    tasks_mod._RADS_PARSER_PATCHED = True
