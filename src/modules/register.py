"""Register CBAM with Ultralytics' parse_model.

Author: Rutuja Kulkarni

This module is the glue that realises the RADS Layer 3 architecture philosophy
of "custom YAML + monkey-patch over forking Ultralytics". The CBAM attention
block (src/modules/cbam.py) is a plain nn.Module; on its own Ultralytics' model
builder has no idea that the token `CBAM` in a model YAML refers to it, and —
crucially — does not know to apply its standard channel-scaling to that layer.
Calling `register_all()` once at process start-up teaches Ultralytics about
CBAM, after which any custom model YAML that lists a `CBAM` layer parses and
builds correctly. Every stage that constructs a CBAM-enabled model (the CBAM
ablation, the combined teacher, and the distillation teacher) calls this first.

Approach: monkey-patch parse_model by re-defining it with our module added
to the channel-aware set. We do this by reading the original source, adding
"CBAM" to the set literal, and exec'ing the modified source.

This sounds invasive but is actually the most robust path: we inherit all of
Ultralytics' channel-scaling, depth-scaling, and verbose-printing logic
unchanged, just with CBAM joining Conv/C2f/SPPF/... in the standard handling.
The alternative (forking Ultralytics or reimplementing parse_model) would have
to be re-synced on every upstream version bump; source-rewriting instead reuses
whatever the installed version's parse_model already does, and keeps the
enhancement a self-contained, reversible toggle.

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

# Module-level idempotency guard: register_all() may be called from several
# entry points, but the patch must be applied exactly once per process.
_REGISTERED = False


def register_all() -> None:
    """Make Ultralytics aware of the CBAM layer (idempotent).

    Performs the two steps needed for `CBAM` to be usable in a model YAML:
    (1) bind the CBAM class as an attribute of `ultralytics.nn.tasks` so the
    parser can look the name up, and (2) patch `parse_model` so CBAM is treated
    like a channel-aware layer (and thus gets proper width scaling). Subsequent
    calls are no-ops thanks to the `_REGISTERED` guard.

    Args:
        None.

    Returns:
        None.

    Side effects:
        Mutates the imported `ultralytics.nn.tasks` module in place (adds a
        `CBAM` attribute and replaces its `parse_model`), sets the module-level
        `_REGISTERED` flag, and prints a confirmation line.
    """
    global _REGISTERED
    if _REGISTERED:
        # Already patched in this process; nothing more to do.
        return
    # Step 1: expose CBAM by name on the tasks module so the parser can resolve
    # the literal `CBAM` token from a YAML to this actual class.
    setattr(tasks_mod, "CBAM", CBAM)
    # Step 2: rewrite parse_model so CBAM joins the channel-aware layer set.
    _patch_parse_model_source()
    _REGISTERED = True
    print("[register] CBAM module registered with Ultralytics parser")


def _patch_parse_model_source() -> None:
    """Re-define parse_model by source-rewriting: add CBAM to its channel-set.

    This is the actual monkey-patch. Ultralytics' `parse_model` contains a set
    literal (the first `if m in { ... }`) listing the layer types whose first
    arg is an input-channel count and whose `c2` should be width-scaled. CBAM
    belongs in that set, but we cannot edit the installed library file. Instead
    we read parse_model's source text, splice `CBAM,` into that set literal, and
    re-exec the modified source to obtain a new function object that behaves
    identically to the original except that CBAM now receives the standard
    channel handling. The freshly compiled function is then swapped back onto
    the tasks module.

    Args:
        None.

    Returns:
        None.

    Raises:
        RuntimeError: if the expected `Conv,` marker is absent from the source
            (e.g. an incompatible Ultralytics version), so the failure is loud
            rather than a silently mis-built model.

    Side effects:
        Replaces `tasks_mod.parse_model` and sets `tasks_mod._RADS_PARSER_PATCHED`.
    """
    # Per-module guard (distinct from _REGISTERED) so re-importing this module
    # without re-running register_all() still cannot double-patch parse_model.
    if getattr(tasks_mod, "_RADS_PARSER_PATCHED", False):
        return

    # Grab the live source of the installed parse_model and normalise its
    # indentation so it compiles standalone at module scope (dedent strips the
    # common leading whitespace that inspect.getsource preserves).
    src = inspect.getsource(tasks_mod.parse_model)
    src = textwrap.dedent(src)

    # Inject CBAM into the FIRST `if m in {` set (the channel-aware one).
    # We find the exact string `Conv,` (with the comma + newline) inside that
    # set literal and insert `CBAM,` right after it.
    marker = "Conv,"
    # find() returns the offset of the first occurrence, which lives inside the
    # first (channel-aware) set literal — exactly where CBAM needs to go.
    idx = src.find(marker)
    if idx < 0:
        # Upstream renamed/moved Conv: refuse to proceed rather than guess.
        raise RuntimeError("Could not find Conv marker in parse_model source")
    # Make sure we're inside the first set, not later. The first `if m in {`
    # occurs once; we splice just past the Conv line.
    # Locate the end of the line that holds `Conv,` and insert a new `CBAM,`
    # line right after it, copying the source's 12-space indentation so the
    # rewritten set literal stays syntactically valid.
    line_end = src.find("\n", idx)
    new_src = src[:line_end + 1] + "            CBAM,\n" + src[line_end + 1:]

    # Build a namespace that mirrors what `tasks` module sees.
    # Copying tasks_mod's globals means every name parse_model references
    # (other layer classes, helpers, torch, etc.) resolves exactly as it would
    # inside the real module — the rewritten function is a faithful clone.
    ns = dict(vars(tasks_mod))
    ns["CBAM"] = CBAM  # so the rewritten source can resolve it

    # Compile + exec the patched source in that namespace, then lift out the
    # newly defined parse_model and install it over the original.
    exec(compile(new_src, "<rads_parse_model_patched>", "exec"), ns)
    new_parse = ns["parse_model"]

    tasks_mod.parse_model = new_parse
    # Mark the module so the guard above short-circuits any future patch attempt.
    tasks_mod._RADS_PARSER_PATCHED = True
