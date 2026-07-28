"""Verify that prompts longer than CLIP's 77 tokens actually reach the model.

Background: SDXL's CLIP encoders have a hard 77-token context. picture-ai
configures compel with `truncate_long_prompts=False` to chunk past that — but
compel's SDXL path raised AttributeError on `EmbeddingsProviderMulti.empty_z`,
`_encode_prompts` swallowed it, and generation silently fell back to the raw
(truncated) prompt. See pipeline_manager._compel_padding.

This test proves three things against a really-loaded SDXL pipeline:
  1. the upstream bug is real (the unpatched call still raises),
  2. our padding tensor fixes it, and
  3. the resulting embeddings are longer than 77 tokens, i.e. the tail of a
     long prompt is genuinely conditioning the model.

    venv\\Scripts\\python.exe scripts\\test_compel_longprompt.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from picture_ai.logging_utils import configure_logging  # noqa: E402
from picture_ai.pipeline_manager import PipelineManager  # noqa: E402

MODEL = "RunDiffusion/Juggernaut-XL-v9"

# Comfortably over 77 CLIP tokens, with a distinctive tail that would be the
# first thing lost to truncation.
LONG_PROMPT = (
    "a stunning highly detailed photorealistic portrait of a weathered sea "
    "captain standing on the deck of a wooden sailing ship during a violent "
    "storm at dusk, rain streaming down his face, thick grey beard, deep set "
    "eyes, heavy oilskin coat with brass buttons, rigging and torn sails "
    "behind him, dramatic rim lighting from a swinging lantern, volumetric "
    "spray, 85mm lens, shallow depth of field, cinematic colour grading, "
    "and finally a small red glass bead tied into the very end of his beard"
)
SHORT_NEGATIVE = "blurry, low quality"

FAILURES = []


def check(label, condition, detail=""):
    print(f"  [{'ok  ' if condition else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(label)


def main() -> int:
    logger = configure_logging(ROOT / "logs" / "compel_test.log")
    pm = PipelineManager(models_root=ROOT / "models_cache", logger=logger)
    print(f"loading {MODEL} ...")
    pm.ensure_pipeline(MODEL, None)

    tokenizer = pm._pipeline.tokenizer
    n_tokens = len(tokenizer(LONG_PROMPT).input_ids)
    print(f"prompt is {n_tokens} CLIP tokens (limit is {tokenizer.model_max_length})\n")
    check("test prompt actually exceeds the limit", n_tokens > 77, f"{n_tokens} tokens")

    compel = pm._get_compel()
    check("compel initialised", compel is not None)
    if compel is None:
        return 1

    cond, _ = compel(LONG_PROMPT)
    neg_cond, _ = compel(SHORT_NEGATIVE)
    print(f"raw conditioning shapes: positive={tuple(cond.shape)} negative={tuple(neg_cond.shape)}")
    check("long prompt produced >77 conditioning rows (chunking works)",
          cond.shape[1] > 77, f"{cond.shape[1]} rows")
    check("shapes genuinely differ (this is what triggers the bug)",
          cond.shape != neg_cond.shape)

    # 1. the upstream bug still exists — this is what used to reach the user
    try:
        compel.pad_conditioning_tensors_to_same_length([cond, neg_cond])
        bug_present = False
        detail = "no exception — upstream compel may have been fixed"
    except AttributeError as exc:
        bug_present = True
        detail = str(exc)[:80]
    check("unpatched compel call still fails (bug is real)", bug_present, detail)

    # 2. our padding tensor fixes it
    padding = PipelineManager._compel_padding(compel)
    check("padding tensor built", padding is not None,
          str(tuple(padding.shape)) if padding is not None else "None")
    padded = compel.pad_conditioning_tensors_to_same_length(
        [cond, neg_cond], precomputed_padding=padding
    )
    check("padding produced equal shapes", padded[0].shape == padded[1].shape,
          f"{tuple(padded[0].shape)} vs {tuple(padded[1].shape)}")

    # 3. the real code path now returns embeddings instead of falling back
    embeds = pm._encode_prompts(LONG_PROMPT, SHORT_NEGATIVE)
    check("_encode_prompts returned embeddings (no silent truncation)",
          embeds is not None)
    if embeds is not None:
        rows = embeds["prompt_embeds"].shape[1]
        check("embeddings carry the whole prompt (>77 rows)", rows > 77, f"{rows} rows")
        check("negative padded to match",
              embeds["prompt_embeds"].shape == embeds["negative_prompt_embeds"].shape)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
        return 1
    print("long prompts reach the model — truncation lifted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
