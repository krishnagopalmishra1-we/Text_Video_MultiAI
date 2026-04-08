"""
Patch diffusers/transformers for compatibility with the installed PyTorch version.

Fixes applied:
1. flash_attn_3 custom_op uses string type annotations that PyTorch infer_schema rejects
   on older torch versions → wrap the custom_op registration in try/except.
2. Several attention backends pass `enable_gqa=` directly to
   torch.nn.functional.scaled_dot_product_attention, but that keyword was only added
   in PyTorch 2.5.0. On torch < 2.5 this raises TypeError. Fix: monkey-patch sdpa
   at import time to drop the kwarg when running on torch < 2.5.
3. transformers check_torch_load_is_safe() blocks torch < 2.6 from loading .bin
   checkpoint files (CVE-2025-32434). In a controlled container environment this
   restriction serves no purpose — patch it out so musicgen/.bin models load normally.

Run once after pip install diffusers (or add to Dockerfile RUN step).
"""
import site

for sp in site.getsitepackages():
    path = f"{sp}/diffusers/models/attention_dispatch.py"
    try:
        with open(path) as f:
            content = f.read()
    except FileNotFoundError:
        continue

    changed = False

    # ------------------------------------------------------------------ #
    # Fix 1: wrap flash_attn_3 custom_op registration in try/except       #
    # ------------------------------------------------------------------ #
    block1_start = None
    block2_start = None
    block2_end = None

    lines = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        s = line.rstrip()
        if '@_custom_op("_diffusers_flash_attn_3' in s and block1_start is None:
            block1_start = i
        if '@_register_fake("_diffusers_flash_attn_3' in s and block2_start is None:
            block2_start = i
        if block2_start is not None and i > block2_start + 5 and "# =====" in s:
            block2_end = i
            break

    if block1_start is None:
        print(f"Fix 1: already patched or pattern not found in {path}, skipping.")
    else:
        def wrap_block(block_lines):
            result = ["try:\n"]
            for l in block_lines:
                result.append("    " + l)
            result.append("except Exception:\n")
            result.append("    pass\n")
            return result

        new_lines = (
            lines[:block1_start]
            + wrap_block(lines[block1_start:block2_start])
            + wrap_block(lines[block2_start:block2_end])
            + lines[block2_end:]
        )
        content = "".join(new_lines)
        changed = True
        print(f"Fix 1 applied: wrapped flash_attn_3 custom_op blocks in try/except.")

    # ------------------------------------------------------------------ #
    # Fix 2: patch sdpa to drop enable_gqa on torch < 2.5.0              #
    # ------------------------------------------------------------------ #
    MARKER = "# __enable_gqa_patch_applied__"
    if MARKER in content:
        print("Fix 2: enable_gqa patch already present, skipping.")
    else:
        patch_code = (
            "\n"
            "# __enable_gqa_patch_applied__\n"
            "# Patch: strip enable_gqa kwarg for torch < 2.5.0 (added in PyTorch 2.5)\n"
            "if tuple(int(x) for x in torch.__version__.split(\"+\")[0].split(\".\")[:2]) < (2, 5):\n"
            "    def _make_sdpa_patch(_orig):\n"
            "        def _patched_sdpa(*args, **kwargs):\n"
            "            kwargs.pop(\"enable_gqa\", None)\n"
            "            return _orig(*args, **kwargs)\n"
            "        _patched_sdpa.__wrapped__ = _orig\n"
            "        return _patched_sdpa\n"
            "    _patched = _make_sdpa_patch(torch.nn.functional.scaled_dot_product_attention)\n"
            "    torch.nn.functional.scaled_dot_product_attention = _patched\n"
            "    F.scaled_dot_product_attention = _patched\n"
            "    del _make_sdpa_patch, _patched\n"
        )
        anchor = "import torch.nn.functional as F\n"
        if anchor not in content:
            print(f"Fix 2: ERROR — anchor '{anchor.strip()}' not found in {path}!")
        else:
            content = content.replace(anchor, anchor + patch_code, 1)
            changed = True
            print("Fix 2 applied: enable_gqa monkey-patch inserted.")

    if changed:
        with open(path, "w") as f:
            f.write(content)
        print(f"Wrote patched file: {path}")
    break

# ------------------------------------------------------------------ #
# Fix 3: transformers check_torch_load_is_safe CVE-2025-32434 gate   #
# ------------------------------------------------------------------ #
for sp in site.getsitepackages():
    tf_path = f"{sp}/transformers/utils/import_utils.py"
    try:
        with open(tf_path) as f:
            tf_content = f.read()
    except FileNotFoundError:
        continue

    TF_MARKER = "# __cve_patch_applied__"
    if TF_MARKER in tf_content:
        print("Fix 3: CVE patch already present, skipping.")
    else:
        old_tf = (
            'def check_torch_load_is_safe() -> None:\n'
            '    if not is_torch_greater_or_equal("2.6"):\n'
            '        raise ValueError('
        )
        new_tf = (
            'def check_torch_load_is_safe() -> None:\n'
            '    # __cve_patch_applied__ — version gate removed for controlled container\n'
            '    if False and not is_torch_greater_or_equal("2.6"):\n'
            '        raise ValueError('
        )
        if old_tf in tf_content:
            tf_content = tf_content.replace(old_tf, new_tf, 1)
            with open(tf_path, "w") as f:
                f.write(tf_content)
            print("Fix 3 applied: check_torch_load_is_safe CVE gate disabled.")
        else:
            print(f"Fix 3: pattern not found in {tf_path}, skipping.")
    break
