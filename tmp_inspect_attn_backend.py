from diffusers.utils.constants import DIFFUSERS_ATTN_BACKEND, DIFFUSERS_ATTN_CHECKS
from diffusers.models.attention_dispatch import _AttentionBackendRegistry
print(f"DIFFUSERS_ATTN_BACKEND={DIFFUSERS_ATTN_BACKEND}")
print(f"DIFFUSERS_ATTN_CHECKS={DIFFUSERS_ATTN_CHECKS}")
print(f"ACTIVE_BACKEND={_AttentionBackendRegistry.get_active_backend()[0]}")
