# ComfyUI Quick Start — Wan2.1 Ready-to-Use Workflows

You now have **2 pre-built workflows** ready to load. ComfyUI uses JSON workflows that chain nodes together automatically.

## Option 1: Use Web UI Import (Easiest)

1. **Open ComfyUI**: http://172.177.139.168:8188/
2. **Load a workflow**:
   - Click `Load` (top left)
   - Paste the JSON from `workflows/wan2_basic.json` or `workflows/wan2_cinematic.json`
   - Click `Queue Prompt` to generate

## Option 2: Copy Workflow Files Directly

The workflow files are already in your local `d:\Video\project\workflows/`:
- `wan2_basic.json` — simple 25-step generation, 5s video
- `wan2_cinematic.json` — advanced settings, upscale option, better quality

## Workflow Breakdown

Both workflows do this automatically:

```
Text Prompt
    ↓
CLIPTextEncode (converts text to embeddings)
    ↓
WanVideoWrapper (loads Wan2.1 model + text encoder)
    ↓
KSampler (runs diffusion: 25-30 steps)
    ↓
VAE Decode (converts latents to video frames)
    ↓
Video Output (saves as MP4)
```

**No manual node-building required.** Just:
1. Change the prompt text
2. Click "Queue Prompt"
3. Wait for output MP4

## Changing Parameters (In ComfyUI UI)

Once workflow is loaded, right-click any node to edit:
- **Prompt**: Change the text
- **Steps**: 25-50 (more = better quality, slower)
- **CFG**: 6-9 (higher = follow prompt more strictly)
- **Seed**: For reproducibility
- **Resolution**: 1280x720 (default)
- **Frame count**: 81 frames = ~5s @ 16fps

## Example Prompts to Try

```
"A golden retriever running through sunlit forest, 4K, cinematic"
"Ocean waves crashing on rocky beach at sunset, ultra realistic"
"A red sports car driving smoothly down coastal highway, cinematic lighting"
"Waterfall in lush green jungle, mist in air, golden hour light"
```

## Troubleshooting

**Workflow won't load**: Copy the JSON text, click Load, paste it into the text box that appears

**Missing nodes**: Make sure WanVideoWrapper custom node is installed (should be auto-loaded)

**CUDA out of memory**: Reduce steps (20→15) or resolution (1280x720→960x544)

**Slow generation**: That's normal—Wan2.1 takes 2-5 minutes per video on A100

## Next Steps

1. Load `wan2_basic.json` in ComfyUI now
2. Keep the default prompt or change it
3. Click "Queue Prompt"
4. Monitor at http://172.177.139.168:8188/
5. Output saves to `/outputs/` on the VM

Questions? Workflows are 100% ready—no coding needed.
