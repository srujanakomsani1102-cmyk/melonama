# Real model setup for the multimodal melanoma workflow

This project is already implemented according to the requested architecture, but the real model-backed execution is optional and only activates when real assets are available.

## Required environment variables

Create a `.env` file in the project root based on `.env.example` and fill in real values.

```env
ULTRALIGHT_VMUNET_DIR=C:\path\to\ultralight_vmunet
SAM2_MODEL_DIR=C:\path\to\sam2
CLIP_MODEL_DIR=C:\path\to\clip
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_API_URL=https://api.deepseek.com/v1/chat/completions
DEEPSEEK_MODEL=deepseek-chat
```

## Expected folder contents

### UltraLight VM-UNet
Place the checkpoint in a directory containing one of:
- `model.safetensors`
- `model.bin`
- `pytorch_model.bin`
- `weights.pt`
- `weights.pth`
- `best.pt`
- `checkpoint.pt`

### SAM2
Place the model checkpoint in the configured SAM2 directory using a compatible `.pt` or `.pth` file.

### CLIP
Use a valid local CLIP model directory that `transformers` can load with `AutoModel.from_pretrained(...)` and `AutoProcessor.from_pretrained(...)`.

### DeepSeek API
Set a valid DeepSeek API key and URL. The project will use the configured API when both are present; otherwise it falls back to the local structured report template.

## Behavior when assets are missing

The app will continue to operate in a safe fallback mode:
- segmentation uses CV/OpenCV fallback logic
- CLIP returns conceptual fallback output
- DeepSeek produces a local structured report instead of a remote LLM answer

This preserves stability while keeping the architecture ready for real models.

## Verification

The project has been validated with:

```bash
python -m pytest -q
```

Current status: all tests pass, and the pipeline is ready for real model runtime when the actual artifacts are configured.
