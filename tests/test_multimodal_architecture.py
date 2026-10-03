from app.services.multimodal import run_full_image_pipeline, run_multimodal_pipeline


def test_multimodal_pipeline_exposes_each_stage():
    result = run_multimodal_pipeline(
        {
            "melanoma_probability": 0.72,
            "risk_level": "high",
            "predicted_class": "mel",
        },
        {
            "asymmetry": 0.7,
            "border_irregularity": 0.8,
            "color": {"index": 0.65},
        },
        {"change_detected": True},
    )

    assert set(result.keys()) >= {
        "vision_transformer",
        "segmentation",
        "abc_features",
        "contrastive_alignment",
        "clip_concepts",
        "report",
    }
    assert result["vision_transformer"]["backbone"] == "google/vit-base-patch16-224"
    assert result["segmentation"]["model"] == "UltraLight VM-UNet + SAM2"
    assert "UltraLight-VM-UNet" in str(result["segmentation"]["vmunet"])
    assert "SAM2" in str(result["segmentation"]["sam2"])
    assert result["contrastive_alignment"]["score"] >= 0.0
    assert "asymmetry" in result["report"]["summary"].lower()


def test_full_image_pipeline_executes_for_real_image_path():
    image_path = "data/uploads/effdb754e96f47f3a0c523ebd153767b.png"

    result = run_full_image_pipeline(image_path)

    import os
    assert os.path.normpath(result["input_image"]) == os.path.normpath(os.path.abspath(image_path)) or result["input_image"] == image_path
    assert result["segmentation"]["model"] == "UltraLight VM-UNet + SAM2"
    assert "summary" in result["report"]
