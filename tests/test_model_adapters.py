from app.services.clip_model import CLIPConceptExtractor
from app.services.llm_report import DeepSeekReportSynthesizer
from app.services.segmentation_model import UltraLightVMUNetAdapter
from app.services.vision_model import VisionTransformerAdapter


def test_vision_model_adapter_reports_backbone_and_status():
    result = VisionTransformerAdapter().predict()
    assert result.backbone == "google/vit-base-patch16-224"
    assert result.status in {"available", "fallback"}


def test_segmentation_adapter_uses_fallback_cleanly():
    result = UltraLightVMUNetAdapter().segment("dummy.png")
    assert result.status in {"fallback", "ready"}
    assert result.quality_score >= 0.0


def test_clip_and_llm_adapters_are_configurable():
    clip = CLIPConceptExtractor().extract(
        image_features={"texture": 0.8},
        clinical_features={"border_irregularity": 0.7, "asymmetry": 0.6, "color": {"index": 0.8}},
    )
    llm = DeepSeekReportSynthesizer().generate("High risk lesion with asymmetric border.")

    assert "concepts" in clip
    assert llm["format"] == "medical_style_report"
    assert llm["status"] in {"ready", "fallback"}
