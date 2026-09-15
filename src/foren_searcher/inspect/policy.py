"""INSPECT 运行时资源预算。"""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class InspectPolicy:
    """集中保存帧数、页数、模型输入和媒体大小等实现上限。"""

    default_frame_count: int = 8
    default_page_count: int = 8
    default_top_k: int = 5
    ocr_base_fps: float = 2.0
    ocr_refinement_s: float = 0.25
    backend_workers: int = 4
    execution_timeout_s: float = 300.0
    acquisition_timeout_s: float = 180.0
    maximum_file_bytes: int = 2 * 1024**3
    maximum_duration_s: float = 4 * 60 * 60
    maximum_pages: int = 64
    maximum_frames: int = 2000
    maximum_visual_qa_images: int = 8
    maximum_audio_qa_duration_s: float = 30.0
    maximum_model_image_pixels: int = 1_500_000
    maximum_image_pixels: int = 50_000_000
    pdf_dpi: int = 144
    maximum_page_pixels: int = 20_000_000
    maximum_download_bytes: int = 384 * 1024**2

    def __post_init__(self) -> None:
        """拒绝会关闭资源保护或产生无界行为的配置。"""

        for name in (
            "ocr_base_fps",
            "ocr_refinement_s",
            "execution_timeout_s",
            "acquisition_timeout_s",
            "maximum_duration_s",
            "maximum_audio_qa_duration_s",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")
        for name in (
            "default_frame_count",
            "default_page_count",
            "default_top_k",
            "backend_workers",
            "maximum_file_bytes",
            "maximum_download_bytes",
            "maximum_pages",
            "maximum_frames",
            "maximum_visual_qa_images",
            "maximum_model_image_pixels",
            "maximum_image_pixels",
            "pdf_dpi",
            "maximum_page_pixels",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        """转成后端 details 和审计记录可保存的普通字典。"""

        return asdict(self)
