from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a privacy-safe NAFNet deblurring demonstration on a CCTV-style image."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--nafnet-root", default="outputs/build/nafnet-official")
    parser.add_argument(
        "--checkpoint",
        default=(
            "outputs/build/nafnet-official/experiments/pretrained_models/"
            "NAFNet-REDS-width64.pth"
        ),
    )
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--blur-length", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="docs/assets/nafnet_cctv_demo")
    args = parser.parse_args()
    if args.max_width <= 0 or args.blur_length <= 1 or args.blur_length % 2 == 0:
        raise SystemExit("--max-width must be positive and --blur-length must be odd and > 1")

    nafnet_root = _resolve(args.nafnet_root)
    sys.path.insert(0, str(nafnet_root))
    from basicsr.models.archs.NAFNet_arch import NAFNet

    input_path = _resolve(args.input)
    checkpoint_path = _resolve(args.checkpoint)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Cannot read input image: {input_path}")
    image = _privacy_safe_reference(image)
    if image.shape[1] > args.max_width:
        scale = args.max_width / image.shape[1]
        image = cv2.resize(
            image,
            (args.max_width, round(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    degraded = cv2.filter2D(image, -1, _motion_blur_kernel(args.blur_length))

    model = NAFNet(
        img_channel=3,
        width=64,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 28],
        dec_blk_nums=[1, 1, 1, 1],
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["params"], strict=True)
    device = torch.device(args.device)
    model = model.to(device).eval()
    tensor = torch.from_numpy(
        cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    ).unsqueeze(0).float().div(255.0).to(device)
    started = time.perf_counter()
    with torch.inference_mode(), (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    ):
        restored_tensor = model(tensor).clamp(0.0, 1.0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1000.0
    restored_rgb = (
        restored_tensor[0].float().cpu().numpy().transpose(1, 2, 0) * 255.0
    ).round().astype(np.uint8)
    restored = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR)

    degraded_psnr = peak_signal_noise_ratio(image, degraded, data_range=255)
    restored_psnr = peak_signal_noise_ratio(image, restored, data_range=255)
    degraded_ssim = structural_similarity(image, degraded, channel_axis=2, data_range=255)
    restored_ssim = structural_similarity(image, restored, channel_axis=2, data_range=255)
    _write_image(output_dir / "privacy_safe_reference.jpg", image)
    _write_image(output_dir / "synthetic_motion_blur.jpg", degraded)
    _write_image(output_dir / "nafnet_restored.jpg", restored)
    _write_image(
        output_dir / "comparison.jpg",
        _comparison_panel(
            degraded,
            restored,
            degraded_psnr,
            restored_psnr,
            degraded_ssim,
            restored_ssim,
        ),
    )
    report = {
        "method": "NAFNet-REDS-width64 pretrained deblurring",
        "input": input_path.name,
        "checkpoint": _portable_path(checkpoint_path),
        "privacy": {
            "plate": "pixelated before synthetic degradation and inference",
            "timestamp_and_address": "removed by inpainting before synthetic degradation and inference",
        },
        "evaluation_contract": (
            "The privacy-safe source is treated as the clean reference; a deterministic synthetic "
            "motion blur is applied before restoration. This is a restoration demo, not a traffic-event metric."
        ),
        "image_shape": list(image.shape),
        "blur_length": args.blur_length,
        "device": str(device),
        "latency_ms": latency_ms,
        "degraded": {"psnr_db": degraded_psnr, "ssim": degraded_ssim},
        "restored": {"psnr_db": restored_psnr, "ssim": restored_ssim},
        "delta": {
            "psnr_db": restored_psnr - degraded_psnr,
            "ssim": restored_ssim - degraded_ssim,
        },
        "source": "https://github.com/megvii-research/NAFNet",
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _privacy_safe_reference(image: np.ndarray) -> np.ndarray:
    result = image.copy()
    height, width = result.shape[:2]
    plate = _box(width, height, (0.40, 0.52, 0.61, 0.61))
    x1, y1, x2, y2 = plate
    region = result[y1:y2, x1:x2]
    if region.size:
        tiny = cv2.resize(region, (8, 4), interpolation=cv2.INTER_AREA)
        result[y1:y2, x1:x2] = cv2.resize(
            tiny, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
        )
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = _box(width, height, (0.50, 0.84, 1.0, 1.0))
    mask[y1:y2, x1:x2] = 255
    return cv2.inpaint(result, mask, 7, cv2.INPAINT_TELEA)


def _motion_blur_kernel(length: int) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    cv2.line(kernel, (1, length - 3), (length - 2, 2), 1.0, 1)
    return kernel / kernel.sum()


def _comparison_panel(
    degraded: np.ndarray,
    restored: np.ndarray,
    degraded_psnr: float,
    restored_psnr: float,
    degraded_ssim: float,
    restored_ssim: float,
) -> np.ndarray:
    panel = np.hstack([degraded, restored])
    height, width = degraded.shape[:2]
    cv2.rectangle(panel, (0, 0), (width * 2, 54), (20, 20, 20), -1)
    cv2.putText(
        panel,
        f"Synthetic blur  PSNR {degraded_psnr:.2f}  SSIM {degraded_ssim:.3f}",
        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"NAFNet restored  PSNR {restored_psnr:.2f}  SSIM {restored_ssim:.3f}",
        (width + 12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA,
    )
    return panel


def _box(
    width: int,
    height: int,
    normalized: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = normalized
    return round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)


def _write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode output image: {path}")
    encoded.tofile(str(path))


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


if __name__ == "__main__":
    raise SystemExit(main())
