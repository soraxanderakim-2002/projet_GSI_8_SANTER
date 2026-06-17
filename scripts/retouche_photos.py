from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageStat


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "retouches"
CUTOUT = OUTPUT_DIR / "personne_detouree.png"


TARGETS = [
    {
        "name": "WhatsApp Image 2026-06-17 at 2.40.25 PM.jpeg",
        "height": 455,
        "x": 22,
        "y": 355,
        "brightness": 0.93,
        "contrast": 0.94,
        "color": 0.92,
        "shadow": (14, 16, 32, 92),
    },
    {
        "name": "WhatsApp Image 2026-06-17 at 2.40.29 PM.jpeg",
        "height": 350,
        "x": 775,
        "y": 245,
        "flip": True,
        "brightness": 1.08,
        "contrast": 0.96,
        "color": 0.94,
        "shadow": (8, 10, 26, 76),
        "occlusions": [(720, 430, 1080, 640)],
    },
    {
        "name": "WhatsApp Image 2026-06-17 at 2.40.29 PM(1).jpeg",
        "height": 350,
        "x": 785,
        "y": 250,
        "flip": True,
        "brightness": 1.07,
        "contrast": 0.96,
        "color": 0.94,
        "shadow": (8, 10, 26, 76),
        "occlusions": [(720, 430, 1080, 640)],
    },
    {
        "name": "WhatsApp Image 2026-06-17 at 4.04.16 PM.jpeg",
        "height": 530,
        "x": 28,
        "y": 710,
        "angle": -8,
        "brightness": 0.98,
        "contrast": 0.96,
        "color": 0.94,
        "shadow": (12, 18, 34, 86),
        "occlusions": [(0, 1040, 960, 1280)],
    },
    {
        "name": "WhatsApp Image 2026-06-17 at 4.04.17 PM.jpeg",
        "height": 500,
        "x": 18,
        "y": 720,
        "angle": -11,
        "brightness": 0.89,
        "contrast": 0.92,
        "color": 0.88,
        "shadow": (12, 18, 34, 98),
        "occlusions": [(0, 960, 960, 1280)],
    },
    {
        "name": "WhatsApp Image 2026-06-17 at 4.16.54 PM.jpeg",
        "height": 450,
        "x": 128,
        "y": 350,
        "brightness": 0.86,
        "contrast": 0.91,
        "color": 0.88,
        "shadow": (10, 14, 28, 92),
    },
    {
        "name": "WhatsApp Image 2026-06-17 at 4.16.54 PM(1).jpeg",
        "height": 430,
        "x": 18,
        "y": 360,
        "brightness": 0.80,
        "contrast": 0.90,
        "color": 0.84,
        "shadow": (10, 14, 30, 112),
    },
]


def trim_alpha(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()
    return image.crop(bbox) if bbox else image


def fit_height(image: Image.Image, height: int) -> Image.Image:
    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def clean_alpha(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A")
    data = np.array(alpha)
    data[data < 8] = 0
    alpha = Image.fromarray(data, "L").filter(ImageFilter.GaussianBlur(0.35))
    result = image.copy()
    result.putalpha(alpha)
    return result


def adjust_subject(subject: Image.Image, *, brightness: float, contrast: float, color: float) -> Image.Image:
    alpha = subject.getchannel("A")
    rgb = subject.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
    rgb = ImageEnhance.Contrast(rgb).enhance(contrast)
    rgb = ImageEnhance.Color(rgb).enhance(color)
    result = rgb.convert("RGBA")
    result.putalpha(alpha)
    return result


def local_luminance_match(base: Image.Image, subject: Image.Image, x: int, y: int) -> Image.Image:
    visible_alpha = np.array(subject.getchannel("A")) > 64
    if not visible_alpha.any():
        return subject

    sx0 = max(0, x)
    sy0 = max(0, y)
    sx1 = min(base.width, x + subject.width)
    sy1 = min(base.height, y + subject.height)
    if sx1 <= sx0 or sy1 <= sy0:
        return subject

    patch = base.crop((sx0, sy0, sx1, sy1)).convert("L")
    target_luma = ImageStat.Stat(patch).mean[0]

    rgb = subject.convert("RGB")
    arr = np.array(rgb.convert("L"))
    src_luma = arr[visible_alpha].mean()
    if src_luma <= 1:
        return subject

    ratio = max(0.82, min(1.18, target_luma / src_luma))
    alpha = subject.getchannel("A")
    adjusted = ImageEnhance.Brightness(rgb).enhance(ratio)
    result = adjusted.convert("RGBA")
    result.putalpha(alpha)
    return result


def paste_shadow(base: Image.Image, subject: Image.Image, x: int, y: int, shadow_spec: tuple[int, int, int, int]) -> None:
    dx, dy, blur, opacity = shadow_spec
    alpha = subject.getchannel("A").filter(ImageFilter.GaussianBlur(blur))
    alpha = alpha.point(lambda value: min(opacity, value * opacity // 255))
    shadow = Image.new("RGBA", subject.size, (0, 0, 0, 0))
    shadow.putalpha(alpha)
    base.alpha_composite(shadow, (x + dx, y + dy))


def paste_original_regions(result: Image.Image, original: Image.Image, occlusions: list[tuple[int, int, int, int]]) -> None:
    for box in occlusions:
        region = original.crop(box)
        result.paste(region, box)


def build_subject(cutout: Image.Image, config: dict) -> Image.Image:
    subject = cutout
    if config.get("flip"):
        subject = subject.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    subject = fit_height(subject, config["height"])
    subject = adjust_subject(
        subject,
        brightness=config.get("brightness", 1.0),
        contrast=config.get("contrast", 1.0),
        color=config.get("color", 1.0),
    )
    if config.get("angle"):
        subject = subject.rotate(config["angle"], resample=Image.Resampling.BICUBIC, expand=True)
    return clean_alpha(trim_alpha(subject))


def compose_one(cutout: Image.Image, config: dict) -> Path:
    original = Image.open(ROOT / config["name"]).convert("RGBA")
    result = original.copy()
    subject = build_subject(cutout, config)
    x, y = config["x"], config["y"]
    subject = local_luminance_match(original, subject, x, y)

    paste_shadow(result, subject, x, y, config.get("shadow", (10, 12, 28, 80)))
    result.alpha_composite(subject, (x, y))
    paste_original_regions(result, original, config.get("occlusions", []))

    output = OUTPUT_DIR / f"retouche_{Path(config['name']).stem}.jpg"
    result.convert("RGB").save(output, quality=94, subsampling=1)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    cutout = trim_alpha(Image.open(CUTOUT).convert("RGBA"))
    for config in TARGETS:
        output = compose_one(cutout, config)
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
