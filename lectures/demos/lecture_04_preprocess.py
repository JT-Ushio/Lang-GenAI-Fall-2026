"""Offline LLaVA-1.5 image preprocessing and shape demonstrations.

Uses checked-in official JSON configs, synthetic teaching images, Pillow,
NumPy and PyTorch. Does not load pretrained vision/language weights.
"""
from pathlib import Path
from html import escape
import base64
import json
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent / "lecture_04_configs"
OUT = ROOT / "images/lecture_04/preprocess"


def load_configs():
    llava = json.loads((CONFIG_DIR / "llava_15_7b_config.json").read_text())
    vision = json.loads((CONFIG_DIR / "clip_l14_336_config.json").read_text())["vision_config"]
    processor = json.loads((CONFIG_DIR / "clip_l14_336_preprocessor_config.json").read_text())
    return llava, vision, processor


def load_rgb(path, alpha_background=(255, 255, 255)):
    # EXIF records display orientation; this makes the pixel order agree with it.
    with Image.open(path) as source:
        oriented = ImageOps.exif_transpose(source)
        if "A" in oriented.getbands() or "transparency" in oriented.info:
            rgba = oriented.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (*alpha_background, 255))
            return Image.alpha_composite(background, rgba).convert("RGB")
        return oriented.convert("RGB")


def pad_to_square(image, background):
    width, height = image.size
    side = max(width, height)
    square = Image.new("RGB", (side, side), background)
    square.paste(image, ((side - width) // 2, (side - height) // 2))
    return square


def geometry(image, strategy="pad", size=336, background=(122, 116, 104)):
    if strategy == "pad":
        prepared = pad_to_square(image, background)
    elif strategy == "center_crop":
        prepared = image
    elif strategy == "stretch":
        return image.resize((size, size), Image.Resampling.BICUBIC)
    else:
        raise ValueError(strategy)
    width, height = prepared.size
    scale = size / min(width, height)
    # PIL stores size as (width, height), rather than NumPy's (height, width).
    resized = prepared.resize((int(width * scale), int(height * scale)),
                              Image.Resampling.BICUBIC)
    left = (resized.width - size) // 2
    top = (resized.height - size) // 2
    return resized.crop((left, top, left + size, top + size))


def normalize_rgb(image, processor):
    uint8_hwc = np.asarray(image, dtype=np.uint8)
    rgb01 = uint8_hwc.astype(np.float32) / 255.0
    mean = np.asarray(processor["image_mean"], dtype=np.float32)
    std = np.asarray(processor["image_std"], dtype=np.float32)
    normalized_hwc = (rgb01 - mean) / std
    chw = np.ascontiguousarray(normalized_hwc.transpose(2, 0, 1))
    return torch.from_numpy(chw)


def preprocess(image, strategy="pad"):
    _, _, processor = load_configs()
    background = tuple(int(x * 255) for x in processor["image_mean"])
    processed = geometry(image, strategy, size=processor["size"], background=background)
    return processed, normalize_rgb(processed, processor)


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow 10.0/10.1 compatibility
        return ImageFont.load_default()


def make_cases():
    OUT.mkdir(parents=True, exist_ok=True)
    wide = Image.new("RGB", (600, 300), "#edf3f8")
    d = ImageDraw.Draw(wide)
    d.rectangle((20, 70, 130, 180), fill=(235, 65, 65))
    d.ellipse((465, 70, 575, 180), fill=(45, 125, 220))
    d.text((35, 207), "LEFT", font=_font(35), fill="#17334a")
    d.text((465, 207), "RIGHT", font=_font(35), fill="#17334a")
    d.rectangle((220, 110, 380, 158), fill="#e8c173")
    d.text((237, 119), "CENTER", font=_font(24), fill="#17334a")
    wide.save(OUT / "wide.png")
    document = Image.new("RGB", (300, 600), "white")
    d = ImageDraw.Draw(document)
    d.rectangle((0, 0, 299, 105), fill="#deeee7")
    d.text((20, 25), "YEAR: 1280", font=_font(35), fill="#17334a")
    for i in range(7):
        d.rectangle((25, 155 + i * 43, 275, 165 + i * 43), fill="#95a7b5")
    d.rectangle((0, 490, 299, 599), fill="#ffe0ba")
    d.text((25, 518), "ID: 0088", font=_font(37), fill="#17334a")
    document.save(OUT / "document.png")
    gray = Image.new("L", (48, 48), 128)
    gray.save(OUT / "grayscale.png")
    rgba = Image.new("RGBA", (48, 48), (255, 0, 0, 0))
    ImageDraw.Draw(rgba).rectangle((14, 14, 34, 34), fill=(255, 0, 0, 128))
    rgba.save(OUT / "transparent.png")
    camera = Image.new("RGB", (160, 80), "red")
    ImageDraw.Draw(camera).rectangle((80, 0, 159, 79), fill="blue")
    exif = Image.Exif()
    exif[274] = 6  # Orientation: rotate 90 degrees clockwise for display.
    camera.save(OUT / "orientation.jpg", exif=exif, quality=95)
    return {"wide": OUT / "wide.png", "document": OUT / "document.png",
            "gray": OUT / "grayscale.png", "rgba": OUT / "transparent.png",
            "exif": OUT / "orientation.jpg"}


def _text(x, y, value, size=21, fill="#17334a"):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}">{escape(value)}</text>'


def make_comparison(name, title, image):
    # These are actual PIL results; embed PNG bytes so the SVG is standalone.
    _, _, cfg = load_configs()
    background = tuple(int(x * 255) for x in cfg["image_mean"])
    pictures = [image, geometry(image, "center_crop", background=background),
                geometry(image, "pad", background=background)]
    captions = ["原始图像", "短边缩放＋中心裁剪", "补边成方图＋缩放"]
    notes = (["左右两端都含信息", "左右边缘区域被裁掉", "保留两端，但有效内容变小"] if name == "wide" else
             ["标题与编号在上下边缘", "年份和编号可能被裁掉", "保留完整版面，字也变小"])
    s = '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="505" viewBox="0 0 1000 505"><title>'+escape(title)+'</title><rect width="1000" height="505" rx="18" fill="#f5f9fc"/><g font-family="PingFang SC,Microsoft YaHei,Arial,sans-serif">'
    s += _text(30, 43, title, 27)
    for i, picture in enumerate(pictures):
        x = 30 + i * 328
        path = OUT / f"{name}_{i}.png"
        picture.save(path)
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        s += _text(x, 87, captions[i], 22)
        s += f'<rect x="{x}" y="105" width="300" height="300" rx="9" fill="#fff" stroke="#a6b9c7"/>'
        s += f'<image x="{x}" y="105" width="300" height="300" preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,{data}"/>'
        s += _text(x, 439, notes[i], 19)
    s += _text(30, 480, "教学像素的真实处理结果；所选 LLaVA-1.5 7B 配置使用 pad，其他模型须查自己的处理器。", 17)
    s += '</g></svg>'
    (OUT.parent / f"preprocess-{name}.svg").write_text(s, encoding="utf-8")


def prepare_assets():
    cases = make_cases()
    make_comparison("wide", "Case A · 横图：中心裁剪可能删掉左右物体", load_rgb(cases["wide"]))
    make_comparison("document", "Case B · 长文档：版面保住了，细字仍可能变小", load_rgb(cases["document"]))
    return cases


def shape_walkthrough():
    """Execute dimension-changing operations on meta tensors, with no weights loaded."""
    llava, vision, _ = load_configs()
    width, size, patch = vision["hidden_size"], vision["image_size"], vision["patch_size"]
    hidden, vocab = llava["hidden_size"], llava["vocab_size"]
    pixels = torch.empty(1, 3, size, size, device="meta")
    patch_layer = nn.Conv2d(3, width, kernel_size=patch, stride=patch, bias=False, device="meta")
    grid = patch_layer(pixels)
    patches = grid.flatten(2).transpose(1, 2)
    cls = torch.empty(1, 1, width, device="meta")
    with_cls = torch.cat([cls, patches], dim=1)
    # A full vision Transformer is not executed; this is the output contract.
    selected_hidden = torch.empty_like(with_cls)
    visual_features = selected_hidden[:, 1:]
    projector = nn.Sequential(nn.Linear(width, hidden, device="meta"), nn.GELU(),
                              nn.Linear(hidden, hidden, device="meta"))
    projected = projector(visual_features)
    # Illustrative valid word IDs, not a measured tokenization of a sentence.
    input_ids = torch.arange(1, 22).unsqueeze(0)
    input_ids[0, 3] = -200
    embed = nn.Embedding(vocab, hidden, device="meta")
    prefix = embed(input_ids[:, :3])
    suffix = embed(input_ids[:, 4:])
    sequence = torch.cat([prefix, projected, suffix], dim=1)
    final_hidden = torch.empty_like(sequence)  # 32 LLM layers preserve this shape.
    output_head = nn.Linear(hidden, vocab, bias=False, device="meta")
    logits = output_head(final_hidden)
    last_logits = output_head(final_hidden[:, -1:])
    values = {"pixels": pixels, "patch_grid": grid, "patch_sequence": patches,
              "with_cls": with_cls, "selected_layer": selected_hidden,
              "drop_cls": visual_features, "projected": projected,
              "raw_input_ids": input_ids, "text_prefix": prefix, "text_suffix": suffix,
              "combined": sequence, "lm_hidden": final_hidden,
              "logits": logits, "generation_last_logits": last_logits}
    shapes = {k: list(v.shape) for k, v in values.items()}
    assert shapes["combined"] == [1, 596, 4096]
    assert shapes["logits"] == [1, 596, 32000]
    return shapes


def run_checks():
    cases = prepare_assets()
    _, _, cfg = load_configs()
    report = {"scope": "synthetic image preprocessing; no VLM weights/inference",
              "cases": {}, "shapes": shape_walkthrough()}
    for name in ("wide", "document"):
        image = load_rgb(cases[name])
        processed, tensor = preprocess(image)
        assert tensor.shape == (3, 336, 336) and tensor.dtype == torch.float32
        restored = tensor.numpy().transpose(1, 2, 0) * np.array(cfg["image_std"]) + np.array(cfg["image_mean"])
        reconstruction_error = float(np.max(np.abs(restored - np.asarray(processed) / 255.0)))
        assert reconstruction_error < 1e-6
        report["cases"][name] = {"original_wh": list(image.size), "processed_wh": list(processed.size),
                                  "chw": list(tensor.shape), "reconstruction_error": reconstruction_error}
    cropped = np.asarray(geometry(load_rgb(cases["wide"]), "center_crop"))
    padded, _ = preprocess(load_rgb(cases["wide"]))
    padded = np.asarray(padded)
    def red_count(array):
        return int(((array[..., 0] > 200) & (array[..., 1] < 100) & (array[..., 2] < 100)).sum())
    assert red_count(cropped) == 0 and red_count(padded) > 0
    report["crop_check"] = {"red_pixels_after_center_crop": red_count(cropped),
                             "red_pixels_after_pad": red_count(padded)}
    gray = np.asarray(load_rgb(cases["gray"]))
    rgba = np.asarray(load_rgb(cases["rgba"]))
    assert gray.shape == (48, 48, 3) and gray[0, 0].tolist() == [128, 128, 128]
    assert rgba[0, 0].tolist() == [255, 255, 255]
    assert load_rgb(cases["exif"]).size == (80, 160)
    report["channels"] = {"gray_rgb": gray[0, 0].tolist(), "transparent_over_white": rgba[0, 0].tolist(),
                          "exif_corrected_wh": [80, 160]}
    (Path(__file__).parent / "lecture_04_preprocess_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def verify_hf_processor():
    """Optional reference check; Transformers is not needed by the course compiler."""
    import transformers
    import PIL
    from transformers import CLIPImageProcessor
    _, _, config = load_configs()
    reference = CLIPImageProcessor.from_dict(config)
    cases = prepare_assets()
    errors = {}
    for name in ("wide", "document"):
        image = load_rgb(cases[name])
        _, actual = preprocess(image)
        square = pad_to_square(image, tuple(int(x * 255) for x in config["image_mean"]))
        expected = reference(images=square, return_tensors="pt")["pixel_values"][0]
        errors[name] = float((actual - expected).abs().max())
        assert errors[name] < 1e-5, errors
    report = {"reference_class": reference.__class__.__name__,
              "transformers_version": transformers.__version__, "pillow_version": PIL.__version__,
              "max_absolute_error": errors,
              "scope": "synthetic RGB inputs; no pretrained model weights"}
    (CONFIG_DIR / "processor_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-hf", action="store_true", help="Optional: compare against installed Transformers processor")
    args = parser.parse_args()
    result = run_checks()
    if args.verify_hf:
        result["hf_processor_comparison"] = verify_hf_processor()
    print(json.dumps(result, ensure_ascii=False, indent=2))
