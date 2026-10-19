"""Local, reproducible figures made from lecture 05's actual numerical runs."""
import base64
import io
import json
from pathlib import Path
from html import escape
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images/lecture_05"


def make_rgb_scene():
    """Original 32x32 scene, CHW float32 in [-1, 1]."""
    rgb = np.full((32, 32, 3), 235, dtype=np.uint8)
    rgb[10:23, 2:15] = [225, 65, 60]
    yy, xx = np.mgrid[:32, :32]
    rgb[(xx - 23) ** 2 + (yy - 16) ** 2 <= 6 ** 2] = [50, 110, 220]
    return rgb.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1


def pixels(array):
    x = np.asarray(array)
    if x.ndim == 3 and x.shape[0] == 3:
        x = x.transpose(1, 2, 0)
    return Image.fromarray(np.round((np.clip(x, -1, 1) + 1) * 127.5).astype(np.uint8))


def embedded(array, x, y, size):
    buffer = io.BytesIO()
    pixels(array).resize((size, size), Image.Resampling.NEAREST).save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode()
    return f'<image x="{x}" y="{y}" width="{size}" height="{size}" href="data:image/png;base64,{data}"/>'


def label(x, y, text, size=21, color="#17334a"):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{escape(str(text))}</text>'


def save(name, title, body, foot, height):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="{height}" viewBox="0 0 1000 {height}" role="img"><title>{escape(title)}</title>'
           f'<rect width="1000" height="{height}" rx="18" fill="#f5f9fc"/><g font-family="PingFang SC,Microsoft YaHei,Arial,sans-serif">' +
           label(30, 44, title, 28) + body + label(30, height - 24, foot, 17, "#536c80") + '</g></svg>')
    (OUT / name).write_text(svg, encoding="utf-8")


def prepare_forward(clean, epsilon, alpha_bar):
    OUT.mkdir(parents=True, exist_ok=True)
    original = np.asarray(clean)
    noise = np.asarray(epsilon)
    alpha = np.asarray(alpha_bar)
    panels = [("干净原图", original)]
    for t in (49, 199, 499, 999):
        panels.append((f"数组索引 t={t}", np.sqrt(alpha[t]) * original + np.sqrt(1 - alpha[t]) * noise))
    body = ''
    for i, (name, values) in enumerate(panels):
        x = 30 + 196 * i
        body += embedded(values, x, 85, 158) + label(x, 276, name, 19)
        if i:
            body += label(x, 311, f"保留系数 {np.sqrt(alpha[(49,199,499,999)[i-1]]):.3f}", 18)
    save("forward-panels.svg", "实际像素加噪：信号减少，随机纹理增加", body,
         "各面板复用同一份 ε 作控制对照；不是一条逐步采样的马尔可夫轨迹。显示时裁剪到 [-1,1]。", 375)


def prepare_assets(clean, epsilon, alpha_bar, report, data, shapes):
    prepare_forward(clean, epsilon, alpha_bar)
    original, noise, alpha = np.asarray(clean), np.asarray(epsilon), np.asarray(alpha_bar)
    history = report["history"]
    body = label(30, 92, "固定评估噪声与时间索引；干净模板仍来自六张训练模板", 21)
    body += '<path d="M90 130V430H930" stroke="#536c80" stroke-width="2" fill="none"/>'
    for tick in (0, .25, .5, .75, 1):
        y = 430 - 280 * tick
        body += f'<path d="M90 {y}H930" stroke="#dbe5ed"/>' + label(36, y + 7, f"{tick:g}", 18)
    points = [(90 + h["updates"] / report["updates"] * 820, 430 - h["noise_mse"] * 280) for h in history]
    body += '<polyline points="' + ' '.join(f'{x},{y}' for x, y in points) + '" stroke="#248887" stroke-width="4" fill="none"/>'
    for (x, y), h in zip(points, history):
        body += f'<circle cx="{x}" cy="{y}" r="5" fill="#248887"/>'
        if h["updates"] in (0, 500, report["updates"]):
            body += label(x + (10 if x < 850 else -68), y - 13, f'{h["noise_mse"]:.3f}', 19)
    for t in (0, 1000, 2000, 3000, 4000):
        body += label(78 + t / report["updates"] * 820, 462, t, 18)
    body += label(385, 500, "参数更新次数（横轴）", 20) + label(30, 540, "纵轴：噪声预测均方误差 MSE；这里没有独立留出的干净图像测试集。", 21)
    save("toy-training.svg", "真实训练记录：微型 MLP 学习预测噪声", body,
         f'{report["parameters"]:,} 个参数；CPU；{report["updates"]:,} 次更新；不下载 Stable Diffusion 权重。', 600)

    body = ''
    for i, values in enumerate(data["samples"]):
        row, col = divmod(i, 8)
        body += embedded(values, 30 + col * 120, 115 + row * 172, 104)
    body += label(30, 91, "条件 0：竖条（从新噪声生成的前 8 张）", 21)
    body += label(30, 263, "条件 1：横条（从新噪声生成的前 8 张）", 21)
    body += label(30, 465, f'64 张新噪声样本的方向匹配：{report["new_noise_orientation_accuracy"]:.1%}（受限规则指标）', 22)
    save("toy-generated.svg", "实际生成样本：同一模型根据类别条件画横条或竖条", body,
         "任务仅有六张 8×8 模板；没有验证自然图像质量、未见场景泛化或自然语言理解。", 530)

    body = ''
    for row, name in enumerate(("vertical", "horizontal")):
        trajectory = data[name]
        chosen = [trajectory[i] for i in (0, 2, 6, 12, 18, len(trajectory) - 1)]
        body += label(30, 90 + row * 210, "竖条条件" if row == 0 else "横条条件", 21)
        for col, state in enumerate(chosen):
            x, y = 30 + col * 163, 110 + row * 210
            body += embedded(state["image"], x, y, 132)
            body += label(x, y + 159, f'已更新 {state["updates"]} 次', 18)
    save("toy-sampling.svg", "实际 DDPM 采样：相同初始噪声，不同类别条件", body,
         "图像来自训练后网络的采样记录；不是将加噪过程倒放。两行还复用同一随机数序列。", 555)

    lab = OUT / "diffusion-lab"
    lab.mkdir(exist_ok=True)
    payload = {"clean": original.tolist(), "epsilon": noise.tolist(), "alpha_bar": alpha.tolist(),
               "training": report, "sampling": data, "shapes": shapes}
    (lab / "data.js").write_text("window.DIFFUSION_DATA=" + json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + ";\n", encoding="utf-8")
    (ROOT / "demos/lecture_05_diffusion_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
