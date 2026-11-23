"""CPU examples for video sampling, temporal order, token shapes and grounding."""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def load_clip():
    with np.load(ROOT / 'images/lecture_09/teaching_clip.npz') as data:
        frames = torch.from_numpy(data['frames'].copy()).permute(0, 3, 1, 2)
        timestamps = torch.from_numpy(data['timestamps'].copy())
    metadata = json.loads((ROOT / 'demos/lecture_09_assets.json').read_text())
    assert frames.shape == (96, 3, 90, 160)
    return frames, timestamps, metadata


def uniform_indices(total, count):
    """Pick the center frame of each of count equal temporal bins."""
    if not 1 <= count <= total:
        raise ValueError('count must be between 1 and total')
    return torch.floor((torch.arange(count) + .5) * total / count).long()


def fps_indices(timestamps, target_fps, duration):
    """Use presentation timestamps, rather than assuming arbitrary video is CFR."""
    if target_fps <= 0:
        raise ValueError('target_fps must be positive')
    targets = torch.arange(0., duration, 1 / target_fps, dtype=timestamps.dtype)
    indices = torch.searchsorted(timestamps, targets).clamp_max(len(timestamps) - 1)
    return torch.unique_consecutive(indices)


def reorder_frames(frames, seed=9):
    order = torch.randperm(len(frames), generator=torch.Generator().manual_seed(seed))
    shuffled = frames[order]
    restored = shuffled[torch.argsort(order)]
    assert torch.equal(restored, frames)
    return shuffled, order, restored


def pixel_motion(frames):
    return (frames[1:].float() - frames[:-1].float()).abs().mean((1, 2, 3)) / 255


def temporal_pooling_example(frames):
    # Three observed RGB frames; the demonstration does not infer event labels.
    x = frames[torch.tensor([8, 40, 64])].float() / 255
    reversed_x = x.flip(0)
    means = torch.stack([x.mean(0), reversed_x.mean(0)])
    deltas = torch.stack([x[-1] - x[0], reversed_x[-1] - reversed_x[0]])
    assert torch.allclose(means[0], means[1], atol=1e-6)
    assert torch.allclose(deltas[0], -deltas[1])
    assert deltas[0].abs().sum() > 0
    return means, deltas


def tubelet_example(frames):
    resized = F.interpolate(frames[:8].float() / 255, size=(32, 32), mode='bilinear', align_corners=False)
    # [T,C,H,W] -> [T/2,C,H/8,W/8,2,8,8] -> [64,384].
    blocks = resized.unfold(0, 2, 2).unfold(2, 8, 8).unfold(3, 8, 8)
    patches = blocks.permute(0, 2, 3, 1, 4, 5, 6).reshape(64, 384)
    expected_first = resized[:2, :, :8, :8].permute(1, 0, 2, 3).reshape(-1)
    assert torch.allclose(patches[0], expected_first)
    return resized, patches


def stc_shape_example():
    with torch.random.fork_rng():
        torch.manual_seed(9)
        features = torch.randn(1, 8, 8, 24, 24)
        conv = nn.Conv3d(8, 16, kernel_size=2, stride=2)
        reduced = conv(features)
        tokens = reduced.flatten(2).transpose(1, 2)
    assert reduced.shape == (1, 16, 4, 12, 12)
    assert tokens.shape == (1, 576, 16)
    return features, reduced, tokens


def qwen_shape_example():
    # Qwen2.5-VL-7B config dimensions. No weights or pretrained inference.
    pixels = torch.empty(1, 8, 3, 224, 224, device='meta')
    tubelets = torch.empty(1024, 3 * 2 * 14 * 14, device='meta')
    vision_hidden = torch.empty(4, 16, 16, 1280, device='meta')
    groups = vision_hidden.reshape(4, 8, 2, 8, 2, 1280)
    groups = groups.permute(0, 1, 3, 2, 4, 5).reshape(256, 5120)
    projected = nn.Linear(5120, 3584, device='meta')(groups)
    assert projected.shape == (256, 3584)
    return {'pixels': tuple(pixels.shape), 'tubelets': tuple(tubelets.shape),
            'vision_hidden': tuple(vision_hidden.shape), 'merged': tuple(groups.shape),
            'llm_visual_tokens': tuple(projected.shape)}


def answer_loss_example():
    # Toy vocabulary; the three answer IDs stand for 倒水、之后、喝水.
    sequence = torch.tensor([1, 2, 3, 4, 5, 6])
    labels = sequence[1:].clone()
    labels[:2] = -100
    generator = torch.Generator().manual_seed(9)
    logits = torch.randn(5, 8, generator=generator, requires_grad=True)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    assert logits.grad[:2].abs().sum() == 0
    assert logits.grad[2:].abs().sum() > 0
    return labels, float(loss.detach()), logits.grad.norm(dim=1)


def interval_iou(prediction, reference):
    if prediction[1] <= prediction[0] or reference[1] <= reference[0]:
        raise ValueError('intervals must have positive duration')
    intersection = max(0., min(prediction[1], reference[1]) - max(prediction[0], reference[0]))
    union = (prediction[1] - prediction[0]) + (reference[1] - reference[0]) - intersection
    return intersection / union


def aligned_time_example():
    # Teaching grids, not claims about MOVA VAE rates.
    video_time = torch.arange(5) / 5
    audio_time = torch.arange(50) / 50
    target = video_time[2]
    nearest = (audio_time - target).abs().argmin()
    assert torch.allclose(video_time[2], audio_time[nearest])
    return video_time, audio_time, int(nearest)
