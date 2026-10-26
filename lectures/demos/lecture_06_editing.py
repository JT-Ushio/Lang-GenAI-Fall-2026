"""Offline numerical experiments for language-guided image editing.

The RGB scenes/composites are explicit array operations. Only the 8x8 bar
experiment uses a trained denoiser, reusing lecture 05's six-template MLP.
Meta-tensor contracts and zero-conv mechanics are not pretrained SD inference.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageFilter
from demos.lecture_05_diffusion import make_schedule, ddpm_step, train_demo

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).parent / "lecture_06_configs"


def scene_and_masks():
    source = torch.empty(3, 64, 64)
    source[:] = torch.tensor([0.94, 0.96, 0.98])[:, None, None]
    source[:, 16:40, 8:32] = torch.tensor([0.88, 0.22, 0.20])[:, None, None]
    yy, xx = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    source[:, (xx - 48).square() + (yy - 28).square() <= 10**2] = torch.tensor([0.17, 0.42, 0.86])[:, None]
    mask = torch.zeros(1, 64, 64)
    mask[:, 16:40, 8:32] = 1
    target = source.clone()
    target[:, 16:40, 8:32] = torch.tensor([0.16, 0.73, 0.30])[:, None, None]
    candidate = source * 0.84  # Deliberate background/blue-circle drift.
    candidate[:, 16:40, 8:32] = target[:, 16:40, 8:32]
    blur = Image.fromarray((mask[0].numpy() * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))
    soft_mask = torch.from_numpy(np.asarray(blur).copy()).float().unsqueeze(0) / 255
    return source, mask, soft_mask, candidate, target


def composite(source, candidate, mask):
    """Here mask=1 means edit, mask=0 means keep; CHW with broadcast mask."""
    return mask * candidate + (1 - mask) * source


def region_mae(output, reference, region):
    """Per-scalar MAE, normalized by region area AND number of channels."""
    expanded = region.expand_as(output)
    if expanded.sum() == 0:
        return 0.0
    return float(((output - reference).abs() * expanded).sum() / expanded.sum())


def sobel_edges(source, threshold=0.25):
    gray = (source * source.new_tensor([0.299, 0.587, 0.114])[:, None, None]).sum(0)
    kx = source.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernels = torch.stack([kx, kx.T]).unsqueeze(1)
    gradients = F.conv2d(F.pad(gray[None, None], (1, 1, 1, 1), mode="replicate"), kernels)
    magnitude = gradients.square().sum(1, keepdim=True).sqrt()
    return (magnitude > threshold).float()[0]


@torch.no_grad()
def edit_bars(model, source, label=1, strength=0.6, seed=42, mask=None):
    """Actual DDPM from a noised source, optionally projecting known regions.

    Educational strength mapping: round(strength*T) updates. Real pipelines
    use their inference grid and scheduler order instead of this exact rule.
    """
    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0,1]")
    model.eval()
    beta, alpha, alpha_bar = make_schedule()
    updates = min(len(beta), max(1, round(strength * len(beta)))) if strength else 0
    clean = source.reshape(1, 64)
    if updates == 0:
        return clean.reshape(8, 8).clone(), clean.reshape(8, 8).clone(), []
    rng = torch.Generator().manual_seed(seed)
    original_epsilon = torch.randn(clean.shape, generator=rng)
    a = alpha_bar[updates - 1]
    noisy = a.sqrt() * clean + (1 - a).sqrt() * original_epsilon
    initial = noisy.reshape(8, 8).clone()
    paths = []
    for t in reversed(range(updates)):
        times = torch.tensor([t])
        eps_u = model(noisy, times, torch.tensor([2]))
        eps_c = model(noisy, times, torch.tensor([label]))
        prediction = eps_u + 2.0 * (eps_c - eps_u)
        random_noise = torch.randn(noisy.shape, generator=rng)
        proposal = ddpm_step(noisy, prediction, t, beta, alpha, alpha_bar, random_noise)
        if mask is not None:
            # The proposal is at t-1, so compare to source at t-1, not clean source.
            previous = alpha_bar[t - 1] if t > 0 else alpha_bar.new_tensor(1.)
            reference = previous.sqrt() * clean + (1 - previous).sqrt() * original_epsilon
            m = mask.reshape(1, 64)
            proposal = m * proposal + (1 - m) * reference
        noisy = proposal
        if t in (updates - 1, 60, 30, 10, 0):
            paths.append({"updates": updates - t, "image": noisy.reshape(8, 8).tolist()})
    return noisy.reshape(8, 8), initial, paths


def zero_conv_experiment():
    """A small residual-branch analogue, not a complete ControlNet."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(6)
        base = nn.Conv2d(4, 8, 3, padding=1)
        branch = nn.Conv2d(3, 8, 3, padding=1)
        zero = nn.Conv2d(8, 8, 1)
        nn.init.zeros_(zero.weight)
        nn.init.zeros_(zero.bias)
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.SGD([*branch.parameters(), *zero.parameters()], lr=0.2)
        noisy, hint = torch.randn(1, 4, 8, 8), torch.randn(1, 3, 8, 8)
        original = base(noisy)
        combined = original + zero(branch(hint))
        initial_delta = float((combined - original).abs().max().detach())
        report = {"initial_output_delta": initial_delta}
        for step in (1, 2):
            optimizer.zero_grad()
            combined = base(noisy) + zero(branch(hint))
            loss = (combined - 1).square().mean()
            loss.backward()
            report[f"zero_grad_step_{step}"] = float(zero.weight.grad.abs().sum())
            report[f"branch_grad_step_{step}"] = float(branch.weight.grad.abs().sum())
            optimizer.step()
        report["frozen_base_has_gradient"] = any(p.grad is not None for p in base.parameters())
        assert initial_delta == 0
        assert report["zero_grad_step_1"] > 0
        assert report["branch_grad_step_1"] == 0
        assert report["branch_grad_step_2"] > 0
        assert not report["frozen_base_has_gradient"]
        return report


def shape_walkthrough(batch=1, cfg=True):
    """Meta interfaces of selected SD1.5-compatible model variants."""
    configs = {name: json.loads((CONFIG / file).read_text()) for name, file in
               {"inpaint": "inpaint_unet.json", "control": "controlnet.json", "instruct": "instruct_unet.json"}.items()}
    z = torch.empty(batch, 4, 64, 64, device="meta")
    mask_pixels = torch.empty(batch, 1, 512, 512, device="meta")
    mask_latent = F.interpolate(mask_pixels, size=(64, 64), mode="nearest")
    masked_encoding = torch.empty_like(z)  # Encode the masked image; VAE not executed.
    inpaint_input = torch.cat([z, mask_latent, masked_encoding], dim=1)
    actual_input = torch.cat([inpaint_input, inpaint_input]) if cfg else inpaint_input
    first = nn.Conv2d(configs["inpaint"]["in_channels"], 320, 3, padding=1, device="meta")(actual_input)
    prediction = torch.empty(actual_input.shape[0], 4, 64, 64, device="meta")
    instruction_input = torch.cat([z, torch.empty_like(z)], dim=1)
    three_branches = torch.cat([instruction_input] * 3) if cfg else instruction_input
    # Older serialized configs omit the standard 3-channel conditioning default.
    hint = torch.empty(batch, configs["control"].get("conditioning_channels", 3), 512, 512, device="meta")
    channels = configs["control"]["conditioning_embedding_out_channels"]
    control = nn.Conv2d(hint.shape[1], channels[0], 3, padding=1, device="meta")(hint)
    for channel in channels[1:]:
        control = nn.Conv2d(control.shape[1], channel, 3, stride=2, padding=1, device="meta")(control)
    control = nn.Conv2d(control.shape[1], 320, 3, padding=1, device="meta")(control)
    tensors = {"noisy_latent": z, "pixel_mask": mask_pixels, "latent_mask": mask_latent,
               "masked_image_latent": masked_encoding, "inpaint_input": inpaint_input,
               "inpaint_cfg_input": actual_input, "first_feature": first, "noise_prediction": prediction,
               "text_condition": torch.empty(actual_input.shape[0], 77, 768, device="meta"),
               "instruction_input": instruction_input, "instruction_three_branches": three_branches,
               "control_image": hint, "control_hint_feature": control}
    assert inpaint_input.shape[1] == configs["inpaint"]["in_channels"] == 9
    assert instruction_input.shape[1] == configs["instruct"]["in_channels"] == 8
    assert control.shape == (batch, 320, 64, 64)
    return {name: list(t.shape) for name, t in tensors.items()}


def combine_predictions(unconditional, image_only, image_and_text, image_scale, text_scale):
    return (unconditional + image_scale * (image_only - unconditional)
            + text_scale * (image_and_text - image_only))


def build_experiments():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        source, mask, soft, candidate, target = scene_and_masks()
        results = {"unchanged": source, "global": candidate,
                   "local": composite(source, candidate, mask), "soft": composite(source, candidate, soft)}
        metrics = {name: {"outside_mae": region_mae(im, source, 1-mask),
                          "inside_target_mae": region_mae(im, target, mask)} for name, im in results.items()}
        assert torch.equal(results["local"], target)
        assert metrics["unchanged"]["outside_mae"] == 0
        assert metrics["unchanged"]["inside_target_mae"] > 0
        assert metrics["global"]["inside_target_mae"] == 0
        assert metrics["global"]["outside_mae"] > 0
        assert metrics["soft"]["outside_mae"] > 0
        model, training, _ = train_demo()
        bar = -torch.ones(8, 8)
        bar[:, 3:5] = 1
        cases = []
        for seed in (7, 42):
            for label in (0, 1):
                for strength in (0., .1, .3, .6, .9, 1.):
                    output, noisy, _ = edit_bars(model, bar, label, strength, seed)
                    cases.append({"seed": seed, "label": label, "strength": strength,
                                  "updates": round(strength * 100), "noisy": noisy.tolist(),
                                  "result": output.tolist(), "change_mae": float((output-bar).abs().mean())})
        edit_mask = torch.zeros(8, 8)
        edit_mask[2:6] = 1
        local, _, trajectory = edit_bars(model, bar, 1, .9, 42, edit_mask)
        outside = (edit_mask == 0)
        assert torch.equal(local[outside], bar[outside])
        all_known, _, _ = edit_bars(model, bar, 1, .9, 42, torch.zeros(8, 8))
        assert torch.equal(all_known, bar)
        unrestricted, _, _ = edit_bars(model, bar, 1, .9, 42)
        all_editable, _, _ = edit_bars(model, bar, 1, .9, 42, torch.ones(8, 8))
        assert torch.equal(unrestricted, all_editable)
        for label in (0, 1):
            assert cases[12 + label*6]["result"] == bar.tolist()
        report = {"training_parameters": training["parameters"], "training_updates": training["updates"],
                  "noise_mse_before": training["history"][0]["noise_mse"], "noise_mse_after": training["history"][-1]["noise_mse"],
                  "rgb_metrics": metrics, "toy_inpaint_outside_max_error": float((local-bar).abs()[outside].max()),
                  "all_known_mask_returns_source": True, "all_editable_mask_matches_unrestricted": True,
                  "zero_conv": zero_conv_experiment(),
                  "scope": "RGB edits are array operations. Learned edits use the six-template lecture05 MLP; no natural-language encoder, pretrained SD inference or held-out clean-image generalization."}
        return {"report": report, "source": source.tolist(), "mask": mask.tolist(), "soft_mask": soft.tolist(),
                "candidate": candidate.tolist(), "target": target.tolist(), "rgb_results": {k:v.tolist() for k,v in results.items()},
                "edges": sobel_edges(source).tolist(), "bar_source": bar.tolist(), "bar_cases": cases,
                "bar_mask": edit_mask.tolist(), "bar_local": local.tolist(), "bar_unrestricted": unrestricted.tolist(),
                "bar_trajectory": trajectory, "alpha_bar": make_schedule()[2].tolist(), "shapes": shape_walkthrough()}
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    print(json.dumps(build_experiments()["report"], ensure_ascii=False, indent=2))
