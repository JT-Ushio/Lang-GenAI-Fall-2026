"""Small, offline image diffusion experiment for lecture 05.

Six 8x8 grayscale bar templates; two class labels plus a null condition.
This is a noise-predicting MLP trained with a DDPM objective, not Stable
Diffusion or a natural-language text encoder. Evaluation noise is new, but
the six clean templates all participate in training.
"""
import json
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).parent / "lecture_05_configs"


def make_schedule(steps=100, beta_start=0.0001, beta_end=0.15):
    beta = torch.linspace(beta_start, beta_end, steps)
    alpha = 1.0 - beta
    alpha_bar = alpha.cumprod(0)
    return beta, alpha, alpha_bar


def make_templates():
    images, labels = [], []
    for label in (0, 1):  # 0: vertical; 1: horizontal
        for offset in (2, 3, 4):
            image = -torch.ones(8, 8)
            if label == 0:
                image[:, offset:offset + 2] = 1.
            else:
                image[offset:offset + 2, :] = 1.
            images.append(image.flatten())
            labels.append(label)
    return torch.stack(images), torch.tensor(labels)


def add_noise(clean, noise, timesteps, alpha_bar):
    a = alpha_bar[timesteps].view(-1, *([1] * (clean.ndim - 1)))
    return a.sqrt() * clean + (1.0 - a).sqrt() * noise


class NoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition = nn.Embedding(3, 16)  # 2 is the null condition.
        self.network = nn.Sequential(nn.Linear(64 + 32 + 16, 128), nn.SiLU(),
                                     nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 64))

    def forward(self, noisy, timesteps, labels):
        frequencies = torch.exp(-math.log(10000) * torch.arange(16, device=noisy.device) / 16)
        phase = timesteps.float()[:, None] * frequencies[None]
        time_embedding = torch.cat([phase.sin(), phase.cos()], dim=-1)
        condition = self.condition(labels)
        return self.network(torch.cat([noisy, time_embedding, condition], dim=-1))


def training_step(model, optimizer, clean, labels, alpha_bar):
    t = torch.randint(len(alpha_bar), (len(clean),))
    epsilon = torch.randn_like(clean)
    noisy = add_noise(clean, epsilon, t, alpha_bar)
    # Learning the null branch makes classifier-free guidance possible later.
    condition = labels.clone()
    condition[torch.rand(len(labels)) < 0.15] = 2
    predicted_noise = model(noisy, t, condition)
    loss = F.mse_loss(predicted_noise, epsilon)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def ddpm_step(noisy, predicted_noise, t, beta, alpha, alpha_bar, random_noise):
    a = alpha_bar[t]
    previous = alpha_bar[t - 1] if t > 0 else a.new_tensor(1.)
    predicted_clean = ((noisy - (1 - a).sqrt() * predicted_noise) / a.sqrt()).clamp(-1, 1)
    coefficient_clean = beta[t] * previous.sqrt() / (1 - a)
    coefficient_noisy = (1 - previous) * alpha[t].sqrt() / (1 - a)
    mean = coefficient_clean * predicted_clean + coefficient_noisy * noisy
    variance = beta[t] * (1 - previous) / (1 - a)
    return mean + variance.clamp_min(0).sqrt() * random_noise if t > 0 else mean


@torch.no_grad()
def sample(model, labels, seed=30, guidance=2.0, record=False):
    model.eval()
    beta, alpha, alpha_bar = make_schedule()
    rng = torch.Generator().manual_seed(seed)
    noisy = torch.randn(len(labels), 64, generator=rng)
    trajectory = [{"updates": 0, "next_t": len(beta) - 1,
                   "image": noisy[0].reshape(8, 8).tolist()}]
    for t in reversed(range(len(beta))):
        times = torch.full((len(labels),), t, dtype=torch.long)
        unconditional = model(noisy, times, torch.full_like(labels, 2))
        conditional = model(noisy, times, labels)
        predicted = unconditional + guidance * (conditional - unconditional)
        random_noise = torch.randn(noisy.shape, generator=rng)
        noisy = ddpm_step(noisy, predicted, t, beta, alpha, alpha_bar, random_noise)
        if record and (t % 5 == 0 or t == len(beta) - 1):
            trajectory.append({"updates": len(beta) - t, "next_t": t - 1,
                               "image": noisy[0].reshape(8, 8).tolist()})
    return noisy.reshape(-1, 8, 8), trajectory


def orientation_predictions(images):
    # A restricted teaching metric: vertical bars vary more across columns.
    columns = images.mean(dim=1).var(dim=1)
    rows = images.mean(dim=2).var(dim=1)
    return (rows > columns).long()


def train_demo(updates=4000):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(15)
            clean, labels = make_templates()
            _, _, alpha_bar = make_schedule()
            model = NoisePredictor()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)
            rng = torch.Generator().manual_seed(78)
            indices = torch.randint(6, (512,), generator=rng)
            eval_clean, eval_labels = clean[indices], labels[indices]
            eval_t = torch.randint(100, (512,), generator=rng)
            eval_noise = torch.randn(512, 64, generator=rng)
            eval_noisy = add_noise(eval_clean, eval_noise, eval_t, alpha_bar)
            def evaluate():
                model.eval()
                with torch.no_grad():
                    return float(F.mse_loss(model(eval_noisy, eval_t, eval_labels), eval_noise))
            history = [{"updates": 0, "noise_mse": evaluate()}]
            for step in range(1, updates + 1):
                model.train()
                batch = torch.randint(6, (128,))
                training_step(model, optimizer, clean[batch], labels[batch], alpha_bar)
                if step in (50, 200, 500, 1000, updates):
                    history.append({"updates": step, "noise_mse": evaluate()})
            sample_labels = torch.tensor([0] * 32 + [1] * 32)
            generated, _ = sample(model, sample_labels)
            accuracy = float((orientation_predictions(generated) == sample_labels).float().mean())
            vertical, vertical_path = sample(model, torch.tensor([0]), seed=42, record=True)
            horizontal, horizontal_path = sample(model, torch.tensor([1]), seed=42, record=True)
            repeat, _ = sample(model, torch.tensor([0]), seed=42)
            assert torch.equal(vertical, repeat)
            assert vertical_path[0]["image"] == horizontal_path[0]["image"]
            assert history[-1]["noise_mse"] < history[0]["noise_mse"] * 0.5
            assert accuracy >= 0.90, accuracy
            report = {"parameters": sum(p.numel() for p in model.parameters()),
                      "updates": updates, "clean_templates": 6, "diffusion_steps": 100,
                      "schedule": {"type": "linear", "beta_start": 0.0001, "beta_end": 0.15,
                                   "terminal_alpha_bar": float(alpha_bar[-1])},
                      "history": history, "new_noise_orientation_accuracy": accuracy,
                      "generated_count": 64, "same_seed_reproducible": True,
                      "same_initial_noise_across_conditions": True,
                      "guidance_scale": 2.0,
                      "scope": "MLP on six training-domain 8x8 bar templates; fresh noise, no held-out clean-image generalization or natural-language capability."}
            data = {"vertical": vertical_path, "horizontal": horizontal_path,
                    "samples": generated[:8].tolist() + generated[32:40].tolist(),
                    "sample_labels": [0] * 8 + [1] * 8,
                    "templates": clean.reshape(-1, 8, 8).tolist()}
            return model, report, data
    finally:
        torch.set_num_threads(previous_threads)


def load_sd_configs():
    return {name: json.loads((CONFIG_DIR / filename).read_text()) for name, filename in
            {"unet": "unet_config.json", "vae": "vae_config.json",
             "text": "text_encoder_config.json", "tokenizer": "tokenizer_tokenizer_config.json",
             "scheduler": "scheduler_scheduler_config.json"}.items()}


def sd_shape_walkthrough(batch=1, cfg=True):
    """Abbreviated meta-tensor contracts; not a full Stable Diffusion forward."""
    config = load_sd_configs()
    width = config["text"]["hidden_size"]
    length = config["tokenizer"]["model_max_length"]
    tokens = torch.ones(batch, length, dtype=torch.long)
    embedding = nn.Embedding(config["text"]["vocab_size"], width, device="meta")
    token_vectors = embedding(tokens)
    text_hidden = torch.empty_like(token_vectors)  # CLIP's contextual output contract.
    rgb = torch.empty(batch, 3, 512, 512, device="meta")
    moments = torch.empty(batch, 8, 64, 64, device="meta")
    mean, log_variance = moments.chunk(2, dim=1)
    sampled = torch.empty_like(mean)  # Posterior-sample shape; no numeric VAE is run.
    encoded = sampled * 0.18215  # Amplitude scaling; spatial size is unchanged.
    latent = torch.empty(batch, 4, 64, 64, device="meta")  # Initial noise for text-to-image.
    model_latent = torch.cat([latent, latent]) if cfg else latent
    null_hidden = torch.empty_like(text_hidden)  # Placeholder contract for empty-prompt encoding.
    model_text = torch.cat([null_hidden, text_hidden]) if cfg else text_hidden
    # The null and conditional branches have the same shape, not the same values.
    channels = config["unet"]["block_out_channels"]
    time_features = torch.empty(model_latent.shape[0], channels[0], device="meta")
    time_hidden = nn.Sequential(nn.Linear(channels[0], channels[0] * 4, device="meta"),
                                nn.SiLU(), nn.Linear(channels[0] * 4, channels[0] * 4,
                                                     device="meta"))(time_features)
    x = nn.Conv2d(4, channels[0], 3, padding=1, device="meta")(model_latent)
    downs = [x]
    for channel in channels[1:]:
        x = F.avg_pool2d(x, 2)
        x = nn.Conv2d(x.shape[1], channel, 3, padding=1, device="meta")(x)
        downs.append(x)
    ups = [x]
    for skip in reversed(downs[:-1]):
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = nn.Conv2d(x.shape[1], skip.shape[1], 3, padding=1, device="meta")(x)
        ups.append(x)
    predicted_noise = nn.Conv2d(channels[0], 4, 3, padding=1, device="meta")(x)
    if cfg:
        eps_unconditional, eps_conditional = predicted_noise.chunk(2)
        guided = eps_unconditional + 7.5 * (eps_conditional - eps_unconditional)
    else:
        guided = predicted_noise
    decoded = torch.empty_like(rgb)  # Actual VAE decoder is not executed here.
    values = {"token_ids": tokens, "text_embeddings": text_hidden, "training_rgb": rgb,
              "vae_posterior_moments": moments, "vae_mean": mean, "vae_log_variance": log_variance,
              "scaled_training_latent": encoded, "initial_noise": latent,
              "cfg_model_input": model_latent, "cfg_text_context": model_text,
              "timestep_features": time_features, "timestep_embedding": time_hidden}
    values.update({f"unet_down_{i}": t for i, t in enumerate(downs)})
    values.update({f"unet_up_{i}": t for i, t in enumerate(ups)})
    values.update({"predicted_noise": predicted_noise, "guided_noise": guided, "decoded_rgb": decoded})
    shapes = {name: list(tensor.shape) for name, tensor in values.items()}
    assert shapes["guided_noise"] == [batch, 4, 64, 64]
    assert shapes["text_embeddings"] == [batch, 77, 768]
    return shapes


if __name__ == "__main__":
    _, report, _ = train_demo()
    print(json.dumps(report, ensure_ascii=False, indent=2))
