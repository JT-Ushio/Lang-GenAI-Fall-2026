"""Offline CPU teaching model: two RGB patches + question -> colour + EOS.

This is a small, randomly initialized model, not a LLaVA implementation or a
pretrained VLM. All eight combinations are used for training; reported answers
only check fitting and controlled changes, not held-out generalization.
"""
import json
import torch
from torch import nn
from torch.nn import functional as F

VOCAB = ["<pad>", "<bos>", "<eos>", "left?", "right?", "red", "blue"]
IDS = {word: i for i, word in enumerate(VOCAB)}


def make_data():
    colours = {"red": torch.tensor([1., 0., 0.]),
               "blue": torch.tensor([0., 0., 1.])}
    images, tokens, names = [], [], []
    for left in colours:
        for right in colours:
            pixels = torch.zeros(3, 8, 16)
            pixels[:, :, :8] = colours[left][:, None, None]
            pixels[:, :, 8:] = colours[right][:, None, None]
            for question, answer in [("left?", left), ("right?", right)]:
                images.append(pixels.clone())
                tokens.append([IDS[question], IDS["<bos>"],
                               IDS[answer], IDS["<eos>"]])
                names.append((left, right, question, answer))
    return torch.stack(images), torch.tensor(tokens), names


class MiniVLM(nn.Module):
    def __init__(self):
        super().__init__()
        # Local patch embedding only: this is not a complete ViT or CLIP.
        self.vision = nn.Conv2d(3, 16, kernel_size=8, stride=8)
        self.projector = nn.Linear(16, 32)
        self.words = nn.Embedding(len(VOCAB), 32)
        self.positions = nn.Embedding(8, 32)
        layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64,
            dropout=0., batch_first=True)
        # A causal mask makes these Transformer blocks act as a decoder-only LM.
        self.lm = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(32, len(VOCAB))

    def forward(self, pixels, token_ids):
        patches = self.vision(pixels).flatten(2).transpose(1, 2)
        visual_tokens = self.projector(patches)
        text_tokens = self.words(token_ids)
        sequence = torch.cat([visual_tokens, text_tokens], dim=1)
        positions = torch.arange(sequence.shape[1], device=sequence.device)
        sequence = sequence + self.positions(positions)[None]
        mask = torch.triu(torch.ones(sequence.shape[1], sequence.shape[1],
                                    dtype=torch.bool, device=sequence.device), 1)
        hidden = self.lm(sequence, mask=mask)
        return self.head(hidden)


def answer_loss(logits, token_ids):
    # Two visual positions + question + BOS are input context, not targets.
    labels = torch.full(logits.shape[:2], -100, dtype=torch.long,
                        device=logits.device)
    labels[:, 4:] = token_ids[:, 2:]
    # Position t predicts the token at t+1; EOS is also supervised.
    return F.cross_entropy(logits[:, :-1].reshape(-1, len(VOCAB)),
                           labels[:, 1:].reshape(-1), ignore_index=-100)


@torch.no_grad()
def generate(model, pixels, question_id, max_new_tokens=3):
    model.eval()
    prefix = torch.tensor([[question_id, IDS["<bos>"]]], device=pixels.device)
    generated = []
    for _ in range(max_new_tokens):
        logits = model(pixels, prefix)
        next_id = int(logits[0, -1].argmax())
        generated.append(next_id)
        prefix = torch.cat([prefix, prefix.new_tensor([[next_id]])], dim=1)
        if next_id == IDS["<eos>"]:
            break
    return [VOCAB[i] for i in generated]


def train_demo(steps=180):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(4)
            model = MiniVLM()
            pixels, tokens, names = make_data()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            history = []
            for step in range(steps + 1):
                model.train()
                loss = answer_loss(model(pixels, tokens), tokens)
                if step in (0, 1, 10, 30, 60, 120, steps):
                    history.append({"step": step, "loss": round(float(loss.detach()), 6)})
                if step < steps:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            answers = [generate(model, pixels[i:i + 1], int(tokens[i, 0]))
                       for i in range(len(names))]
            correct = sum(answer == [name[3], "<eos>"]
                          for answer, name in zip(answers, names))
            assert history[-1]["loss"] < history[0]["loss"] * 0.1
            assert correct == len(names), (names, answers)
            # Changing a future answer token must not affect its prediction.
            model.eval()
            changed = tokens.clone()
            changed[:, 2] = IDS["blue"]
            with torch.no_grad():
                original = model(pixels, tokens)[:, 3]
                modified = model(pixels, changed)[:, 3]
            causal_delta = float((original - modified).abs().max())
            assert causal_delta < 1e-6
            report = {"training_examples": len(names), "updates": steps,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "history": history, "training_exact_match": f"{correct}/{len(names)}",
                      "causal_future_token_delta": causal_delta,
                      "answers": [{"left": name[0], "right": name[1],
                                   "question": name[2], "expected": name[3],
                                   "generated": answer}
                                  for name, answer in zip(names, answers)],
                      "scope": "All examples were used in training; no generalization claim."}
            return model, report
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    _, result = train_demo()
    print(json.dumps(result, ensure_ascii=False, indent=2))
