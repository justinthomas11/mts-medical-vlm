from types import SimpleNamespace

import torch
from PIL import Image

from src.models.maira2 import Maira2Generator, clean_indication


def test_clean_indication():
    assert clean_indication("Dyspnea") == "Dyspnea."
    assert clean_indication(" Chest pain. ") == "Chest pain."
    assert clean_indication("") is None and clean_indication(None) is None and clean_indication(float("nan")) is None


class _Proc:
    tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)

    def format_and_preprocess_reporting_input(self, **kw):
        assert kw["current_frontal"].mode == "L" and kw["prior_report"] is None and kw["get_grounding"] is False
        self.indication = kw["indication"]
        return {"input_ids": torch.tensor([[5, 6, 7]]), "pixel_values": torch.zeros(1, 3, 4, 4)}

    def decode(self, ids, skip_special_tokens=True):
        return " " + " ".join(f"t{int(i)}" for i in ids if int(i) != 2)

    def convert_output_to_plaintext_or_grounded_sequence(self, text):
        return text


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids, pixel_values, do_sample, **kw):
        new = torch.tensor([[9, 8, 2, 0]])
        if not do_sample:
            logits = tuple(torch.zeros(1, 10) for _ in range(4))   # uniform -> entropy log(10)
            return SimpleNamespace(sequences=torch.cat([input_ids, new], 1), logits=logits)
        return torch.cat([input_ids, new], 1).repeat(kw["num_return_sequences"], 1)


def test_generate_plumbing_entropy_and_samples():
    g = object.__new__(Maira2Generator)
    g.processor, g.model, g.max_new_tokens = _Proc(), _Model(), 10
    g.eos_token_id, g.pad_token_id = 2, 0
    r = g.generate(Image.new("RGB", (8, 8)), "Cough.", n_samples=3, seed=1)
    assert r.text == "t9 t8 t0" or r.text.startswith("t9 t8")
    assert r.n_tokens == 3                                  # up to and including the first EOS
    assert abs(r.mean_token_entropy - torch.log(torch.tensor(10.0)).item()) < 1e-5
    assert len(r.samples) == 3 and g.processor.indication == "Cough."
