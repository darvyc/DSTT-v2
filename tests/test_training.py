"""Tests for the GPT-style training pipeline."""

import torch
import pytest
from dstt import DSTTConfig, DSTTTransformer
from dstt.data import TextDataset, create_datasets
from dstt.tokenizer import CharTokenizer
from dstt.generate import generate, generate_text
from dstt.train_config import TrainConfig


SAMPLE_TEXT = "Hello world! " * 200  # enough for a small dataset


@pytest.fixture
def tokenizer():
    return CharTokenizer.from_text(SAMPLE_TEXT)


@pytest.fixture
def config():
    return DSTTConfig.tiny()


@pytest.fixture
def model(config, tokenizer):
    config.vocab_size = tokenizer.vocab_size
    config.max_seq_len = 64
    return DSTTTransformer(config)


class TestCharTokenizer:

    def test_roundtrip(self, tokenizer):
        text = "Hello world!"
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        assert decoded == text

    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size > 0
        assert tokenizer.vocab_size == len(set(SAMPLE_TEXT))


class TestTextDataset:

    def test_length(self, tokenizer):
        ds = TextDataset(SAMPLE_TEXT, tokenizer, block_size=32)
        assert len(ds) > 0

    def test_item_shape(self, tokenizer):
        ds = TextDataset(SAMPLE_TEXT, tokenizer, block_size=32)
        x, y = ds[0]
        assert x.shape == (32,)
        assert y.shape == (32,)

    def test_target_is_shifted(self, tokenizer):
        ds = TextDataset(SAMPLE_TEXT, tokenizer, block_size=32)
        x, y = ds[0]
        # y should be x shifted right by 1
        ids = tokenizer.encode(SAMPLE_TEXT)
        expected_x = torch.tensor(ids[:32])
        expected_y = torch.tensor(ids[1:33])
        assert torch.equal(x, expected_x)
        assert torch.equal(y, expected_y)

    def test_create_datasets(self, tokenizer):
        train_ds, val_ds = create_datasets(SAMPLE_TEXT, tokenizer, block_size=32)
        assert len(train_ds) > 0
        assert len(val_ds) > 0


class TestTrainConfig:

    def test_lr_schedule_warmup(self):
        tc = TrainConfig(warmup_steps=100, max_steps=1000, learning_rate=1e-3, min_lr=1e-4)
        assert tc.get_lr(0) < tc.get_lr(50) < tc.get_lr(99)

    def test_lr_schedule_peak(self):
        tc = TrainConfig(warmup_steps=100, max_steps=1000, learning_rate=1e-3, min_lr=1e-4)
        peak_lr = tc.get_lr(100)
        assert abs(peak_lr - 1e-3) < 1e-6

    def test_lr_schedule_decay(self):
        tc = TrainConfig(warmup_steps=100, max_steps=1000, learning_rate=1e-3, min_lr=1e-4)
        assert tc.get_lr(500) < tc.get_lr(100)
        assert tc.get_lr(999) >= tc.min_lr

    def test_effective_batch(self):
        tc = TrainConfig(batch_size=16, gradient_accumulation_steps=4)
        assert tc.effective_batch_size == 64


class TestGenerate:

    def test_greedy(self, model, tokenizer):
        prompt = torch.tensor(tokenizer.encode("Hello"), dtype=torch.long).unsqueeze(0)
        out = generate(model, prompt, max_new_tokens=10, temperature=0.0)
        assert out.shape[1] == prompt.shape[1] + 10

    def test_sampling(self, model, tokenizer):
        prompt = torch.tensor(tokenizer.encode("Hello"), dtype=torch.long).unsqueeze(0)
        out = generate(model, prompt, max_new_tokens=10, temperature=1.0, top_k=5)
        assert out.shape[1] == prompt.shape[1] + 10

    def test_generate_text(self, model, tokenizer):
        text = generate_text(model, tokenizer, prompt="He", max_new_tokens=20, temperature=0.8)
        assert isinstance(text, str)
        assert len(text) > 2  # at least the prompt
        assert text.startswith("He")

    def test_top_p(self, model, tokenizer):
        prompt = torch.tensor(tokenizer.encode("Hi"), dtype=torch.long).unsqueeze(0)
        out = generate(model, prompt, max_new_tokens=5, temperature=0.8, top_p=0.9)
        assert out.shape[1] == prompt.shape[1] + 5

    def test_repetition_penalty(self, model, tokenizer):
        prompt = torch.tensor(tokenizer.encode("A"), dtype=torch.long).unsqueeze(0)
        out = generate(model, prompt, max_new_tokens=10, temperature=0.5, repetition_penalty=1.2)
        assert out.shape[1] == prompt.shape[1] + 10
