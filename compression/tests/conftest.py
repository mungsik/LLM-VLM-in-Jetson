import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM


@pytest.fixture
def tiny_model():
    """Phi-4 구조(GQA, MLP gate/up/down)를 축소 재현한 결정론적 프록시 모델."""
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,   # GQA
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config).eval()
    return model


@pytest.fixture
def example_inputs():
    """프루닝 DependencyGraph 빌드/forward용 입력."""
    torch.manual_seed(1)
    return torch.randint(0, 128, (2, 16))
