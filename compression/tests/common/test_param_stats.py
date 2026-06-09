def test_tiny_model_forward(tiny_model, example_inputs):
    out = tiny_model(example_inputs)
    assert out.logits.shape == (2, 16, 128)
