"""Tests for multi-layer SAE extraction functions."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from model_evaluation.main_agent.gemma_scope_sae import (
    AVAILABLE_LAYERS,
    EVALUATION_LAYERS,
    MultiLayerSAEFeatureResult,
    SAEConfig,
    SAEFeatureResult,
    extract_multi_layer_sae_features,
    gather_multi_layer_residual_activations,
    get_evaluation_layers,
)


class TestGetEvaluationLayers:
    """Tests for get_evaluation_layers."""

    def test_returns_tuple_for_each_model_size(self) -> None:
        """Each model size should return a (middle, upper) tuple."""
        for size in ("1b", "4b", "12b", "27b"):
            result = get_evaluation_layers(model_size=size)
            assert isinstance(result, tuple)
            assert len(result) == 2

    def test_layers_are_valid(self) -> None:
        """Returned layers must exist in AVAILABLE_LAYERS."""
        for size in ("1b", "4b", "12b", "27b"):
            middle, upper = get_evaluation_layers(model_size=size)
            assert middle in AVAILABLE_LAYERS[size]
            assert upper in AVAILABLE_LAYERS[size]

    def test_middle_is_less_than_upper(self) -> None:
        """Middle layer index should be smaller than upper."""
        for size in ("1b", "4b", "12b", "27b"):
            middle, upper = get_evaluation_layers(model_size=size)
            assert middle < upper

    def test_invalid_model_size_raises(self) -> None:
        """Unknown model size should raise KeyError."""
        with pytest.raises(KeyError):
            get_evaluation_layers(model_size="99b")

    def test_evaluation_layers_matches_constant(self) -> None:
        """Function should return values from the EVALUATION_LAYERS dict."""
        for size, expected in EVALUATION_LAYERS.items():
            assert get_evaluation_layers(model_size=size) == expected


class TestGatherMultiLayerResidualActivations:
    """Tests for gather_multi_layer_residual_activations."""

    @pytest.fixture()
    def mock_model(self) -> MagicMock:
        """Create a mock model with hookable layers."""
        model = MagicMock()
        model.parameters.return_value = iter([torch.zeros(1)])

        d_model = 64
        seq_len = 10

        layer_modules: list[MagicMock] = []
        for _ in range(5):
            layer_mod = MagicMock()
            hooks: list = []

            def make_register(hooks_list: list) -> callable:
                def register_forward_hook(hook_fn: callable) -> MagicMock:
                    hooks_list.append(hook_fn)
                    handle = MagicMock()
                    handle.remove = MagicMock()
                    return handle

                return register_forward_hook

            layer_mod.register_forward_hook = make_register(hooks)
            layer_mod._hooks = hooks
            layer_modules.append(layer_mod)

        module_list = MagicMock()
        module_list.__getitem__ = lambda self, idx: layer_modules[idx]

        def forward_side_effect(input_ids: torch.Tensor) -> MagicMock:
            for layer_mod in layer_modules:
                for hook_fn in layer_mod._hooks:
                    output = (torch.randn(1, seq_len, d_model),)
                    hook_fn(layer_mod, (input_ids,), output)
            return MagicMock()

        model.side_effect = forward_side_effect

        with patch(
            "model_evaluation.main_agent.gemma_scope_sae._get_model_layers",
            return_value=module_list,
        ):
            model._module_list = module_list
            model._layer_modules = layer_modules

        return model

    def test_returns_activations_for_each_layer(self, mock_model: MagicMock) -> None:
        """Should return a dict with an entry for each requested layer."""
        input_ids = torch.randint(0, 100, (1, 10))
        target_layers = [1, 3]

        with patch(
            "model_evaluation.main_agent.gemma_scope_sae._get_model_layers",
            return_value=mock_model._module_list,
        ):
            result = gather_multi_layer_residual_activations(
                model=mock_model,
                target_layers=target_layers,
                input_ids=input_ids,
            )

        assert set(result.keys()) == {1, 3}
        for layer_idx in target_layers:
            assert isinstance(result[layer_idx], torch.Tensor)

    def test_hooks_are_removed_after_forward(self, mock_model: MagicMock) -> None:
        """All hooks must be removed after the forward pass."""
        input_ids = torch.randint(0, 100, (1, 10))

        with patch(
            "model_evaluation.main_agent.gemma_scope_sae._get_model_layers",
            return_value=mock_model._module_list,
        ):
            gather_multi_layer_residual_activations(
                model=mock_model,
                target_layers=[0, 2],
                input_ids=input_ids,
            )

        for layer_mod in mock_model._layer_modules:
            for _hook in layer_mod._hooks:
                pass


class TestExtractMultiLayerSaeFeatures:
    """Tests for extract_multi_layer_sae_features."""

    def _make_mock_sae(self, *, d_in: int, d_sae: int) -> MagicMock:
        """Create a mock SAE that returns deterministic outputs."""
        sae = MagicMock()

        def encode(acts: torch.Tensor) -> torch.Tensor:
            batch_shape = acts.shape[:-1]
            return torch.rand(*batch_shape, d_sae)

        def decode(acts: torch.Tensor) -> torch.Tensor:
            batch_shape = acts.shape[:-1]
            return torch.rand(*batch_shape, d_in)

        sae.encode = encode
        sae.decode = decode
        return sae

    @pytest.fixture()
    def mock_setup(self) -> dict:
        """Create mock model, tokenizer, and SAEs for two layers."""
        d_in = 64
        d_sae = 128
        seq_len = 8
        prompt_len = 5

        model = MagicMock()
        model.parameters.return_value = iter([torch.zeros(1, device="cpu")])

        generated_ids = torch.randint(0, 100, (1, seq_len))
        model.generate.return_value = generated_ids

        input_ids = torch.randint(0, 100, (1, prompt_len))

        class TokenizerOutput(dict):
            """Dict subclass mimicking HuggingFace BatchEncoding."""

            def to(self, device: str) -> "TokenizerOutput":
                return self

            def __getattr__(self, name: str) -> torch.Tensor:
                if name in self:
                    return self[name]
                raise AttributeError(name)

        token_output = TokenizerOutput(
            input_ids=input_ids,
            attention_mask=torch.ones(1, prompt_len),
        )

        tokenizer = MagicMock()
        tokenizer.side_effect = lambda *args, **kwargs: token_output
        tokenizer.decode.return_value = "mock answer"
        tokenizer.convert_ids_to_tokens.return_value = [f"tok_{i}" for i in range(seq_len)]

        sae_10 = self._make_mock_sae(d_in=d_in, d_sae=d_sae)
        config_10 = SAEConfig(
            model_size="4b",
            model_type="it",
            layer=10,
            width="16k",
            l0_size="medium",
            d_in=d_in,
            d_sae=d_sae,
        )

        sae_20 = self._make_mock_sae(d_in=d_in, d_sae=d_sae)
        config_20 = SAEConfig(
            model_size="4b",
            model_type="it",
            layer=20,
            width="16k",
            l0_size="medium",
            d_in=d_in,
            d_sae=d_sae,
        )

        return {
            "model": model,
            "tokenizer": tokenizer,
            "saes": {
                10: (sae_10, config_10),
                20: (sae_20, config_20),
            },
            "d_in": d_in,
            "d_sae": d_sae,
            "seq_len": seq_len,
            "prompt_len": prompt_len,
        }

    def test_returns_multi_layer_result(self, mock_setup: dict) -> None:
        """Should return a MultiLayerSAEFeatureResult."""
        residuals = {
            10: torch.randn(mock_setup["seq_len"], mock_setup["d_in"]),
            20: torch.randn(mock_setup["seq_len"], mock_setup["d_in"]),
        }

        with patch(
            "model_evaluation.main_agent.gemma_scope_sae.gather_multi_layer_residual_activations",
            return_value=residuals,
        ):
            result = extract_multi_layer_sae_features(
                model=mock_setup["model"],
                tokenizer=mock_setup["tokenizer"],
                saes=mock_setup["saes"],
                text="test prompt",
                max_new_tokens=50,
                top_k=5,
            )

        assert isinstance(result, MultiLayerSAEFeatureResult)
        assert set(result.layer_results.keys()) == {10, 20}
        assert result.answer == "mock answer"
        assert len(result.tokens) == mock_setup["seq_len"]

    def test_each_layer_has_sae_feature_result(self, mock_setup: dict) -> None:
        """Each layer in the result should be a proper SAEFeatureResult."""
        residuals = {
            10: torch.randn(mock_setup["seq_len"], mock_setup["d_in"]),
            20: torch.randn(mock_setup["seq_len"], mock_setup["d_in"]),
        }

        with patch(
            "model_evaluation.main_agent.gemma_scope_sae.gather_multi_layer_residual_activations",
            return_value=residuals,
        ):
            result = extract_multi_layer_sae_features(
                model=mock_setup["model"],
                tokenizer=mock_setup["tokenizer"],
                saes=mock_setup["saes"],
                text="test prompt",
                max_new_tokens=50,
                top_k=5,
            )

        for layer_idx in (10, 20):
            layer_result = result.layer_results[layer_idx]
            assert isinstance(layer_result, SAEFeatureResult)
            assert isinstance(layer_result.l0, float)
            assert isinstance(layer_result.fvu, float)
            assert layer_result.top_features.shape[-1] == 5
            assert layer_result.top_activations.shape[-1] == 5

    def test_shared_answer_across_layers(self, mock_setup: dict) -> None:
        """All layers should share the same generated answer."""
        residuals = {
            10: torch.randn(mock_setup["seq_len"], mock_setup["d_in"]),
            20: torch.randn(mock_setup["seq_len"], mock_setup["d_in"]),
        }

        with patch(
            "model_evaluation.main_agent.gemma_scope_sae.gather_multi_layer_residual_activations",
            return_value=residuals,
        ):
            result = extract_multi_layer_sae_features(
                model=mock_setup["model"],
                tokenizer=mock_setup["tokenizer"],
                saes=mock_setup["saes"],
                text="test prompt",
                max_new_tokens=50,
            )

        for layer_result in result.layer_results.values():
            assert layer_result.answer == result.answer
            assert layer_result.tokens == result.tokens
