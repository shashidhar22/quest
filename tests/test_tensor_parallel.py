"""
Tests for tensor parallelism module (quest.models.tensor_parallel).

These tests use mock ESM2-like models to verify that:
1. pad_model() correctly pads attention heads when needed
2. apply_tensor_parallelism() replaces layer types appropriately
3. get_tp_loss_fn() returns correct loss functions
4. NeuronBackend(tp_degree=1) behaves identically to default NeuronBackend()
"""

import math
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn


# ============================================================================
# Mock ESM2-like model for testing without HuggingFace dependency
# ============================================================================


class MockConfig:
    """Minimal ESM2-like config."""

    def __init__(self, num_attention_heads=20, hidden_size=640, vocab_size=33):
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size


class MockSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_attention_heads = num_heads
        self.all_head_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)


class MockAttentionOutput(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)


class MockAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.self = MockSelfAttention(hidden_size, num_heads)
        self.output = MockAttentionOutput(hidden_size)


class MockIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)


class MockOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)


class MockLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size=2560):
        super().__init__()
        self.attention = MockAttention(hidden_size, num_heads)
        self.intermediate = MockIntermediate(hidden_size, intermediate_size)
        self.output = MockOutput(intermediate_size, hidden_size)


class MockEncoder(nn.Module):
    def __init__(self, hidden_size, num_heads, num_layers=2):
        super().__init__()
        self.layer = nn.ModuleList(
            [MockLayer(hidden_size, num_heads) for _ in range(num_layers)]
        )


class MockEmbeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=1)


class MockEsm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = MockEmbeddings(config.vocab_size, config.hidden_size)
        self.encoder = MockEncoder(config.hidden_size, config.num_attention_heads)


class MockLMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.decoder = nn.Linear(hidden_size, vocab_size)


class MockESM2Model(nn.Module):
    """Minimal ESM2 model structure for testing tensor parallelism."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or MockConfig()
        self.esm = MockEsm(self.config)
        self.lm_head = MockLMHead(self.config.hidden_size, self.config.vocab_size)


# ============================================================================
# Tests for pad_model
# ============================================================================


class TestPadModel:
    """Tests for the pad_model function."""

    def test_no_padding_needed(self):
        """When num_heads is divisible by tp_degree, no padding occurs."""
        from quest.models.tensor_parallel import pad_model

        config = MockConfig(num_attention_heads=20, hidden_size=640)
        model = MockESM2Model(config)

        result = pad_model(model, tp_degree=4)  # 20 / 4 = 5, no padding

        assert result == 20
        assert config.num_attention_heads == 20

    def test_padding_needed_tp8(self):
        """TP=8 with 20 heads should pad to 24 heads."""
        from quest.models.tensor_parallel import pad_model

        config = MockConfig(num_attention_heads=20, hidden_size=640)
        model = MockESM2Model(config)
        head_dim = 640 // 20  # 32

        result = pad_model(model, tp_degree=8)

        assert result == 24  # ceil(20/8) * 8 = 24
        assert config.num_attention_heads == 24

        # Verify QKV projections are padded
        for layer in model.esm.encoder.layer:
            attn = layer.attention.self
            new_attn_dim = 24 * head_dim  # 768
            assert attn.query.out_features == new_attn_dim
            assert attn.key.out_features == new_attn_dim
            assert attn.value.out_features == new_attn_dim
            assert attn.num_attention_heads == 24

            # Attention output should accept padded input
            assert layer.attention.output.dense.in_features == new_attn_dim

    def test_padding_tp3(self):
        """TP=3 with 20 heads should pad to 21 heads."""
        from quest.models.tensor_parallel import pad_model

        config = MockConfig(num_attention_heads=20, hidden_size=640)
        model = MockESM2Model(config)

        result = pad_model(model, tp_degree=3)

        assert result == 21  # ceil(20/3) * 3 = 21

    def test_padding_preserves_existing_weights(self):
        """Padded weights should preserve original values."""
        from quest.models.tensor_parallel import pad_model

        config = MockConfig(num_attention_heads=20, hidden_size=640)
        model = MockESM2Model(config)

        # Store original weights
        orig_query_weight = model.esm.encoder.layer[0].attention.self.query.weight.data.clone()

        pad_model(model, tp_degree=8)

        # Original portion should be preserved
        padded_weight = model.esm.encoder.layer[0].attention.self.query.weight.data
        assert torch.allclose(padded_weight[:orig_query_weight.shape[0]], orig_query_weight)

        # Padded portion should be zeros
        assert torch.all(padded_weight[orig_query_weight.shape[0]:] == 0)

    def test_tp_degree_1_noop(self):
        """TP degree 1 should always be a no-op (any head count is divisible by 1)."""
        from quest.models.tensor_parallel import pad_model

        config = MockConfig(num_attention_heads=20, hidden_size=640)
        model = MockESM2Model(config)

        result = pad_model(model, tp_degree=1)
        assert result == 20

    def test_compatible_tp_degrees(self):
        """Verify all compatible TP degrees for ESM2 (20 heads) need no padding."""
        from quest.models.tensor_parallel import pad_model

        for tp in [1, 2, 4, 5, 10, 20]:
            config = MockConfig(num_attention_heads=20, hidden_size=640)
            model = MockESM2Model(config)
            result = pad_model(model, tp_degree=tp)
            assert result == 20, f"TP={tp} should not need padding"


# ============================================================================
# Tests for apply_tensor_parallelism
# ============================================================================


class MockColumnParallelLinear(nn.Module):
    """Mock for neuronx_distributed.parallel_layers.ColumnParallelLinear."""

    def __init__(self, in_features, out_features, bias=True, gather_output=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output


class MockRowParallelLinear(nn.Module):
    """Mock for neuronx_distributed.parallel_layers.RowParallelLinear."""

    def __init__(self, in_features, out_features, bias=True, input_is_parallel=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel


class MockParallelEmbedding(nn.Module):
    """Mock for neuronx_distributed.parallel_layers.ParallelEmbedding."""

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx


class TestApplyTensorParallelism:
    """Tests for the apply_tensor_parallelism function."""

    @pytest.fixture(autouse=True)
    def mock_nxd(self):
        """Mock neuronx_distributed imports."""
        mock_module = MagicMock()
        mock_module.parallel_layers.ColumnParallelLinear = MockColumnParallelLinear
        mock_module.parallel_layers.RowParallelLinear = MockRowParallelLinear
        mock_module.parallel_layers.ParallelEmbedding = MockParallelEmbedding

        with patch.dict("sys.modules", {
            "neuronx_distributed": mock_module,
            "neuronx_distributed.parallel_layers": mock_module.parallel_layers,
        }):
            yield

    def test_tp_degree_1_noop(self):
        """TP degree 1 should not modify the model."""
        from quest.models.tensor_parallel import apply_tensor_parallelism

        model = MockESM2Model()
        original_type = type(model.esm.encoder.layer[0].attention.self.query)

        apply_tensor_parallelism(model, tp_degree=1)

        # Should still be regular nn.Linear
        assert type(model.esm.encoder.layer[0].attention.self.query) == original_type

    def test_replaces_qkv_with_column_parallel(self):
        """QKV projections should become ColumnParallelLinear."""
        from quest.models.tensor_parallel import apply_tensor_parallelism

        model = MockESM2Model()
        apply_tensor_parallelism(model, tp_degree=2)

        for layer in model.esm.encoder.layer:
            attn = layer.attention.self
            assert isinstance(attn.query, MockColumnParallelLinear)
            assert isinstance(attn.key, MockColumnParallelLinear)
            assert isinstance(attn.value, MockColumnParallelLinear)
            assert attn.query.gather_output is False

    def test_replaces_attn_output_with_row_parallel(self):
        """Attention output should become RowParallelLinear."""
        from quest.models.tensor_parallel import apply_tensor_parallelism

        model = MockESM2Model()
        apply_tensor_parallelism(model, tp_degree=2)

        for layer in model.esm.encoder.layer:
            dense = layer.attention.output.dense
            assert isinstance(dense, MockRowParallelLinear)
            assert dense.input_is_parallel is True

    def test_replaces_ffn_intermediate_with_column_parallel(self):
        """FFN intermediate should become ColumnParallelLinear."""
        from quest.models.tensor_parallel import apply_tensor_parallelism

        model = MockESM2Model()
        apply_tensor_parallelism(model, tp_degree=2)

        for layer in model.esm.encoder.layer:
            assert isinstance(layer.intermediate.dense, MockColumnParallelLinear)
            assert layer.intermediate.dense.gather_output is False

    def test_replaces_ffn_output_with_row_parallel(self):
        """FFN output should become RowParallelLinear."""
        from quest.models.tensor_parallel import apply_tensor_parallelism

        model = MockESM2Model()
        apply_tensor_parallelism(model, tp_degree=2)

        for layer in model.esm.encoder.layer:
            assert isinstance(layer.output.dense, MockRowParallelLinear)
            assert layer.output.dense.input_is_parallel is True

    def test_replaces_embeddings_with_parallel(self):
        """Word embeddings should become ParallelEmbedding."""
        from quest.models.tensor_parallel import apply_tensor_parallelism

        model = MockESM2Model()
        apply_tensor_parallelism(model, tp_degree=2)

        assert isinstance(model.esm.embeddings.word_embeddings, MockParallelEmbedding)

    def test_replaces_lm_head_decoder(self):
        """LM head decoder should become ColumnParallelLinear."""
        from quest.models.tensor_parallel import apply_tensor_parallelism

        model = MockESM2Model()
        apply_tensor_parallelism(model, tp_degree=2)

        assert isinstance(model.lm_head.decoder, MockColumnParallelLinear)

    def test_updates_num_attention_heads_per_layer(self):
        """Per-layer num_attention_heads should be divided by tp_degree."""
        from quest.models.tensor_parallel import apply_tensor_parallelism

        model = MockESM2Model()  # 20 heads
        apply_tensor_parallelism(model, tp_degree=4)

        for layer in model.esm.encoder.layer:
            assert layer.attention.self.num_attention_heads == 5  # 20 / 4

    def test_tp8_with_padding(self):
        """TP=8 should pad from 20 to 24 heads, then divide."""
        from quest.models.tensor_parallel import apply_tensor_parallelism

        model = MockESM2Model()
        apply_tensor_parallelism(model, tp_degree=8)

        # Should have padded to 24, then 24 / 8 = 3 per partition
        for layer in model.esm.encoder.layer:
            assert layer.attention.self.num_attention_heads == 3


# ============================================================================
# Tests for get_tp_loss_fn
# ============================================================================


class TestGetTpLossFn:
    """Tests for the get_tp_loss_fn function."""

    def test_tp1_returns_none(self):
        """TP=1 should return None (use standard CrossEntropyLoss)."""
        from quest.models.tensor_parallel import get_tp_loss_fn

        assert get_tp_loss_fn(1) is None

    def test_tp_gt1_returns_callable(self):
        """TP > 1 should return parallel_cross_entropy callable."""
        mock_pce = MagicMock()
        mock_loss_module = MagicMock()
        mock_loss_module.parallel_cross_entropy = mock_pce

        with patch.dict("sys.modules", {
            "neuronx_distributed": MagicMock(),
            "neuronx_distributed.parallel_layers": MagicMock(),
            "neuronx_distributed.parallel_layers.loss_functions": mock_loss_module,
        }):
            from quest.models.tensor_parallel import get_tp_loss_fn

            result = get_tp_loss_fn(2)
            assert result is mock_pce

    def test_tp_gt1_import_error(self):
        """TP > 1 without neuronx_distributed should raise ImportError."""
        with patch.dict("sys.modules", {
            "neuronx_distributed": None,
            "neuronx_distributed.parallel_layers": None,
            "neuronx_distributed.parallel_layers.loss_functions": None,
        }):
            # Need to reimport to get fresh import attempts
            import importlib
            import quest.models.tensor_parallel as tp_module
            importlib.reload(tp_module)

            with pytest.raises(ImportError):
                tp_module.get_tp_loss_fn(2)


# ============================================================================
# Tests for NeuronBackend TP behavior
# ============================================================================


class TestNeuronBackendTPDefaults:
    """Verify NeuronBackend(tp_degree=1) matches original behavior."""

    @pytest.fixture
    def mock_xla(self):
        """Mock torch_xla imports."""
        mock_xm = MagicMock()
        mock_xm.xla_device.return_value = torch.device("cpu")
        mock_pl = MagicMock()

        with patch.dict("sys.modules", {
            "torch_xla": MagicMock(),
            "torch_xla.core": MagicMock(),
            "torch_xla.core.xla_model": mock_xm,
            "torch_xla.distributed": MagicMock(),
            "torch_xla.distributed.parallel_loader": mock_pl,
            "torch_xla.distributed.xla_backend": MagicMock(),
        }):
            # Patch module-level globals
            import quest.training.backends.neuron_backend as nb_module
            old_available = nb_module._XLA_AVAILABLE
            old_xm = nb_module._xm
            old_pl = nb_module._pl

            nb_module._XLA_AVAILABLE = True
            nb_module._xm = mock_xm
            nb_module._pl = mock_pl

            yield mock_xm, mock_pl

            nb_module._XLA_AVAILABLE = old_available
            nb_module._xm = old_xm
            nb_module._pl = old_pl

    def test_default_tp_degree_is_1(self, mock_xla):
        """Default NeuronBackend should have tp_degree=1."""
        from quest.training.backends.neuron_backend import NeuronBackend

        backend = NeuronBackend()
        assert backend.tp_degree == 1
        assert backend.pp_degree == 1

    def test_tp1_wrap_model_returns_unchanged(self, mock_xla):
        """TP=1 wrap_model_distributed should return model unchanged."""
        from quest.training.backends.neuron_backend import NeuronBackend

        backend = NeuronBackend(tp_degree=1)
        model = nn.Linear(10, 10)

        result = backend.wrap_model_distributed(model, local_rank=0)
        assert result is model

    def test_tp1_optimizer_step_standard(self, mock_xla):
        """TP=1 optimizer_step should call xm.optimizer_step without groups."""
        mock_xm, _ = mock_xla
        from quest.training.backends.neuron_backend import NeuronBackend

        backend = NeuronBackend(tp_degree=1)
        optimizer = MagicMock()

        backend.optimizer_step(optimizer)

        mock_xm.optimizer_step.assert_called_once_with(optimizer)

    def test_tp1_save_checkpoint_standard(self, mock_xla):
        """TP=1 save_checkpoint should use xm.save."""
        mock_xm, _ = mock_xla
        from quest.training.backends.neuron_backend import NeuronBackend

        backend = NeuronBackend(tp_degree=1)
        state_dict = {"key": "value"}

        backend.save_checkpoint(state_dict, "/tmp/test.pt", True)

        mock_xm.save.assert_called_once_with(state_dict, "/tmp/test.pt", master_only=True)

    def test_tp1_clip_grad_norm_standard(self, mock_xla):
        """TP=1 clip_grad_norm should use standard PyTorch clipping."""
        from quest.training.backends.neuron_backend import NeuronBackend

        backend = NeuronBackend(tp_degree=1)
        model = nn.Linear(10, 10)
        # Set some gradients
        model.weight.grad = torch.randn_like(model.weight)
        model.bias.grad = torch.randn_like(model.bias)

        result = backend.clip_grad_norm(model, max_norm=1.0)
        assert isinstance(result, torch.Tensor)

    def test_tp_gt1_stores_degree(self, mock_xla):
        """TP > 1 should store the degree."""
        from quest.training.backends.neuron_backend import NeuronBackend

        backend = NeuronBackend(tp_degree=4, pp_degree=2)
        assert backend.tp_degree == 4
        assert backend.pp_degree == 2


# ============================================================================
# Tests for get_backend factory with TP args
# ============================================================================


class TestGetBackendTP:
    """Test that get_backend passes TP args through."""

    @pytest.fixture
    def mock_xla(self):
        """Mock XLA availability."""
        mock_xm = MagicMock()
        mock_xm.xla_device.return_value = torch.device("cpu")

        with patch.dict("sys.modules", {
            "torch_xla": MagicMock(),
            "torch_xla.core": MagicMock(),
            "torch_xla.core.xla_model": mock_xm,
            "torch_xla.distributed": MagicMock(),
            "torch_xla.distributed.parallel_loader": MagicMock(),
            "torch_xla.distributed.xla_backend": MagicMock(),
        }):
            import quest.training.backends.neuron_backend as nb_module
            old_available = nb_module._XLA_AVAILABLE
            old_xm = nb_module._xm
            old_pl = nb_module._pl

            nb_module._XLA_AVAILABLE = True
            nb_module._xm = mock_xm
            nb_module._pl = MagicMock()

            yield

            nb_module._XLA_AVAILABLE = old_available
            nb_module._xm = old_xm
            nb_module._pl = old_pl

    def test_explicit_xla_passes_tp(self, mock_xla):
        """get_backend('xla', tp_degree=4) should create NeuronBackend with TP=4."""
        from quest.training.backends import get_backend

        backend = get_backend("xla", tp_degree=4, pp_degree=2)
        assert backend.tp_degree == 4
        assert backend.pp_degree == 2

    def test_default_tp_is_1(self, mock_xla):
        """get_backend('xla') should default to TP=1."""
        from quest.training.backends import get_backend

        backend = get_backend("xla")
        assert backend.tp_degree == 1
        assert backend.pp_degree == 1
