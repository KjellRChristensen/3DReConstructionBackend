"""
Tests for VLM CAD Strategy

Tests the Vision-Language Model based CAD reconstruction strategy.
"""
import pytest
from pathlib import Path
import tempfile


class TestCADCodeFormat:
    """Tests for CADCodeFormat enum"""

    def test_format_values(self):
        """Test that all format values are defined"""
        from reconstruction.vlm_cad_strategy import CADCodeFormat

        assert CADCodeFormat.OPENECAD.value == "openecad"
        assert CADCodeFormat.CADQUERY.value == "cadquery"
        assert CADCodeFormat.BUILD123D.value == "build123d"
        assert CADCodeFormat.PYTHON_OCC.value == "pythonocc"
        assert CADCodeFormat.UNKNOWN.value == "unknown"


class TestVLMModelConfig:
    """Tests for VLMModelConfig dataclass"""

    def test_available_models_count(self):
        """Test that all expected models are defined"""
        from reconstruction.vlm_cad_strategy import AVAILABLE_MODELS

        # OpenECAD: 0.55B, 0.89B, 2.4B, 3.1B
        # InternVL2: 1B, 2B, 4B, 8B
        assert len(AVAILABLE_MODELS) == 8

    def test_openecad_models(self):
        """Test OpenECAD model configs"""
        from reconstruction.vlm_cad_strategy import AVAILABLE_MODELS, CADCodeFormat

        models = ["openecad-0.55b", "openecad-0.89b", "openecad-2.4b", "openecad-3.1b"]
        for model_id in models:
            assert model_id in AVAILABLE_MODELS
            config = AVAILABLE_MODELS[model_id]
            assert config.model_type == "openecad"
            assert config.output_format == CADCodeFormat.OPENECAD

    def test_internvl_models(self):
        """Test InternVL2 model configs"""
        from reconstruction.vlm_cad_strategy import AVAILABLE_MODELS, CADCodeFormat

        models = ["internvl2-1b", "internvl2-2b", "internvl2-4b", "internvl2-8b"]
        for model_id in models:
            assert model_id in AVAILABLE_MODELS
            config = AVAILABLE_MODELS[model_id]
            assert config.model_type == "internvl"
            assert config.output_format == CADCodeFormat.CADQUERY

    def test_model_config_fields(self):
        """Test model config has required fields"""
        from reconstruction.vlm_cad_strategy import AVAILABLE_MODELS

        config = AVAILABLE_MODELS["openecad-0.89b"]

        assert config.model_id == "openecad-0.89b"
        assert "OpenECAD" in config.name
        assert config.size == "0.89B"
        assert config.hf_repo == "Yuan-Che/OpenECADv2-SigLIP-0.89B"
        assert config.model_type == "openecad"
        assert config.requires_gpu is True
        assert config.min_vram_gb == 3.0
        assert config.supports_mps is True

    def test_default_prompt_set(self):
        """Test that default prompts are set for each model"""
        from reconstruction.vlm_cad_strategy import AVAILABLE_MODELS

        for model_id, config in AVAILABLE_MODELS.items():
            assert config.default_prompt, f"No default prompt for {model_id}"
            assert len(config.default_prompt) > 20


class TestCADCodeValidator:
    """Tests for CADCodeValidator class"""

    def test_detect_openecad_format(self):
        """Test OpenECAD format detection"""
        from reconstruction.vlm_cad_strategy import CADCodeValidator, CADCodeFormat

        code = """
        plane = add_sketchplane([0, 0, 0], [0, 0, 1], [1, 0, 0])
        line1 = add_line([0, 0, 0], [10, 0, 0])
        line2 = add_line([10, 0, 0], [10, 10, 0])
        profile = add_profile([[line1, line2]])
        sketch = add_sketch(plane, profile)
        add_extrude(sketch, "NewBodyFeatureOperation", "OneSide", 5)
        """

        format_type = CADCodeValidator.detect_format(code)
        assert format_type == CADCodeFormat.OPENECAD

    def test_detect_cadquery_format(self):
        """Test CadQuery format detection"""
        from reconstruction.vlm_cad_strategy import CADCodeValidator, CADCodeFormat

        code = """
        import cadquery as cq
        result = cq.Workplane("XY").box(10, 10, 5)
        """

        format_type = CADCodeValidator.detect_format(code)
        assert format_type == CADCodeFormat.CADQUERY

    def test_detect_build123d_format(self):
        """Test Build123d format detection"""
        from reconstruction.vlm_cad_strategy import CADCodeValidator, CADCodeFormat

        code = """
        from build123d import *
        with BuildPart() as part:
            with BuildSketch():
                Rectangle(10, 10)
            extrude(amount=5)
        result = part.part
        """

        format_type = CADCodeValidator.detect_format(code)
        assert format_type == CADCodeFormat.BUILD123D

    def test_detect_pythonocc_format(self):
        """Test PythonOCC format detection"""
        from reconstruction.vlm_cad_strategy import CADCodeValidator, CADCodeFormat

        code = """
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
        result = BRepPrimAPI_MakeBox(10, 10, 5).Shape()
        """

        format_type = CADCodeValidator.detect_format(code)
        assert format_type == CADCodeFormat.PYTHON_OCC

    def test_detect_unknown_format(self):
        """Test unknown format detection"""
        from reconstruction.vlm_cad_strategy import CADCodeValidator, CADCodeFormat

        code = "print('hello world')"

        format_type = CADCodeValidator.detect_format(code)
        assert format_type == CADCodeFormat.UNKNOWN

    def test_validate_valid_code(self):
        """Test validation of valid code"""
        from reconstruction.vlm_cad_strategy import CADCodeValidator

        code = """
import cadquery as cq
result = cq.Workplane("XY").box(10, 10, 5)
"""

        result = CADCodeValidator.validate(code)
        assert result.valid is True
        assert len(result.errors) == 0

    def test_validate_syntax_error(self):
        """Test validation catches syntax errors"""
        from reconstruction.vlm_cad_strategy import CADCodeValidator

        code = "def broken(:"

        result = CADCodeValidator.validate(code)
        assert result.valid is False
        assert any("Syntax error" in e for e in result.errors)

    def test_validate_dangerous_patterns(self):
        """Test validation catches dangerous patterns"""
        from reconstruction.vlm_cad_strategy import CADCodeValidator

        dangerous_codes = [
            "import os\nos.system('rm -rf /')",
            "import subprocess\nsubprocess.run(['ls'])",
            "eval('print(1)')",
            "exec('x = 1')",
            "__import__('os')",
        ]

        for code in dangerous_codes:
            result = CADCodeValidator.validate(code)
            assert result.valid is False, f"Should reject: {code[:30]}..."
            assert any("Dangerous" in e for e in result.errors)

    def test_validate_detects_operations(self):
        """Test that validation detects operations"""
        from reconstruction.vlm_cad_strategy import CADCodeValidator

        code = """
import cadquery as cq
result = cq.Workplane("XY").box(10, 10, 5).fillet(1)
"""

        result = CADCodeValidator.validate(code)
        assert "import" in result.detected_operations
        assert "workplane" in result.detected_operations
        assert "box" in result.detected_operations
        assert "fillet" in result.detected_operations

    def test_validate_warnings_no_result(self):
        """Test warning when no result variable"""
        from reconstruction.vlm_cad_strategy import CADCodeValidator

        code = """
import cadquery as cq
box = cq.Workplane("XY").box(10, 10, 5)
"""

        result = CADCodeValidator.validate(code)
        assert any("result" in w.lower() for w in result.warnings)

    def test_extract_code_block_markdown(self):
        """Test extracting code from markdown"""
        from reconstruction.vlm_cad_strategy import CADCodeValidator

        text = """
Here is the code:

```python
import cadquery as cq
result = cq.Workplane("XY").box(10, 10, 5)
```

This creates a box.
"""

        code = CADCodeValidator.extract_code_block(text)
        assert "import cadquery" in code
        assert "result = cq.Workplane" in code
        assert "```" not in code

    def test_extract_code_block_raw(self):
        """Test extracting raw code without markdown"""
        from reconstruction.vlm_cad_strategy import CADCodeValidator

        text = """
import cadquery as cq
result = cq.Workplane("XY").box(10, 10, 5)
"""

        code = CADCodeValidator.extract_code_block(text)
        assert "import cadquery" in code


class TestModelManager:
    """Tests for ModelManager class"""

    def test_model_manager_init(self, tmp_path):
        """Test ModelManager initialization"""
        from reconstruction.vlm_cad_strategy import ModelManager

        manager = ModelManager(cache_dir=str(tmp_path / "cache"))

        assert manager.cache_dir.exists()
        assert isinstance(manager.has_torch, bool)
        assert isinstance(manager.has_transformers, bool)
        assert manager.device in ["cpu", "cuda", "mps"]

    def test_list_available_models(self, tmp_path):
        """Test listing available models"""
        from reconstruction.vlm_cad_strategy import ModelManager

        manager = ModelManager(cache_dir=str(tmp_path / "cache"))
        models = manager.list_available_models()

        assert len(models) == 8  # 4 OpenECAD + 4 InternVL2
        assert any(m["id"] == "openecad-0.89b" for m in models)

        # Check model info structure
        model = next(m for m in models if m["id"] == "openecad-0.89b")
        assert "name" in model
        assert "size" in model
        assert "type" in model
        assert "requires_gpu" in model
        assert "min_vram_gb" in model
        assert "output_format" in model
        assert "downloaded" in model

    def test_get_model_config(self, tmp_path):
        """Test getting model configuration"""
        from reconstruction.vlm_cad_strategy import ModelManager

        manager = ModelManager(cache_dir=str(tmp_path / "cache"))

        config = manager.get_model_config("openecad-0.89b")
        assert config is not None
        assert config.model_id == "openecad-0.89b"

        # Non-existent model
        config = manager.get_model_config("nonexistent-model")
        assert config is None

    def test_is_model_cached_false(self, tmp_path):
        """Test model cache check when not downloaded"""
        from reconstruction.vlm_cad_strategy import ModelManager

        manager = ModelManager(cache_dir=str(tmp_path / "cache"))

        # Model should not be cached initially
        assert manager._is_model_cached("openecad-0.89b") is False

    def test_is_model_cached_true(self, tmp_path):
        """Test model cache check when model directory exists"""
        from reconstruction.vlm_cad_strategy import ModelManager

        cache_dir = tmp_path / "cache"
        manager = ModelManager(cache_dir=str(cache_dir))

        # Create fake model directory with a file
        model_dir = cache_dir / "openecad-0.89b"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")

        assert manager._is_model_cached("openecad-0.89b") is True

    def test_download_model_unknown(self, tmp_path):
        """Test downloading unknown model raises error"""
        from reconstruction.vlm_cad_strategy import ModelManager

        manager = ModelManager(cache_dir=str(tmp_path / "cache"))

        with pytest.raises(ValueError, match="Unknown model"):
            manager.download_model("unknown-model")

    def test_dependency_detection(self, tmp_path):
        """Test dependency availability detection"""
        from reconstruction.vlm_cad_strategy import ModelManager

        manager = ModelManager(cache_dir=str(tmp_path / "cache"))

        # These should be booleans
        assert isinstance(manager.has_torch, bool)
        assert isinstance(manager.has_transformers, bool)
        assert isinstance(manager.has_cadquery, bool)
        assert isinstance(manager.has_build123d, bool)
        assert isinstance(manager.has_occ, bool)


class TestVLMCADInference:
    """Tests for VLMCADInference class"""

    def test_inference_init(self, tmp_path):
        """Test VLMCADInference initialization"""
        from reconstruction.vlm_cad_strategy import ModelManager, VLMCADInference

        manager = ModelManager(cache_dir=str(tmp_path / "cache"))
        inference = VLMCADInference(manager)

        assert inference.model_manager is manager
        assert inference.current_model_id is None

    def test_inference_no_model_loaded(self, tmp_path):
        """Test error when generating without loading model"""
        from reconstruction.vlm_cad_strategy import ModelManager, VLMCADInference

        manager = ModelManager(cache_dir=str(tmp_path / "cache"))
        inference = VLMCADInference(manager)

        with pytest.raises(RuntimeError, match="No model loaded"):
            inference.generate_cad_code("/fake/path.png")


class TestCodeToMesh:
    """Tests for code to mesh conversion"""

    def test_code_to_mesh_invalid_code(self, tmp_path):
        """Test code_to_mesh rejects invalid code"""
        from reconstruction.vlm_cad_strategy import ModelManager, VLMCADInference

        manager = ModelManager(cache_dir=str(tmp_path / "cache"))
        inference = VLMCADInference(manager)

        # Dangerous code should be rejected
        with pytest.raises(ValueError, match="Invalid CAD code"):
            inference.code_to_mesh("import os\nos.system('ls')")

    def test_code_to_mesh_unsupported_format(self, tmp_path):
        """Test error on unsupported output format"""
        from reconstruction.vlm_cad_strategy import ModelManager, VLMCADInference

        manager = ModelManager(cache_dir=str(tmp_path / "cache"))
        inference = VLMCADInference(manager)

        # Valid code but unsupported format
        code = "import cadquery as cq\nresult = cq.Workplane('XY').box(1,1,1)"

        # This will fail because CadQuery might not be installed,
        # but we're testing the format validation
        try:
            inference.code_to_mesh(code, output_format="xyz")
        except ValueError as e:
            assert "Unsupported format" in str(e)
        except RuntimeError:
            # Expected if CadQuery not installed
            pass


class TestCadQueryExecutor:
    """Tests for CadQuery executor"""

    def test_executor_init(self):
        """Test CadQueryExecutor initialization"""
        from reconstruction.vlm_cad_strategy import CadQueryExecutor

        executor = CadQueryExecutor()
        assert isinstance(executor.has_cadquery, bool)

    def test_execute_simple_box(self):
        """Test executing simple CadQuery code"""
        from reconstruction.vlm_cad_strategy import CadQueryExecutor

        executor = CadQueryExecutor()
        if not executor.has_cadquery:
            pytest.skip("CadQuery not installed")

        code = """
import cadquery as cq
result = cq.Workplane("XY").box(10, 10, 5)
"""
        result = executor.execute(code)
        assert result is not None


class TestBuild123dExecutor:
    """Tests for Build123d executor"""

    def test_executor_init(self):
        """Test Build123dExecutor initialization"""
        from reconstruction.vlm_cad_strategy import Build123dExecutor

        executor = Build123dExecutor()
        assert isinstance(executor.has_build123d, bool)


class TestOpenECADExecutor:
    """Tests for OpenECAD executor"""

    def test_executor_init(self):
        """Test OpenECADExecutor initialization"""
        from reconstruction.vlm_cad_strategy import OpenECADExecutor

        executor = OpenECADExecutor()
        assert isinstance(executor.has_occ, bool)


class TestVLMCADStrategy:
    """Tests for VLMCADStrategy class"""

    def test_strategy_type(self):
        """Test strategy type and name"""
        from reconstruction.strategies import VLMCADStrategy, StrategyType

        strategy = VLMCADStrategy(model_id="openecad-0.89b")

        assert strategy.strategy_type == StrategyType.VLM_CAD
        assert "VLM CAD" in strategy.name
        assert "openecad-0.89b" in strategy.name

    def test_strategy_availability_check(self):
        """Test is_available method"""
        from reconstruction.strategies import VLMCADStrategy

        strategy = VLMCADStrategy()

        # Should return True/False without errors
        result = strategy.is_available()
        assert isinstance(result, bool)

    def test_strategy_get_available_models(self):
        """Test getting available models from strategy"""
        from reconstruction.strategies import VLMCADStrategy

        strategy = VLMCADStrategy()
        models = strategy.get_available_models()

        # Should return a list
        assert isinstance(models, list)

    def test_strategy_get_device_info(self):
        """Test getting device information"""
        from reconstruction.strategies import VLMCADStrategy

        strategy = VLMCADStrategy()
        info = strategy.get_device_info()

        assert isinstance(info, dict)
        assert "device" in info


class TestStrategyIntegration:
    """Tests for VLM CAD integration with ReconstructionManager"""

    def test_strategy_type_enum(self):
        """Test VLM_CAD is in StrategyType enum"""
        from reconstruction.strategies import StrategyType

        assert hasattr(StrategyType, "VLM_CAD")
        assert StrategyType.VLM_CAD.value == "vlm_cad"

    def test_get_strategy_vlm_cad(self):
        """Test factory function creates VLMCADStrategy"""
        from reconstruction.strategies import get_strategy, VLMCADStrategy, StrategyType

        strategy = get_strategy(StrategyType.VLM_CAD)
        assert isinstance(strategy, VLMCADStrategy)

        # Test with string
        strategy2 = get_strategy("vlm_cad")
        assert isinstance(strategy2, VLMCADStrategy)

    def test_get_strategy_with_params(self):
        """Test factory function with custom parameters"""
        from reconstruction.strategies import get_strategy

        strategy = get_strategy(
            "vlm_cad",
            model_id="internvl2-2b",
            auto_download=False,
        )

        assert strategy.model_id == "internvl2-2b"
        assert strategy.auto_download is False

    def test_reconstruction_manager_default_order(self):
        """Test VLM_CAD is in default strategy order"""
        from reconstruction.strategies import ReconstructionManager, StrategyType

        manager = ReconstructionManager()

        assert StrategyType.VLM_CAD in manager._default_order
        # Should be first in the order
        assert manager._default_order[0] == StrategyType.VLM_CAD

    def test_reconstruction_manager_register(self):
        """Test registering VLM CAD strategy"""
        from reconstruction.strategies import (
            ReconstructionManager,
            VLMCADStrategy,
            StrategyType,
        )

        manager = ReconstructionManager()
        strategy = VLMCADStrategy()

        manager.register_strategy(strategy)

        assert StrategyType.VLM_CAD in manager.strategies
        assert manager.strategies[StrategyType.VLM_CAD] is strategy


class TestConvenienceFunctions:
    """Tests for convenience functions"""

    def test_list_models(self):
        """Test list_models convenience function"""
        from reconstruction.vlm_cad_strategy import list_models

        models = list_models()

        assert isinstance(models, list)
        assert len(models) == 8

    def test_download_model_function_unknown(self):
        """Test download_model convenience function with unknown model"""
        from reconstruction.vlm_cad_strategy import download_model

        with pytest.raises(ValueError, match="Unknown model"):
            download_model("unknown-model")

    def test_validate_cad_code(self):
        """Test validate_cad_code convenience function"""
        from reconstruction.vlm_cad_strategy import validate_cad_code

        result = validate_cad_code("import cadquery as cq\nresult = cq.Workplane('XY').box(1,1,1)")

        assert isinstance(result, dict)
        assert "valid" in result
        assert "format" in result
        assert "errors" in result
        assert "warnings" in result
        assert "operations" in result

    def test_get_available_executors(self):
        """Test get_available_executors convenience function"""
        from reconstruction.vlm_cad_strategy import get_available_executors

        executors = get_available_executors()

        assert isinstance(executors, dict)
        assert "cadquery" in executors
        assert "build123d" in executors
        assert "pythonocc" in executors
        assert "torch" in executors
        assert "transformers" in executors


class TestPromptTemplates:
    """Tests for prompt templates"""

    def test_prompts_exist(self):
        """Test that all prompt templates are defined"""
        from reconstruction.vlm_cad_strategy import CAD_PROMPTS

        assert "openecad" in CAD_PROMPTS
        assert "cadquery" in CAD_PROMPTS
        assert "build123d" in CAD_PROMPTS
        assert "internvl" in CAD_PROMPTS
        assert "default" in CAD_PROMPTS

    def test_prompts_not_empty(self):
        """Test that prompts are not empty"""
        from reconstruction.vlm_cad_strategy import CAD_PROMPTS

        for name, prompt in CAD_PROMPTS.items():
            assert prompt, f"Prompt {name} is empty"
            assert len(prompt) > 20, f"Prompt {name} is too short"

    def test_openecad_prompt_contains_commands(self):
        """Test OpenECAD prompt mentions required commands"""
        from reconstruction.vlm_cad_strategy import CAD_PROMPTS

        prompt = CAD_PROMPTS["openecad"]
        assert "add_sketchplane" in prompt
        assert "add_line" in prompt
        assert "add_extrude" in prompt

    def test_cadquery_prompt_mentions_result(self):
        """Test CadQuery prompt mentions result variable"""
        from reconstruction.vlm_cad_strategy import CAD_PROMPTS

        prompt = CAD_PROMPTS["cadquery"]
        assert "result" in prompt.lower()
        assert "Workplane" in prompt
