"""Unit tests for configuration management."""

import unittest
import os
import json
import tempfile
from synctalk.config import (
    ModelConfig, AudioConfig, VideoConfig, ProcessingConfig,
    SyncTalkConfig, get_default_config, apply_preset
)


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig dataclass."""
    
    def test_default_paths(self):
        """Test that default paths are set correctly."""
        config = ModelConfig(name="TestModel")
        
        self.assertEqual(config.name, "TestModel")
        self.assertEqual(config.checkpoint_path, "./checkpoint/TestModel")
        self.assertEqual(config.dataset_path, "./dataset/TestModel")
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.audio_mode, "ave")
    
    def test_custom_paths(self):
        """Test custom path configuration."""
        config = ModelConfig(
            name="TestModel",
            checkpoint_path="/custom/checkpoint",
            dataset_path="/custom/dataset",
            device="cpu"
        )
        
        self.assertEqual(config.checkpoint_path, "/custom/checkpoint")
        self.assertEqual(config.dataset_path, "/custom/dataset")
        self.assertEqual(config.device, "cpu")


class TestSyncTalkConfig(unittest.TestCase):
    """Test main configuration container."""
    
    def test_creation(self):
        """Test configuration creation."""
        config = SyncTalkConfig(
            model=ModelConfig(name="TestModel"),
            audio=AudioConfig(),
            video=VideoConfig(),
            processing=ProcessingConfig()
        )
        
        self.assertEqual(config.model.name, "TestModel")
        self.assertEqual(config.audio.sample_rate, 16000)
        self.assertEqual(config.video.fps, 25)
        self.assertTrue(config.processing.cleanup_temp_files)
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "model": {
                "name": "TestModel",
                "device": "cpu"
            },
            "audio": {
                "vad_threshold": 0.6
            },
            "video": {
                "fps": 30
            }
        }
        
        config = SyncTalkConfig.from_dict(config_dict)
        
        self.assertEqual(config.model.name, "TestModel")
        self.assertEqual(config.model.device, "cpu")
        self.assertEqual(config.audio.vad_threshold, 0.6)
        self.assertEqual(config.video.fps, 30)
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = get_default_config("TestModel")
        config_dict = config.to_dict()
        
        self.assertIn("model", config_dict)
        self.assertIn("audio", config_dict)
        self.assertIn("video", config_dict)
        self.assertIn("processing", config_dict)
        self.assertEqual(config_dict["model"]["name"], "TestModel")
    
    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        config = get_default_config("TestModel")
        config.audio.vad_threshold = 0.7
        config.video.fps = 30
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.to_json(f.name)
            temp_path = f.name
        
        try:
            # Load back
            loaded_config = SyncTalkConfig.from_json(temp_path)
            
            self.assertEqual(loaded_config.model.name, "TestModel")
            self.assertEqual(loaded_config.audio.vad_threshold, 0.7)
            self.assertEqual(loaded_config.video.fps, 30)
        finally:
            os.unlink(temp_path)
    
    def test_environment_variables(self):
        """Test updating config from environment variables."""
        # Set test environment variables
        os.environ["SYNCTALK_MODEL_NAME"] = "EnvModel"
        os.environ["SYNCTALK_DEVICE"] = "cpu"
        os.environ["SYNCTALK_VAD_THRESHOLD"] = "0.8"
        os.environ["SYNCTALK_FPS"] = "30"
        
        try:
            config = get_default_config("TestModel")
            config.update_from_env()
            
            self.assertEqual(config.model.name, "EnvModel")
            self.assertEqual(config.model.device, "cpu")
            self.assertEqual(config.audio.vad_threshold, 0.8)
            self.assertEqual(config.video.fps, 30)
        finally:
            # Clean up environment
            for key in ["SYNCTALK_MODEL_NAME", "SYNCTALK_DEVICE", 
                       "SYNCTALK_VAD_THRESHOLD", "SYNCTALK_FPS"]:
                if key in os.environ:
                    del os.environ[key]


class TestPresets(unittest.TestCase):
    """Test configuration presets."""
    
    def test_high_quality_preset(self):
        """Test high quality preset."""
        config = get_default_config("TestModel")
        apply_preset(config, "high_quality")
        
        self.assertEqual(config.video.crf, 18)
        self.assertEqual(config.video.face_crop_size, 512)
        self.assertEqual(config.video.face_output_size, 480)
        self.assertEqual(config.audio.batch_size, 32)
    
    def test_fast_preset(self):
        """Test fast preset."""
        config = get_default_config("TestModel")
        apply_preset(config, "fast")
        
        self.assertEqual(config.video.crf, 23)
        self.assertEqual(config.video.fps, 20)
        self.assertEqual(config.audio.batch_size, 128)
        self.assertEqual(config.audio.vad_threshold, 0.6)
    
    def test_low_memory_preset(self):
        """Test low memory preset."""
        config = get_default_config("TestModel")
        apply_preset(config, "low_memory")
        
        self.assertEqual(config.model.device, "cpu")
        self.assertEqual(config.audio.batch_size, 16)
        self.assertFalse(config.processing.enable_caching)
    
    def test_invalid_preset(self):
        """Test applying invalid preset."""
        config = get_default_config("TestModel")
        
        with self.assertRaises(ValueError):
            apply_preset(config, "invalid_preset")


if __name__ == '__main__':
    unittest.main()