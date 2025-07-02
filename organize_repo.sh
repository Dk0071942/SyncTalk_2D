#!/bin/bash
# Script to reorganize the SyncTalk_2D repository

echo "Starting repository reorganization..."

# Create new directory structure
echo "Creating new directory structure..."
mkdir -p models/{unet,syncnet,audio_encoder}
mkdir -p scripts/{training,inference,utils}
mkdir -p docs
mkdir -p tests
mkdir -p configs
mkdir -p apps

# Move model files
echo "Moving model files..."
mv unet*.py models/unet/ 2>/dev/null || true
mv syncnet*.py models/syncnet/ 2>/dev/null || true

# Move training scripts
echo "Moving training scripts..."
mv train*.py scripts/training/ 2>/dev/null || true
mv training*.sh scripts/training/ 2>/dev/null || true

# Move inference scripts
echo "Moving inference scripts..."
mv inference*.py scripts/inference/ 2>/dev/null || true
mv batch_inference.sh scripts/inference/ 2>/dev/null || true

# Move dataset files
echo "Moving dataset files..."
mv datasetsss*.py scripts/utils/ 2>/dev/null || true
mv utils.py scripts/utils/ 2>/dev/null || true

# Move app files
echo "Moving app files..."
mv app_gradio.py apps/ 2>/dev/null || true

# Move test files
echo "Moving test files..."
mv test_*.py tests/ 2>/dev/null || true

# Move documentation
echo "Moving documentation..."
mv ARCHITECTURE.md docs/ 2>/dev/null || true
mv README_gradio.md docs/ 2>/dev/null || true

# Create __init__.py files for Python packages
echo "Creating __init__.py files..."
touch models/__init__.py
touch models/unet/__init__.py
touch models/syncnet/__init__.py
touch models/audio_encoder/__init__.py
touch scripts/__init__.py
touch scripts/training/__init__.py
touch scripts/inference/__init__.py
touch scripts/utils/__init__.py

# Create a new main inference script
cat > inference.py << 'EOF'
#!/usr/bin/env python3
"""Main inference script - redirects to the actual implementation"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.inference.inference_328 import main

if __name__ == "__main__":
    main()
EOF

# Create a new main training script
cat > train.py << 'EOF'
#!/usr/bin/env python3
"""Main training script - redirects to the actual implementation"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.training.train_328 import main

if __name__ == "__main__":
    main()
EOF

# Create a new app launcher
cat > app.py << 'EOF'
#!/usr/bin/env python3
"""Launch the Gradio application"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apps.app_gradio import create_interface

if __name__ == "__main__":
    demo = create_interface()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
EOF

# Create a requirements.txt that combines all dependencies
cat > requirements.txt << 'EOF'
# Core dependencies
torch>=2.0.0
torchvision
torchaudio
opencv-python
numpy==1.23.5
tqdm

# Audio processing
soundfile
librosa==0.10.1

# Model components
transformers
onnxruntime-gpu
configargparse

# Web interface
gradio>=4.0.0
EOF

# Create a comprehensive README
cat > README_NEW.md << 'EOF'
# SyncTalk 2D - High-Quality Lip-Sync Video Generation

A 2D lip-sync video generation model that creates high-quality lip-synchronized videos with low latency. Based on SyncTalk and Ultralight-Digital-Human with significant improvements.

## Features

- 📹 **High Resolution**: Supports 328x328 resolution (vs 256x256 in original)
- 🎵 **Advanced Audio Processing**: Enhanced audio feature encoder
- 🚀 **Low Latency**: Optimized for real-time generation
- 🌐 **Web Interface**: User-friendly Gradio interface
- 🔄 **Seamless Looping**: Videos can start and end with the same frame

## Project Structure

```
SyncTalk_2D/
├── models/              # Model architectures
│   ├── unet/           # U-Net model for video generation
│   ├── syncnet/        # SyncNet for audio-visual synchronization
│   └── audio_encoder/  # Audio feature extraction
├── scripts/            # Core scripts
│   ├── training/       # Training scripts
│   ├── inference/      # Inference scripts
│   └── utils/          # Utility functions
├── apps/               # Applications
│   └── app_gradio.py   # Web interface
├── data_utils/         # Data preprocessing tools
├── docs/               # Documentation
├── tests/              # Test scripts
├── checkpoint/         # Trained model checkpoints
├── dataset/            # Training data
└── demo/               # Demo audio files
```

## Quick Start

### Installation

```bash
# Create environment
conda create -n synctalk_2d python=3.10
conda activate synctalk_2d

# Install dependencies
pip install -r requirements.txt
```

### Web Interface

```bash
python app.py
```

Open http://localhost:7860 in your browser.

### Command Line

**Training:**
```bash
bash scripts/training/training_328.sh <name> <gpu_id>
```

**Inference:**
```bash
python inference.py --name <model_name> --audio_path <audio.wav>
```

## Training Your Own Model

1. Record a 5-minute video following the guidelines in docs/
2. Place video in `dataset/<name>/<name>.mp4`
3. Run training script
4. Wait ~5 hours for training to complete

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Gradio Interface Guide](docs/README_gradio.md)
- [Training Guidelines](docs/training_guide.md)

## Citation

If you use this code, please cite:
- Original SyncTalk paper
- Ultralight-Digital-Human repository
EOF

echo "Repository reorganization complete!"
echo "Please review the changes and update import statements in the Python files."