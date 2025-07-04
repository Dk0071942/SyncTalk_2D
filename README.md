# SyncTalk 2D

SyncTalk_2D is a 2D lip-sync video generation model based on SyncTalk and [Ultralight-Digital-Human](https://github.com/anliyuan/Ultralight-Digital-Human). It can generate lip-sync videos with high quality and low latency, and it can also be used for real-time lip-sync video generation.

Compared to the Ultralight-Digital-Human, we have improved the audio feature encoder and increased the resolution to 328 to accommodate higher-resolution input video. This version can realize high-definition, commercial-grade digital humans.

与Ultralight-Digital-Human相比，我们改进了音频特征编码器，并将分辨率提升至328以适应更高分辨率的输入视频。该版本可实现高清、商业级数字人。

## Features

- 📹 **High Resolution**: Supports 328x328 resolution (vs 256x256 in original)
- 🎵 **Advanced Audio Processing**: Enhanced audio feature encoder with AVE support
- 🚀 **Low Latency**: Optimized for real-time generation
- 🌐 **Web Interface**: User-friendly Gradio interface for easy interaction
- 🔄 **Seamless Looping**: Videos can start and end with the same frame

## Setting up

Set up the environment:

### Windows
```bash
conda create -n synctalk_2d python=3.10
conda activate synctalk_2d
```

### Linux
```bash
conda create -n synctalk_2d python=3.10
conda activate synctalk_2d
```

Install dependencies:

### Windows
```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install FFmpeg (very important)
conda install -c conda-forge ffmpeg=4.3.* x264

# Install other dependencies
pip install -r requirements.txt
```

### Linux
```bash
# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install FFmpeg (very important)
conda install -c conda-forge 'ffmpeg=4.3.*' x264

# Install other dependencies
pip install -r requirements.txt
```

## Quick Start - Web Interface

Launch the Gradio interface:
```bash
python app_gradio.py
```

Then open http://localhost:7860 in your browser.

## Prepare your data

1. Record a 5-minute video with your head facing the camera and without significant movement. At the same time, ensure that the camera does not move and the background light remains unchanged during video recording.
2. Don't worry about FPS, the code will automatically convert the video to 25fps.
3. No second person's voice can appear in the recorded video, and a 5-second silent clip is left at the beginning and end of the video.
4. Don't wear clothes with overly obvious texture, it's better to wear single-color clothes.
5. The video should be recorded in a well-lit environment.
6. The audio should be clear and without background noise.

## Train

1. Put your video in the 'dataset/name/name.mp4' 
   - Example: dataset/May/May.mp4

2. Run the process and training script:

   ### Linux/macOS
   ```bash
   bash scripts/training_328.sh name gpu_id
   ```
   - Example: `bash scripts/training_328.sh May 0`
   
   ### Windows
   
   Option 1: Using the provided batch script (recommended)
   ```cmd
   scripts\training_328.bat name gpu_id
   ```
   - Example: `scripts\training_328.bat May 0`
   
   Option 2: Using Git Bash or WSL
   ```bash
   bash scripts/training_328.sh name gpu_id
   ```
   
   Option 3: Using Command Prompt/PowerShell (run commands separately)
   ```cmd
   set CUDA_VISIBLE_DEVICES=gpu_id
   python data_utils/process.py ./dataset/name/name.mp4
   python synctalk/core/syncnet_328.py --save_dir ./syncnet_ckpt/name --dataset_dir ./dataset/name --asr ave
   # Find the latest checkpoint file in ./syncnet_ckpt/name/*.pth
   python scripts/train_328.py --dataset_dir ./dataset/name --save_dir ./checkpoint/name --asr ave --use_syncnet --syncnet_checkpoint ./syncnet_ckpt/name/your_checkpoint.pth
   ```
   - Replace `name` with your dataset name and `gpu_id` with your GPU device ID (e.g., 0)
   - Replace `your_checkpoint.pth` with the actual checkpoint filename generated in the syncnet_ckpt folder
   
   **Note:** 
   - Training takes approximately 5 hours
   - If OOM (Out of Memory) occurs, try reducing the batch_size in the training script
   - Windows users: Ensure CUDA toolkit is properly installed and NVIDIA GPU drivers are up to date
   - Windows paths: Use forward slashes (/) or escaped backslashes (\\) in paths
   - **NEW**: Training now automatically resumes from interruptions using `.training_state.json`
   - **NEW**: SyncNet training is automatically skipped if already completed

## Inference

### Command Line
```bash
python scripts/inference_cli.py --name <model_name> --audio_path <audio.wav>
```

### Batch Processing
```bash
./batch_inference.sh <model_name> <audio_directory>
```

### Web Interface
Use the Gradio interface for an easier experience with model selection and real-time preview.

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) - Detailed model architecture and workflow
- [Gradio Interface Guide](docs/README_gradio.md) - How to use the web interface

## Project Structure

```
SyncTalk_2D/
├── app_gradio.py       # Web interface
├── run_synctalk.py     # Unified inference interface
├── synctalk/           # Core package
│   ├── core/           # Model implementations
│   └── utils/          # Utilities
├── scripts/            # Training and inference scripts
│   ├── train_328.py    # Training script (328x328)
│   ├── inference_cli.py # Command-line inference
│   └── preprocess_data.py # Data preprocessing
├── data_utils/         # Data preprocessing tools
├── checkpoint/         # Trained model checkpoints
├── dataset/            # Training data
├── demo/               # Demo audio files
├── docs/               # Documentation
└── gradio_outputs/     # Generated videos from web interface
```

## Available Models

Currently trained models in `checkpoint/`:
- **AD2.2** - Commercial avatar model
- **LS1** - Another trained avatar

## Tips

- For best results, use clear speech audio without background noise
- The model works best with the same language it was trained on
- Videos are generated at 25 FPS
- Use the "Loop back to start frame" option in Gradio for seamless loops

## Citation

If you use this code, please cite the original papers:
- SyncTalk paper
- Ultralight-Digital-Human repository
