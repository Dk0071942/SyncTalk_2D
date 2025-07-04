# SyncTalk_2D Architecture and Workflow Documentation

## Overview

SyncTalk_2D is a 2D lip-sync video generation model that creates high-quality lip-synchronized videos with low latency. It supports 328x328 resolution and uses an enhanced audio feature encoder for commercial-grade digital human generation.

## Model Architecture

### 1. U-Net Architecture (unet_328.py)

The main model uses a modified U-Net architecture with MobileNet-inspired components:

**Key Components:**
- **InvertedResidual blocks**: Based on MobileNetV2 architecture, using depthwise separable convolutions with expansion factor (default=2 or 6)
- **Encoder path**: 5 downsampling stages using `Down` modules that halve spatial dimensions
- **Decoder path**: 5 upsampling stages using `Up` modules with skip connections
- **Audio fusion**: Audio features are concatenated at the bottleneck (lowest resolution)

**Architecture Details:**
- Input: 6-channel image (3 channels from reference face + 3 channels from masked current frame)
- Channel progression: [32, 64, 128, 256, 512] (or reduced version [16, 32, 64, 128, 256] for mobile)
- Final output: 3-channel RGB image (320x320) with sigmoid activation

### 2. Audio Feature Extraction

The system supports three different audio encoders:

**a) AVE (Audio-Visual Encoder)** - Default option:
- Uses mel-spectrogram features (80 mel bins)
- Multi-scale convolutional network with residual blocks
- Progressive downsampling: (1, 32) → (32, 32) → (64, 64) → (128, 128) → (256, 256) → (512, 512)
- Output: 512-dimensional features per frame
- Input window: 16 mel frames centered around current video frame
- Final audio feature shape: (32, 16, 16)

**b) Hubert**:
- Expects pre-extracted Hubert features
- Audio feature shape: (32, 32, 32)
- Uses different convolution parameters in AudioConvHubert

**c) Wenet**:
- Expects pre-extracted Wenet features  
- Audio feature shape: (256, 16, 32)
- Uses AudioConvWenet with modified stride patterns

### 3. Data Preprocessing Pipeline

The preprocessing (`data_utils/process.py`) involves:

1. **Video Processing**:
   - Convert video to 25 FPS if needed
   - Extract frames as individual images
   - Detect facial landmarks (68 points) using PFLD model
   - Store landmarks for each frame

2. **Audio Processing**:
   - Extract audio at 16kHz sample rate
   - Compute mel-spectrograms (for AVE)
   - Generate audio features using selected encoder
   - Save as numpy arrays

3. **Training Data Preparation**:
   - Crop face region based on landmarks (square crop around face)
   - Resize to 328x328, then crop to 320x320
   - Create masked version (black rectangle in mouth region)
   - Concatenate reference face and masked current frame

### 4. Loss Functions

The training uses three loss components:

1. **Pixel Loss (L1 Loss)**:
   - Direct L1 distance between predicted and ground truth images
   - Weight: 1.0

2. **Perceptual Loss**:
   - Uses VGG19 features (up to conv3_3 layer)
   - Compares feature representations between predicted and real images
   - Weight: 0.01

3. **Sync Loss** (optional):
   - Uses a separate SyncNet to ensure audio-visual synchronization
   - Computes cosine similarity between audio and visual embeddings
   - Binary cross-entropy loss on similarity scores
   - Weight: 10.0

**Total Loss** = `L1_loss + 0.01 * Perceptual_loss + 10 * Sync_loss`

### 5. SyncNet Architecture

The SyncNet ensures temporal alignment between audio and visual features:

- **Face Encoder**: 
  - Progressive downsampling from 320x320 to 512-dim embedding
  - Uses residual convolution blocks
  
- **Audio Encoder**:
  - Adapts to different audio feature formats (AVE/Hubert/Wenet)
  - Produces 512-dim embedding
  
- Both embeddings are L2-normalized and compared using cosine similarity

## Complete Workflow: From Audio to Video

### 1. Training Phase

**Step 1: Data Preparation**
```bash
python data_utils/process.py <video_path>
```
- Extracts frames at 25 FPS
- Detects 68 facial landmarks per frame
- Extracts audio and computes mel-spectrograms
- Generates AVE audio features (512-dim per frame)

**Step 2: SyncNet Training**
```bash
python syncnet_328.py --save_dir ./syncnet_ckpt/<name> --dataset_dir <data_dir> --asr ave
```
- Trains audio-visual synchronization network
- Ensures temporal alignment between lip movements and audio
- Saves checkpoints for use in main model training

**Step 3: Main Model Training**
```bash
python train_328.py --dataset_dir <data_dir> --save_dir ./checkpoint/<name> --asr ave --use_syncnet --syncnet_checkpoint <path>
```
- Trains U-Net model to generate lip-synced face regions
- Uses L1, perceptual, and sync losses
- Saves checkpoints every 25 epochs

### 2. Inference Phase (Video Generation)

**Step 1: Audio Processing**
- Load new audio file (WAV format)
- Convert to mel-spectrograms
- Extract AVE features using pre-trained encoder
- Shape: (num_frames, 512) → (num_frames, 32, 16, 16)

**Step 2: Frame Generation**
```python
for each audio frame:
    1. Select reference frame from training video
    2. Crop face region using landmarks
    3. Create masked version (black mouth area)
    4. Concatenate reference + masked → 6-channel input
    5. Pass through U-Net with audio features
    6. Get predicted face region (320x320)
    7. Resize and blend back into original frame
```

**Step 3: Video Assembly**
- Write frames at 25 FPS
- Merge with original audio using FFmpeg
- Output: MP4 file with lip-synced video

### 3. Key Technical Details

**Face Cropping Logic:**
- Uses landmarks 1 (left face) and 31 (right face) for width
- Uses landmark 52 (upper lip) as top reference
- Creates square crop (height = width)

**Audio-Visual Alignment:**
- 16-frame audio window centered on current frame
- Special boundary handling for end-of-audio:
  - When accessing frames beyond audio end, entire 16-frame window uses last frame's features
  - Prevents mouth movement artifacts during silence at video end
  - Ensures consistent neutral mouth position for final ~0.6 seconds
- Beginning frames use first frame features for padding
- Maintains temporal consistency throughout

**Performance Optimizations:**
- Batch processing for audio features
- GPU acceleration throughout
- Efficient video encoding with x264

### 4. Practical Usage

**Single Video Generation:**
```bash
python scripts/inference_cli.py --name <model_name> --audio_path <audio.wav>
```

**Batch Processing:**
```bash
./batch_inference.sh <avatar_name> <audio_directory>
```

## Key Design Choices

1. **Lightweight Architecture**: Uses depthwise separable convolutions for efficiency
2. **Multi-scale Processing**: U-Net skip connections preserve fine details
3. **Audio-Visual Fusion**: Late fusion at bottleneck preserves both modalities
4. **Flexible Audio Support**: Can work with different audio encoders
5. **Landmark-based Cropping**: Ensures consistent face alignment

## Performance

The system achieves real-time performance by:
- Pre-computing audio features
- Using lightweight U-Net architecture
- Efficient face region processing (only mouth area changes)
- Hardware acceleration via CUDA

This workflow enables creating realistic lip-synced videos from any audio input, using just 5 minutes of training video data.