# Gradio Interface for SyncTalk_2D

A comprehensive web-based interface for generating lip-synced videos using trained SyncTalk_2D models.

## Features

- **Multiple Generation Modes**: Three distinct modes for different use cases
- **Model Selection**: Choose from all available trained models with auto-discovery
- **Audio Input**: Upload various audio formats or use demo files
- **Quality Presets**: Pre-configured settings for different requirements
- **Real-time Progress**: Track video generation progress with detailed status
- **Advanced Options**: Fine-tune generation parameters for optimal results
- **Video Preview**: Watch generated videos directly in the browser
- **Easy Download**: Save generated videos with descriptive filenames

## Installation

1. Install dependencies (included in main requirements.txt):
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Gradio interface:
```bash
python app_gradio.py
```

2. Open your browser to `http://localhost:7860`

3. Follow the 5-step workflow in the interface

## Generation Modes

### 📊 Training Data Mode
- Uses frames from the original training dataset
- Fastest and most reliable option
- Best quality with consistent results
- **Settings**:
  - Start Frame: Choose which frame to start from (0-indexed)
  - Loop Back: Enable to reverse at midpoint for seamless loops

### 📹 Custom Video Mode
- Process your own video to match a specific person or style
- Requires a video with clear frontal face views
- More flexible but slower than training data mode
- **Requirements**:
  - Clear frontal face throughout the video
  - Good lighting and minimal movement
  - No occlusions or extreme angles

### 🎞️ Core Clips Mode
- Intelligently selects from pre-recorded clips using Voice Activity Detection
- Preserves natural pauses and breathing
- Most realistic for conversational content
- **Settings**:
  - VAD Threshold: Speech detection sensitivity (0.1-0.9, default 0.5)
  - Min Silence Duration: Minimum pause length (0.1-2.0 seconds)
  - Visualize VAD: Save speech/silence detection plot

## Interface Controls

### Step 1: Model Selection
- **Model Dropdown**: Auto-populated with available models
- **Refresh Button**: Update model list without restarting

### Step 2: Generation Mode
- Choose between Training Data, Custom Video, or Core Clips mode
- Each mode has specific settings and requirements

### Step 3: Mode Settings
- Configure mode-specific parameters
- See mode descriptions above for details

### Step 4: Audio Upload
- **Supported Formats**: WAV (recommended), MP3, and other common formats
- **Requirements**: Clear speech, minimal background noise
- **Demo Files**: Pre-loaded examples for testing

### Step 5: Advanced Settings
- **Audio Encoder**:
  - `ave`: Default, good balance
  - `hubert`: Better quality, multilingual
  - `wenet`: Chinese language support
- **Quality Preset**:
  - `default`: Balanced quality and speed
  - `high_quality`: Best results (slower)
  - `fast`: Quick preview generation
  - `low_memory`: For limited resources

## Output

- **Video Player**: Built-in preview with playback controls
- **Download**: Generated videos saved to `gradio_outputs/`
- **Filename Format**: `{model}_{audio}_{mode}_{timestamp}.mp4`
- **Progress Updates**: Real-time status messages during generation

## Performance Comparison

| Mode | Speed | Quality | Flexibility |
|------|-------|---------|-------------|
| Training Data | ⚡ Fast | ⭐⭐⭐ Excellent | 🔒 Limited |
| Custom Video | 🐌 Slow | ⭐⭐ Good | 🔓 High |
| Core Clips | ⚡ Fast | ⭐⭐⭐ Excellent | 🔐 Medium |

## Tips for Best Results

### Audio Preparation
1. Use high-quality recordings (16kHz+ sample rate)
2. Remove background noise
3. Normalize audio levels
4. Use clear, articulate speech
5. WAV format recommended

### Video Requirements (Custom Mode)
1. Record with stable camera, no movement
2. Ensure good, even lighting
3. Face camera directly, avoid profile views
4. Keep consistent expression
5. 5+ minutes recommended for best results

### Performance Optimization
1. Shorter audio clips process faster
2. Use Training Data mode for quick results
3. High Quality preset needs more VRAM
4. Core Clips mode good for long audio

## Troubleshooting

### Common Issues

1. **No models found**
   - Ensure models exist in `checkpoint/` directory
   - Click refresh button after adding new models

2. **CUDA/GPU errors**
   - Check GPU availability: `nvidia-smi`
   - Verify CUDA installation
   - Try `low_memory` preset

3. **Audio processing errors**
   - Convert to WAV format
   - Check audio isn't corrupted
   - Ensure audio has speech content

4. **Memory issues**
   - Use shorter audio clips
   - Select `low_memory` preset
   - Close other applications
   - Restart the interface

5. **Poor lip-sync quality**
   - Use clearer audio
   - Try different audio encoder
   - Ensure model matches speaker style

## Advanced Configuration

### Remote Access
```python
# In app_gradio.py, modify the launch line:
demo.launch(share=True)  # Creates public URL
```

### Custom Host/Port
```python
demo.launch(server_name="0.0.0.0", server_port=8080)
```

### Increase Maximum File Size
```python
demo.launch(max_file_size=50)  # 50MB limit
```

## Examples

The interface includes pre-configured examples:
- **Training Data Mode**: Quick demo with default settings
- **Custom Video Mode**: Example with template video
- **Core Clips Mode**: VAD demonstration

Click "Load Example" to try these configurations.