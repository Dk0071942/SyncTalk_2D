# Gradio Interface for SyncTalk_2D

A web-based interface for generating lip-synced videos using trained SyncTalk_2D models.

## Features

- **Model Selection**: Choose from all available trained models
- **Audio Input**: Upload WAV audio files or record directly
- **Real-time Progress**: Track video generation progress
- **Advanced Options**: Configure start frame and audio encoder
- **Video Preview**: Watch generated videos directly in the browser
- **Easy Download**: Save generated videos locally

## Installation

1. Install Gradio dependencies:
```bash
pip install -r requirements_gradio.txt
```

## Usage

1. Start the Gradio interface:
```bash
python app_gradio.py
```

2. Open your browser to `http://localhost:7860`

3. Follow these steps:
   - Select a trained model from the dropdown
   - Upload an audio file (WAV format)
   - (Optional) Adjust advanced settings
   - Click "Generate Video"
   - Wait for processing to complete
   - Preview and download your video

## Interface Overview

### Main Controls
- **Model Selection**: Dropdown menu with all available models
- **Refresh Models**: Update the model list without restarting
- **Audio Upload**: Drag-and-drop or click to upload audio files
- **Generate Button**: Start the video generation process

### Advanced Options
- **Start Frame**: Choose which frame to start from (default: 0)
- **Audio Encoder**: Select between AVE, Hubert, or Wenet (default: AVE)

### Output
- **Video Player**: Preview generated videos with playback controls
- **Status Messages**: Real-time feedback on generation progress
- **Download**: Right-click to save generated videos

## Tips

1. **Audio Quality**: Use clear speech audio without background noise
2. **Audio Format**: WAV files work best
3. **Processing Time**: Generation takes 1-3 minutes depending on audio length
4. **Model Selection**: Choose models trained on similar speaking styles
5. **Output Location**: Videos are saved to `gradio_outputs/` folder

## Troubleshooting

1. **No models found**: Ensure you have trained models in the `checkpoint/` directory
2. **CUDA errors**: Check GPU availability and CUDA installation
3. **Audio errors**: Verify audio file is in WAV format
4. **Memory issues**: Try shorter audio clips or restart the app

## Remote Access

To allow remote access, modify the launch parameters:
```python
demo.launch(share=True)  # Creates a public URL
```

Or specify a custom host/port:
```python
demo.launch(server_name="0.0.0.0", server_port=8080)
```