# Animal Detection System

This project is an Animal Detection System that uses a trained ResNet50 model to identify various animals from video input. The system sends notifications when it detects dangerous animals to alert users.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

The Animal Detection System processes video input to detect animals using a pretrained ResNet50 model. It identifies animals and sends notifications if dangerous animals are detected. The notifications are sent via Pushover, and data is also sent to ThingSpeak for logging purposes.

## Requirements

- Python 3.8 or higher
- OpenCV
- TensorFlow
- Keras
- Requests
- A pre-trained ResNet50 model (`ResNet50_DEL.h5`)

## Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/animal-detection-system.git
    cd animal-detection-system
    ```

2. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download the pre-trained model:**

    Place the `ResNet50_DEL.h5` file in the root directory of the project.

4. **Configure Pushover and ThingSpeak:**

    Replace the placeholders in the script with your actual API keys and user keys.

    ```python
    pushover_user_key = 'your_pushover_user_key'
    pushover_api_token = 'your_pushover_api_token'
    api_key = 'your_thingspeak_write_api_key'
    ```

## Usage

1. **Run the script:**

    ```bash
    python smart.py
    ```

2. **Test with a video file:**

    Update the `video_path` variable in the script with the path to your video file.

    ```python
    video_path = r'path_to_your_video_file.mp4'
    ```

    The system will process the video, detect animals, and send notifications if dangerous animals are detected.

## Project Structure

```plaintext
.
├── ResNet50_DEL.h5                # Pre-trained model file
├── smart.py                       # Main script for animal detection
├── requirements.txt               # List of required packages
└── README.md                      # Project README file
