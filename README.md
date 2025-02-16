# EasyUI
### V1.1.2-2.1

**TextToImage** is a free, open-source text-to-image generation tool designed for ease of use, allowing anyone to run advanced models on their computer with customizable parameters and progress tracking.

> **NOTE:** The version available via `git clone` is more up-to-date than the releases because the repository contains the latest changes, which may not yet be included in the official release.

## Features

- **Progress Tracking**
- **Seed Control**
- **Adjustable CFG** `(Classifier-Free Guidance)`
- **Support for multiple schedulers**
- **ControlNet support** `(currently canny/depth is fully supported | normal map is only SD)`
- **Image-to-Image and Text-to-Image generation**

## Installation Guide

### Prerequisites

- **A good GPU** `(it will run slowly if you don't have one)`
- **CUDA** `(to use GPU)`
- **VS Code** `(recommended editor)`

### Steps

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/umittadelen/stableDiffusionEasyUI.git
    cd stableDiffusionEasyUI
    ```

2. **Change the IP**:<br>
    This step is not that important but the starting `ip:port` is `localhost:8080`
    <br>It is possible to change this from settings page
3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Running the Application

1. **Start the Flask Server**:
    ```bash
    python app.py
    ```
   ---
2. **Access the Application**:<br>
    Open your web browser and navigate to `http://localhost:8080`.
---
### Generating Images

0. **Download a Model**:<br>
    - Click to `Model Editor` button and download model using the civitai model id and version id (you can get the ids from `AIR:   123456 @ 654321`)

   ---
1. **Choose a Model**:<br>
    - Select a model from the dropdown list or add a new model URL.
   
   ---
2. **Set Parameters**:
    - **Model Type**:<br>Choose the type of model (this step is automatically done if the model downloaded using the `model editor` page) `(e.g., SDXL, SD1.5, FLUX)`.

       ---
    - **Scheduler**:<br> Select the scheduler model. `(default is [Euler A])`

       ---
    - **Prompt**:<br>Enter the main prompt that describes the image you want to generate.<br>`prevent using short prompts (it can make NSFW images)`<br>(Multiple images can be generated using the multi prompt feature like `"prompt1§prompt2"  >  "prompt1","prompt2"`)
    <br>
    [[Click for a quick guide on writing better prompts]](https://umittadelen.github.io/better_prompting/)


       ---
    - **Negative Prompt**:<br>Enter negative prompts to specify what you want to avoid in the generated image.

       ---
    - **Width and Height**:<br>Set the dimensions of the generated image.

       ---
    - **CFG Scale**:<br>Adjust the scale value to control how closely the AI follows your prompts.

       ---
    - **Sampling Steps**:<br>Set the number of sampling steps. `(higher number means more steps | 30 is recommended)`

       ---
    - **Number of Images**:<br>Specify the number of images to generate.

       ---
    - **Custom Seed**:<br>If set to `-1` generates variated images.
---
3. **Generate Images**:
    Click the "Generate Images" button to start the image generation process.

### Additional Features

- **Stop Generation**: Click the "Stop Generation" button to stop the image generation process.
- **Clear Images**: Click the "Clear Images" button to clear all generated images.
- **Preview ControlNet**: Click the "Preview ControlNet" button to go to the page for previewing the ControlNet image.
- **Get Metadata**: Click the "Get Metadata" button to extract metadata from an uploaded image.
- **Model Editor**: Click the "Model Editor" button to view and edit the JSON configuration of models.

## License

This project is licensed under the MIT License. See the [LICENSE.txt](https://github.com/umittadelen/easyUI/blob/main/LICENSE.txt) file for details.

---

### More About Me
[www.umittadelen.net](https://umittadelen.net)<br>