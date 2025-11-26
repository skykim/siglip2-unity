# siglip2-unity

On-device Multi-modal Retrieval: SigLIP2 in Unity Sentis

## Overview

This repository provides a lightweight inference implementation optimized for **SigLIP2** (specifically `siglip2-base-patch16-224`) using **Unity Sentis**. This project enables powerful multi-modal retrieval tasks directly within Unity, allowing for smart asset search and semantic retrieval across text and images without requiring an internet connection.

## Features

- ✅ Text-to-Image Search
- ✅ Image-to-Image Search
- ✅ Text-to-Text Search
- ✅ Image-to-Text Search

## Requirements

- **Unity**: `6000.2.10f1`
- **Unity Sentis**: `2.4.1` (com.unity.ai.inference)

## Architecture

### 1. SigLIP2 (ONNX)

The project utilizes the ONNX version of the [siglip2-base-patch16-224](https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX) model. It requires both the text encoder and the vision encoder to compute embeddings for multi-modal tasks.

### 2. Tokenizer

Text input processing is handled by the **Google SentencePieceTokenizer**, implemented using the `Microsoft.ML.Tokenizers` library. This ensures that text queries are correctly tokenized and encoded to match the SigLIP2 model's expected input format.

### 3. Embedding Database

The system generates a local database (`image_embeddings.bin`) by processing images located in the StreamingAssets folder. This allows for real-time similarity search during runtime.

## Getting Started

### 1. Model Setup

- Download `text_model.onnx` and `vision_model.onnx` from [onnx-community/siglip2-base-patch16-224-ONNX](https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX)
- Place both files into the `/Assets/SigLIP2` directory in your project

### 2. Assets Setup

- Clone or download this repository
- Unzip the provided [StreamingAssets.zip](https://drive.google.com/file/d/13tnfjXqXM_uT12dPYuq4UhHOawbOwwvg/view?usp=sharing) file
  - *Note: The demo images included in this zip are sourced from the [Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dataset.*
- Place the unzipped contents into the `/Assets/StreamingAssets` directory
- *Ensure that your .jpg or .png files are located inside `/Assets/StreamingAssets/Images`*

### 3. Generate Database

- Open the `/Assets/Scenes/SigLip2Scene.unity` scene in the Unity Editor
- Select the `ImageSearchManager` object in the hierarchy
- Click the **"Generate And Save Embeddings (Create Index)"** button in the Inspector
- This process will read images from StreamingAssets and generate the `image_embeddings.bin` file

### 4. Run the Demo Scene

- Play the scene to see the retrieval in action
- Input keywords to perform image retrieval, or explore other tasks like Image-to-Image or Text-to-Text search

## Demo

Experience SigLIP2 in Unity in action! Check out our demo showcasing the retrieval capabilities:

[![SigLIP2 Unity Demo](https://img.youtube.com/vi/Ae5-d-SADwk/0.jpg)](https://www.youtube.com/watch?v=Ae5-d-SADwk)

## Links

- [Google SigLIP2](https://arxiv.org/abs/2502.14786)
- [Onnx Community: SigLIP2-Base-Patch16-224](https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX)
- [Dataset: Fashion Product Images (Small)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
- [Unity Sentis Documentation](https://docs.unity3d.com/Packages/com.unity.ai.inference@latest)

## License

This project uses the SigLIP2 model which is licensed under the Apache 2.0 License.
