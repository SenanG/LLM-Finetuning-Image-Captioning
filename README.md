# Medical Image Captioning with GPT-4o

This repository contains code for generating captions for medical images using OpenAI's GPT-4o model.

## Functionality

The `main.py` script performs the following steps:

1.  **Loads a dataset:** Reads medical scan image paths and associated prompts from a JSONL file.
2.  **Generates captions:** For each image, it encodes the image to base64 and sends it to the GPT-4o API along with the prompt to generate a descriptive caption.
3.  **Saves results:** The generated captions, along with the original image paths, prompts, and any expected captions, are saved to an output JSONL file.

## Usage

1.  Set your `OPENAI_API_KEY` environment variable.
2.  Prepare your dataset in JSONL format (see `data/medical_dataset.jsonl` for an example structure). Each line should be a JSON object containing `image_path` and `prompt`. Optionally, include `caption` for expected captions.
3.  Run the script: `python main.py`
    *   The script will use `data/medical_dataset.jsonl` as input and save results to `results/generated_captions.jsonl` by default. You can modify these paths in the `if __name__ == "__main__":` block in `main.py`.
