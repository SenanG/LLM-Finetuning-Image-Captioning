# finetune_gpt4o_medical_captioning.py

import os
import json
import openai
from PIL import Image
from typing import List
from tqdm import tqdm

# GPT-4o configuration
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"

# Simulated Dataset Entry
class MedicalScan:
    def __init__(self, image_path: str, prompt: str, expected_caption: str = ""):
        self.image_path = image_path
        self.prompt = prompt
        self.expected_caption = expected_caption

# 🧠 Step 1: Load Dataset
def load_dataset(data_path: str) -> List[MedicalScan]:
    with open(data_path, "r") as f:
        entries = [json.loads(line) for line in f]
    return [MedicalScan(e["image_path"], e["prompt"], e.get("caption", "")) for e in entries]

# 🧠 Step 2: GPT-4o Prompt Call
def generate_caption(scan: MedicalScan, image_base64: str) -> str:
    try:
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a medical expert describing diagnostic scans."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": scan.prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Failed to generate caption: {e}")
        return ""

# 🧠 Step 3: Utility to Encode Image
import base64

def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# 🧠 Step 4: Run Evaluation Loop
def run_captioning(data_path: str, output_path: str):
    dataset = load_dataset(data_path)
    results = []
    
    for scan in tqdm(dataset):
        img_b64 = encode_image_base64(scan.image_path)
        caption = generate_caption(scan, img_b64)
        results.append({
            "image_path": scan.image_path,
            "prompt": scan.prompt,
            "generated_caption": caption,
            "expected_caption": scan.expected_caption
        })

    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} results to {output_path}")

# Run the full flow
if __name__ == "__main__":
    run_captioning("data/medical_dataset.jsonl", "results/generated_captions.jsonl")
