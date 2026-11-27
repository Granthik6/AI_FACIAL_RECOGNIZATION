"""Optional local-only experiment on a consented image, with human judgment."""
import argparse
import csv
import time
from pathlib import Path
import psutil
from PIL import Image
from prepare_fer2013 import RESULTS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",required=True,type=Path,help="Local SmolVLM-256M-Instruct directory")
    p.add_argument("--image",required=True,type=Path)
    args = p.parse_args()
    size = sum(f.stat().st_size for f in args.model.glob("*.safetensors"))
    if not size or psutil.virtual_memory().available < 3*size:
        p.error("Local weights missing or insufficient available RAM for this CPU trial.")
    from transformers import AutoProcessor, AutoModelForVision2Seq
    import torch
    processor = AutoProcessor.from_pretrained(args.model,local_files_only=True)
    model = AutoModelForVision2Seq.from_pretrained(args.model,local_files_only=True,
                                                  torch_dtype=torch.float32).eval()
    messages = [dict(role="user",content=[dict(type="image"),dict(type="text",
                text="Describe only visible eyebrow and mouth movements briefly. If unclear say uncertain.")])]
    prompt = processor.apply_chat_template(messages,add_generation_prompt=True)
    start = time.perf_counter()
    with Image.open(args.image) as im:
        batch = processor(text=prompt,images=[im.convert("RGB")],return_tensors="pt")
    with torch.inference_mode():
        tokens = model.generate(**batch,max_new_tokens=32,do_sample=False)
    elapsed = time.perf_counter()-start
    answer = processor.decode(tokens[0,batch.input_ids.shape[1]:],skip_special_tokens=True)
    print(answer)
    correct = input("Human reviewer: correct, incorrect or uncertain? ")
    unsupported = input("Which details are unsupported by the image? ")
    path = RESULTS/"vlm_reviews.csv"
    fresh = not path.exists()
    with path.open("a",newline="",encoding="utf8") as file:
        writer = csv.writer(file)
        if fresh:
            writer.writerow(["description","review","unsupported","latency_s","rss_bytes"])
        writer.writerow([answer,correct,unsupported,elapsed,psutil.Process().memory_info().rss])


if __name__=="__main__":
    main()
