from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from typing import List
import torch
import open_clip
from PIL import Image
import io
import time
import traceback # 用於印出詳細的錯誤資訊

# --- 1. (已移除) 日誌設定 ---
# 不需要 logging 模組了

app = FastAPI()

# --- 2. 模型與裝置設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

try:
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    print("[INFO] CLIP model 'ViT-B-32' loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    traceback.print_exc() # 印出完整的錯誤堆疊
    # raise e
# -------------------------------------------------


@app.post("/embed-image")
async def embed_image(file: UploadFile = File(...)):
    start_time = time.time()
    print(f"[INFO] Received request for /embed-image. Filename: '{file.filename}', Content-Type: {file.content_type}")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            vector = image_features[0].cpu().tolist()
        
        duration = time.time() - start_time
        print(f"[INFO] Successfully embedded image '{file.filename}'. Vector dim: {len(vector)}. Duration: {duration:.4f}s")
        
        return {"vector": vector, "dim": len(vector)}

    except Exception as e:
        duration = time.time() - start_time
        print(f"[ERROR] Error processing image '{file.filename}'. Duration: {duration:.4f}s. Error: {e}")
        traceback.print_exc() # 印出完整的錯誤堆疊
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")


@app.post("/embed-text")
async def embed_text(text: str = Form(...)):
    start_time = time.time()
    truncated_text = text[:80] + '...' if len(text) > 80 else text
    print(f"[INFO] Received request for /embed-text. Text: '{truncated_text}'")

    try:
        text_tokens = tokenizer([text]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            vector = text_features[0].cpu().tolist()
        
        duration = time.time() - start_time
        print(f"[INFO] Successfully embedded text. Vector dim: {len(vector)}. Duration: {duration:.4f}s")

        return {"text": text, "vector": vector, "dim": len(vector)}

    except Exception as e:
        duration = time.time() - start_time
        print(f"[ERROR] Error processing text: '{truncated_text}'. Duration: {duration:.4f}s. Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to process text: {e}")


@app.post("/batch-embed")
async def batch_embed(files: List[UploadFile] = File(...)):
    start_time = time.time()
    file_count = len(files)
    filenames = [f.filename for f in files]
    print(f"[INFO] Received request for /batch-embed with {file_count} files: {filenames}")
    
    try:
        images = []
        names = []

        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            images.append(preprocess(image))
            names.append(file.filename)

        batch_tensor = torch.stack(images).to(device)

        with torch.no_grad():
            features = model.encode_image(batch_tensor)

        result = []
        for i in range(len(names)):
            result.append({
                "filename": names[i],
                "vector": features[i].cpu().tolist()
            })
        
        duration = time.time() - start_time
        print(f"[INFO] Successfully batch-embedded {len(result)} files. Duration: {duration:.4f}s")

        return result

    except Exception as e:
        duration = time.time() - start_time
        print(f"[ERROR] Error during batch processing of {file_count} files. Error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed during batch processing: {e}")