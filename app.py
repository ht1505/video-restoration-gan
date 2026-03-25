from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import torch, cv2, os, uuid, numpy as np
from models import UNetGenerator  # adjust import to match your actual class name

app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load trained generator weights at startup
generator = UNetGenerator()  # adjust args if needed
weights_path = os.environ.get("WEIGHTS_PATH", "saved_models/generator.pth")
if os.path.exists(weights_path):
    generator.load_state_dict(torch.load(weights_path, map_location=DEVICE))
generator.eval().to(DEVICE)

@app.post("/restore")
async def restore_video(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    in_path = f"/tmp/{uid}_input.avi"
    out_path = f"/tmp/{uid}_output.avi"

    with open(in_path, "wb") as f:
        f.write(await file.read())

    # Run inference (reuse logic from inference.py)
    os.system(
        f"python inference.py --degraded_video {in_path} "
        f"--weights_path {weights_path} --output_dir /tmp --output_name {uid}_output"
    )

    return FileResponse(out_path, media_type="video/x-msvideo", filename="restored.avi")

@app.get("/")
def health():
    return {"status": "ok"}
