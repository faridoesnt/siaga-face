import base64
import os
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import insightface


class EnrollRequest(BaseModel):
    user_id: str
    images: List[str]


class EnrollEmbedding(BaseModel):
    vector: str
    model: str


class EnrollResponse(BaseModel):
    success: bool
    embeddings: List[EnrollEmbedding]


class VerifyRequest(BaseModel):
    user_id: str
    image: Optional[str] = None
    image_base64: Optional[str] = None
    stored_embeddings: List[str]


class VerifyResponse(BaseModel):
    matched: bool
    score: float


app = FastAPI(title="SIAGA Face Service")

_face_app: Optional[insightface.app.FaceAnalysis] = None

@app.get("/health")
def health():
    return "ok"

def get_face_app() -> insightface.app.FaceAnalysis:
    global _face_app
    if _face_app is None:
        providers = ["CPUExecutionProvider"]
        fa = insightface.app.FaceAnalysis(providers=providers)
        # Default det_size is fine for mobile-sized images
        fa.prepare(ctx_id=0)
        _face_app = fa
    return _face_app


def _decode_base64_image(data: str) -> np.ndarray:
    if data.startswith("data:"):
        # data URL, split comma
        _, b64 = data.split(",", 1)
    else:
        b64 = data
    try:
        raw = base64.b64decode(b64, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid base64 image")
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="unable to decode image")
    return img


def _extract_single_embedding(img: np.ndarray) -> Optional[np.ndarray]:
    fa = get_face_app()
    faces = fa.get(img)
    if len(faces) != 1:
        return None
    face = faces[0]
    # use normalized embedding
    emb = face.normed_embedding
    return emb.astype("float32")


def _embedding_to_string(emb: np.ndarray) -> str:
    # Store as base64 of raw float32 bytes
    return base64.b64encode(emb.tobytes()).decode("ascii")


def _string_to_embedding(s: str) -> np.ndarray:
    raw = base64.b64decode(s)
    return np.frombuffer(raw, dtype="float32")


@app.post("/enroll", response_model=EnrollResponse)
def enroll(req: EnrollRequest) -> EnrollResponse:
    if not req.images:
        raise HTTPException(status_code=400, detail="images is required")

    vectors: List[EnrollEmbedding] = []
    model_name = "arcface"

    for img_str in req.images:
        img = _decode_base64_image(img_str)
        emb = _extract_single_embedding(img)
        # Skip images where we can't find exactly one face
        if emb is None:
            continue
        vec_str = _embedding_to_string(emb)
        vectors.append(EnrollEmbedding(vector=vec_str, model=model_name))

    if not vectors:
        raise HTTPException(status_code=400, detail="no valid face detected in images")

    return EnrollResponse(success=True, embeddings=vectors)


@app.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest) -> VerifyResponse:
    if not req.stored_embeddings:
        # No enrolled embeddings; treat as not matched
        return VerifyResponse(matched=False, score=0.0)

    img_str = req.image or req.image_base64
    if not img_str:
        raise HTTPException(status_code=400, detail="image is required")

    img = _decode_base64_image(img_str)
    emb = _extract_single_embedding(img)
    if emb is None:
        # no or multiple faces
        return VerifyResponse(matched=False, score=0.0)

    # Cosine similarity with stored embeddings
    best_score = -1.0
    for s in req.stored_embeddings:
        stored = _string_to_embedding(s)
        # normalize
        stored_norm = stored / (np.linalg.norm(stored) + 1e-6)
        cur = float(np.dot(emb, stored_norm) / (np.linalg.norm(emb) + 1e-6))
        if cur > best_score:
            best_score = cur

    threshold = float(os.getenv("FACE_MATCH_THRESHOLD", "0.6"))
    matched = best_score >= threshold

    if best_score < 0:
        best_score = 0.0

    return VerifyResponse(matched=matched, score=best_score)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)

