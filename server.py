"""Standalone CAPTCHA demo server."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from captcha import generate_captcha, verify_captcha, warm_up_jit


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Warming up CAPTCHA JIT...")
    warm_up_jit()
    yield

app = FastAPI(title="Motion CAPTCHA Demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class VerifyRequest(BaseModel):
    captcha_id: str
    text: str
    honeypot: str = ""
    with_hints: bool = False


class GenerateRequest(BaseModel):
    mode: str = "hardened"

@app.post("/api/captcha/generate-all")
async def generate_captcha_endpoint(request: GenerateRequest):
    mode = request.mode
    if mode not in ("static", "moving", "hardened"):
        mode = "hardened"
    captcha_id, captcha_text, metadata, all_frames_packed = generate_captcha(mode=mode)

    import json as _json
    headers = {
        "X-Frame-Width": str(metadata["width"]),
        "X-Frame-Height": str(metadata["height"]),
        "X-Total-Frame-Count": str(metadata["total_frame_count"]),
        "X-Frame-Packed-Size": str(metadata["frame_packed_size"]),
        "X-Captcha-Id": captcha_id,
        "X-Frame-Interval": str(metadata["frame_interval"]),
        "X-Start-Frame": str(metadata.get("start_frame", 0)),
        "X-Captcha-Params": _json.dumps(metadata.get("params", {})),
        "Access-Control-Expose-Headers": "*",
    }

    return Response(
        content=all_frames_packed,
        media_type="application/octet-stream",
        headers=headers,
    )


@app.post("/api/captcha/verify")
async def verify_captcha_endpoint(request: VerifyRequest):
    if request.honeypot:
        return {"success": False, "error": "Verification failed"}

    result = verify_captcha(request.captcha_id, request.text, with_hints=request.with_hints)
    response = {
        "success": result["success"],
        "message": result.get("message"),
        "error": result.get("error"),
        "attempts_remaining": result.get("attempts_remaining"),
    }
    if "hints" in result:
        response["hints"] = result["hints"]
    return response


# Serve static files (index.html, captcha.js)
static_dir = Path(__file__).parent
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
