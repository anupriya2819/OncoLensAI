"""OncoLens: Grad-CAM and a local upload page for the existing Phikon-v2 API.

Place this file beside the working server.py, outside the models folder.
Stop the old server with Ctrl+C, then run in the activated environment:
    python -m uvicorn gradcam_app:app --host 127.0.0.1 --port 8000
Open http://127.0.0.1:8000/ in your browser.

Uses the existing server.py model, processor, classifier, scaler, encoder and
lock. All model files stay local. No further model download or training.
Existing /predict and /health routes remain available; /analyze adds Grad-CAM.

This is a local research prototype. A model score is not measured accuracy.
Grad-CAM is a coarse attribution, not tumor segmentation or a biomarker test.
Public hosting, authentication and clinical validation are separate work.

Method references:
https://huggingface.co/owkin/phikon-v2
https://jacobgil.github.io/pytorch-gradcam-book/vision_transformers.html

Validation: Python and JavaScript syntax checked. Tested saved classifier
logit parity and gradients on 20 float32/float64 feature vectors. Tested
Grad-CAM forward/backward, unchanged prediction, both class targets, hook
cleanup, empty-map overlay and API uploads with a small random DINOv2
backbone and the uploaded classifier. The alternate-class test modified
only an in-memory test copy of the bias. No test model is shipped.
Local test versions: torch 2.6.0+cpu, transformers 4.57.6,
scikit-learn 1.6.1 and FastAPI 0.115.14.
Real Phikon-v2 weights, real-image accuracy, Windows execution and browser
rendering were not tested in this workspace. Verify the same image locally
against the existing /predict response before using the heatmap.
"""

import base64
from io import BytesIO
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image, UnidentifiedImageError
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# Import the exact backend that has already returned a successful prediction.
from server import app, classifier, encoder, lock, model, processor, scaler

logger = logging.getLogger("uvicorn.error")
MAX_UPLOAD = 10 * 1024 * 1024
MAX_PIXELS = 12_000_000


class RequestSizeLimit:
    """Cap uploads before FastAPI's multipart parser reads the file."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        limit = MAX_UPLOAD + 65536  # multipart headers and boundaries
        chunks = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > limit:
                response = JSONResponse(
                    {"detail": "Upload one PNG or JPEG under 10 MB."},
                    status_code=413,
                )
                return await response(scope, receive, send)
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        chunks.clear()
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


app.add_middleware(RequestSizeLimit)
# Hackathon integration mode: allow the browser frontend to call this API through Ngrok.
# Do not use wildcard CORS for a real clinical deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def validate_components():
    if getattr(model.config, "model_type", "") != "dinov2":
        raise RuntimeError("This Grad-CAM module requires the Phikon-v2 DINOv2 backbone.")
    features = model.config.hidden_size
    if classifier.coef_.shape != (1, features) or classifier.intercept_.shape != (1,):
        raise RuntimeError("Expected a fitted binary linear classifier matching the backbone.")
    if scaler.n_features_in_ != features or not scaler.with_mean or not scaler.with_std:
        raise RuntimeError("Expected the matching fitted StandardScaler with mean and scale.")
    labels = [str(x) for x in encoder.inverse_transform(classifier.classes_)]
    if len(labels) != 2 or set(labels) != {"benign", "malignant"}:
        raise RuntimeError("The classifier and encoder must contain benign and malignant labels.")
    for values in (classifier.coef_, classifier.intercept_, scaler.mean_, scaler.scale_):
        if not np.isfinite(values).all():
            raise RuntimeError("Non-finite parameters in the saved classifier or scaler.")
    if not (scaler.scale_ > 0).all():
        raise RuntimeError("Scaler contains an invalid scale.")
    return labels


LABELS = validate_components()
model.eval()
model.requires_grad_(False)
DEVICE = next(model.parameters()).device
# Grad-CAM must target a layer before the final attention block: the final
# output patch tokens do not affect a classifier that uses only the CLS token.
TARGET_LAYER = model.encoder.layer[-1].norm1
MEAN = torch.as_tensor(scaler.mean_, dtype=torch.float64, device=DEVICE)
SCALE = torch.as_tensor(scaler.scale_, dtype=torch.float64, device=DEVICE)
WEIGHT = torch.as_tensor(classifier.coef_, dtype=torch.float64, device=DEVICE)
BIAS = torch.as_tensor(classifier.intercept_, dtype=torch.float64, device=DEVICE)


def class_logit(embedding):
    # StandardScaler.transform preserves float32 inputs and rounds each
    # in-place operation separately. Preserve these roundings for parity.
    dtype = embedding.dtype
    centered = (embedding.double() - MEAN).to(dtype)
    standardized = (centered.double() / SCALE).to(dtype)
    return (standardized.double() @ WEIGHT.T + BIAS).reshape(())


def decode_image(data):
    if not data or len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Upload one PNG or JPEG under 10 MB.")
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in {"PNG", "JPEG"}:
                raise HTTPException(422, "Choose one PNG or JPEG image, not a ZIP archive.")
            if getattr(source, "n_frames", 1) != 1:
                raise HTTPException(422, "Choose a single-frame image.")
            if source.width * source.height > MAX_PIXELS or min(source.size) < 32:
                raise HTTPException(422, "Use an image at least 32 pixels per side and at most 12 megapixels.")
            # Matches the existing backend and the notebook's RGB conversion.
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(422, "Cannot read this image. Choose a PNG or JPEG from your dataset.")


def png_url(image):
    output = BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def model_view(pixel_values):
    # Invert normalization on the actual tensor, retaining the processor's
    # resize and crop. Thus the overlay is aligned with what the model saw.
    pixels = pixel_values.detach()[0].cpu().float()
    if getattr(processor, "do_normalize", False):
        mean = torch.as_tensor(processor.image_mean).reshape(-1, 1, 1)
        std = torch.as_tensor(processor.image_std).reshape(-1, 1, 1)
        pixels = pixels * std + mean
    if getattr(processor, "do_rescale", False):
        pixels = pixels / processor.rescale_factor
    array = pixels.permute(1, 2, 0).clamp(0, 255).round().numpy().astype(np.uint8)
    return Image.fromarray(array)


def overlay_image(image, heatmap):
    resized = Image.fromarray(heatmap.astype(np.float32)).resize(image.size, Image.Resampling.BILINEAR)
    strength = np.clip(np.asarray(resized), 0, 1)
    colors = colormaps["inferno"](strength)[..., :3] * 255
    alpha = 0.50 * strength[..., None]
    overlay = np.asarray(image).astype(np.float32) * (1 - alpha) + colors * alpha
    return Image.fromarray(np.clip(overlay, 0, 255).round().astype(np.uint8))


def analyze(image):
    with lock, torch.inference_mode(False), torch.enable_grad():
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        captured = {}

        def capture(module, args, output):
            # Frozen earlier layers need no graph. Begin gradient tracking at
            # the attribution layer; values and the prediction are unchanged.
            activation = output.detach().requires_grad_(True)
            captured["activation"] = activation
            return activation

        handle = TARGET_LAYER.register_forward_hook(capture)
        try:
            embedding = model(**inputs).last_hidden_state[:, 0, :]
            logit = class_logit(embedding)
            vector = embedding.detach().cpu().numpy()
            reference_features = scaler.transform(vector)
            probabilities = classifier.predict_proba(reference_features)[0]
            reference_logit = float(classifier.decision_function(reference_features)[0])
            if not np.isfinite(probabilities).all() or not np.isfinite(reference_logit):
                raise RuntimeError("Classifier returned non-finite values.")
            if not np.isclose(logit.detach().item(), reference_logit, atol=1e-6, rtol=1e-6):
                raise RuntimeError("The explanation calculation does not match the saved classifier.")
            best = int(probabilities.argmax())
            score = logit if best == 1 else -logit
            activation = captured["activation"]
            gradients = torch.autograd.grad(score, activation)[0]
            patch_size = model.config.patch_size
            if isinstance(patch_size, (list, tuple)):
                ph, pw = patch_size
            else:
                ph = pw = patch_size
            height = inputs["pixel_values"].shape[-2] // ph
            width = inputs["pixel_values"].shape[-1] // pw
            patches = activation.detach()[0, 1:, :]
            patch_gradients = gradients.detach()[0, 1:, :]
            if len(patches) != height * width:
                raise RuntimeError("Unexpected patch layout; cannot align the heatmap.")
            if not torch.isfinite(patches).all() or not torch.isfinite(patch_gradients).all():
                raise RuntimeError("Non-finite Grad-CAM activations or gradients.")
            weights = patch_gradients.mean(dim=0)
            cam = torch.relu((patches * weights).sum(dim=-1)).reshape(height, width)
            peak = cam.max().item()
            if not np.isfinite(peak):
                raise RuntimeError("Non-finite Grad-CAM heatmap.")
            if peak > 0:
                cam = cam / peak
                status = "available"
            else:
                status = "no_positive_attribution"
            heatmap = cam.cpu().numpy()
            viewed_image = model_view(inputs["pixel_values"])
        finally:
            handle.remove()
            captured.clear()

    overlay = overlay_image(viewed_image, heatmap)
    return {
        "prediction": LABELS[best],
        "confidence": float(probabilities[best]),
        "probabilities": {label: float(p) for label, p in zip(LABELS, probabilities)},
        "originalImage": png_url(viewed_image),
        "heatmapImage": png_url(overlay),
        "gradcam": {
            "status": status,
            "target": LABELS[best],
            "layer": "encoder.layer[-1].norm1",
            "grid": [int(height), int(width)],
            "imageSpace": "model_input",
            "classifierParityPassed": True,
        },
        "biomarkers": [],
        "biomarkerStatus": "not_assessed",
        "note": "Research use only. Scores are uncalibrated. Grad-CAM is not a tumor boundary.",
    }


@app.post("/analyze")
def analyze_upload(file: UploadFile = File(...)):
    """Classify one image and return Grad-CAM for the predicted class."""
    try:
        data = file.file.read(MAX_UPLOAD + 1)
    finally:
        file.file.close()
    image = decode_image(data)
    try:
        return analyze(image)
    except Exception:
        logger.exception("OncoLens image analysis failed")
        raise HTTPException(500, "Analysis failed. Check the final error in the Command Prompt window.")


def _read_upload(upload: UploadFile) -> bytes:
    try:
        data = upload.file.read(MAX_UPLOAD + 1)
    finally:
        upload.file.close()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Upload an image smaller than 10 MB.")
    return data


@app.post("/v1/analysis/quality")
def frontend_quality(image: UploadFile = File(...)):
    """Technical input checks for the frontend quality step.

    These checks verify that the file is readable and large enough for the model input.
    They do NOT assess pathology adequacy, staining quality, focus, or diagnostic quality.
    """
    data = _read_upload(image)
    decoded = decode_image(data)
    width, height = decoded.size
    min_side = min(width, height)
    resolution_score = min(100, round((min_side / 224.0) * 100))
    metrics = [
        {"label": "Readable image", "value": 100, "unit": "%", "ok": True},
        {"label": "Supported format", "value": 100, "unit": "%", "ok": True},
        {
            "label": "Resolution suitability",
            "value": resolution_score,
            "unit": "%",
            "ok": min_side >= 224,
        },
    ]
    return {
        "metrics": metrics,
        "passed": all(item["ok"] for item in metrics),
        "note": "Technical file checks only; not a diagnostic slide-quality assessment.",
    }


@app.post("/v1/analysis")
def frontend_analysis(
    image: UploadFile = File(...),
    sampleId: str = Form(""),
    patientId: str = Form(""),
    clinicalHistory: str = Form(""),
    priority: str = Form("Routine"),
    doctor: str = Form(""),
):
    """Compatibility route for the teammate React frontend."""
    started = time.perf_counter()
    data = _read_upload(image)
    decoded = decode_image(data)
    try:
        raw = analyze(decoded)
    except Exception:
        logger.exception("OncoLens frontend analysis failed")
        raise HTTPException(500, "Analysis failed. Check the Command Prompt window.")

    predicted = str(raw["prediction"]).lower()
    malignant = predicted == "malignant"
    model_score_pct = round(float(raw["confidence"]) * 100.0, 3)
    grad_status = raw.get("gradcam", {}).get("status", "unknown")
    findings = [
        f"Phikon-v2 feature extraction with the fitted downstream classifier suggested {predicted}.",
        (
            "Grad-CAM attribution was generated for the predicted class."
            if grad_status == "available"
            else "No positive Grad-CAM attribution was found at the configured layer."
        ),
        "HER2, Ki-67 and ER/PR status are not assessed by this model.",
    ]
    summary = (
        f"Research prototype result for sample {sampleId or 'unspecified'}: AI-suggested {predicted} "
        f"with an uncalibrated model score of {model_score_pct:.3f}%. "
        "The displayed risk band mirrors the binary research classification and is not a clinical risk estimate. "
        "Grad-CAM shows relative model influence, not tumor boundaries. Biomarker status is not assessed."
    )
    return {
        "diagnosis": f"AI-suggested {predicted.capitalize()}",
        "confidence": model_score_pct,
        "risk": "High" if malignant else "Low",
        "findings": findings,
        "biomarkers": [],
        "rois": [],
        "heatmapUrl": raw.get("heatmapImage"),
        "summary": summary,
        "processingSeconds": round(time.perf_counter() - started, 2),
        "completedAt": datetime.now(timezone.utc).isoformat(),
        "model": "Phikon-v2 + fitted downstream linear classifier",
        "prototype": True,
        "sampleId": sampleId,
        "patientId": patientId,
        "priority": priority,
        "doctor": doctor,
        "note": raw.get("note"),
    }


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>OncoLens | Image analysis</title>
<style>
:root{font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#17312f;background:#f4f7f6;color-scheme:light}
*{box-sizing:border-box}body{margin:0}main{max-width:1040px;margin:0 auto;padding:38px 24px}
header{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:42px}
.brand{font-size:23px;font-weight:750;letter-spacing:-.8px}.badge{font-size:12px;background:#e2ece9;padding:8px 12px;border-radius:24px}
h1{font-size:clamp(28px,5vw,42px);line-height:1.15;letter-spacing:-1.1px;margin:0 0 16px}
p{line-height:1.6}.muted{color:#566d68}.panel{background:white;border:1px solid #dbe6e2;border-radius:18px;padding:26px;margin-top:24px}
form{display:flex;align-items:end;gap:20px;flex-wrap:wrap}label{display:block;font-weight:650;margin-bottom:10px}
input{max-width:100%;font:inherit}button,.download{background:#116755;color:white;border:0;border-radius:9px;padding:13px 20px;font:inherit;font-weight:650;cursor:pointer;text-decoration:none;display:inline-block}
button:disabled{opacity:.55;cursor:wait}button:focus-visible,input:focus-visible,a:focus-visible{outline:3px solid #efa852;outline-offset:3px}
#status{min-height:24px;margin-bottom:0}#status.error{color:#a12929}.result-top{display:flex;justify-content:space-between;align-items:start;gap:20px;flex-wrap:wrap}
h2{margin:0 0 8px;font-size:28px}h3{font-size:16px;margin:0 0 12px}.score{font-size:25px;font-weight:700}.caption{font-size:13px;color:#566d68}
.images{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin:26px 0}.images img{display:block;width:100%;aspect-ratio:1;object-fit:contain;border-radius:10px;background:#f1f4f2}
.legend{display:flex;gap:12px;align-items:center;font-size:12px;color:#566d68}.scale{height:10px;width:130px;background:linear-gradient(90deg,#060519,#72206e,#e85c2f,#f9fca4);border-radius:5px}
.detail{border-top:1px solid #e3eae7;padding-top:16px}.download{font-size:14px;background:#edf5f1;color:#145b4c;border:1px solid #c6dfd5}
footer{font-size:13px;color:#647870;margin-top:32px;line-height:1.6}[hidden]{display:none!important}
@media(max-width:580px){main{padding:24px 16px}.images{grid-template-columns:1fr}.panel{padding:20px}header{margin-bottom:30px}}
</style></head><body><main>
<header><span class="brand">OncoLens</span><span class="badge">Research prototype</span></header>
<h1>Explore a pathology image</h1>
<p class="muted">Upload a breast H&amp;E histopathology image to view its classification and the regions influencing the model.</p>
<section class="panel"><form id="upload-form">
<div style="flex:1;min-width:210px"><label for="image-file">Choose one image</label><input id="image-file" type="file" accept="image/png,image/jpeg" required><p class="caption">PNG or JPEG &middot; up to 10 MB &middot; no ZIP files</p></div>
<button id="submit" type="submit">Analyze image</button></form>
<p id="status" role="status" aria-live="polite">Ready for an image.</p></section>
<section id="result" class="panel" hidden>
<div class="result-top"><div><p class="caption">Model prediction</p><h2 id="prediction"></h2><span class="caption" id="filename"></span></div>
<div><p class="caption">Score for the predicted class</p><div id="score" class="score"></div><span class="caption">Uncalibrated model score</span></div></div>
<div class="images"><div><h3>Image seen by the model</h3><img id="original" alt="The resized and cropped model input"></div>
<div><h3 id="heatmap-title">Grad-CAM overlay</h3><img id="heatmap" alt="Grad-CAM overlay on the model input"></div></div>
<div class="legend"><span>Lower influence</span><span class="scale"></span><span>Higher influence</span></div>
<p id="explanation" class="muted"></p>
<p class="caption detail">Both views use the same resized and cropped input. The map highlights relative influence on this prediction; it does not mark tumor boundaries. HER2 and Ki-67 are not assessed.</p>
<a id="download" class="download" download="oncolens-gradcam.png">Download overlay</a>
</section>
<footer>For research and educational use. One prediction does not measure model accuracy. Use public research images for this prototype.</footer>
</main><script>
const form=document.getElementById('upload-form'), picker=document.getElementById('image-file');
const status=document.getElementById('status'), result=document.getElementById('result'), button=document.getElementById('submit');
picker.addEventListener('change',()=>{result.hidden=true;status.className='';status.textContent='Ready to analyze.';});
form.addEventListener('submit',async(event)=>{
 event.preventDefault(); const file=picker.files[0]; if(!file)return;
 result.hidden=true;status.className='';
 if(!/\.(png|jpe?g)$/i.test(file.name)){status.className='error';status.textContent='Choose one PNG or JPEG image. Extract ZIP files first.';return;}
 if(file.size>10*1024*1024){status.className='error';status.textContent='Choose an image smaller than 10 MB.';return;}
 button.disabled=true;picker.disabled=true;status.textContent='Analyzing the image and calculating Grad-CAM. CPU processing may take a while.';
 try{
  const body=new FormData();body.append('file',file);
  const response=await fetch('/analyze',{method:'POST',body});
  const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'The server could not process this image.');
  if(!Number.isFinite(data.confidence)||!data.gradcam)throw new Error('The response is incomplete.');
  document.getElementById('prediction').textContent=data.prediction.charAt(0).toUpperCase()+data.prediction.slice(1);
  document.getElementById('score').textContent=(data.confidence*100).toFixed(3)+'%';
  document.getElementById('filename').textContent=file.name;
  document.getElementById('original').src=data.originalImage;document.getElementById('heatmap').src=data.heatmapImage;
  document.getElementById('download').href=data.heatmapImage;
  document.getElementById('heatmap-title').textContent='Grad-CAM for '+data.gradcam.target;
  document.getElementById('explanation').textContent=data.gradcam.status==='available'?
   'Brighter highlighted regions have greater positive influence on the predicted class in this map.':
   'No positive attribution was found at this layer for the predicted class. The image is shown without a highlight.';
  result.hidden=false;status.textContent='Analysis complete.';
 }catch(error){status.className='error';status.textContent=error.message;}
 finally{button.disabled=false;picker.disabled=false;}
});
</script></body></html>'''


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def upload_page():
    return HTMLResponse(PAGE, headers={"Cache-Control": "no-store"})


print("OncoLens Grad-CAM page: http://127.0.0.1:8000/", flush=True)
