"""Persistent LanPaint + Qwen-Image-Edit-2509 inpaint/outpaint worker.

Runs under the isolated ``lanpaint-qwen`` venv (LANPAINT_QWEN_PY): loads Qwen ONCE and
serves JSON-line requests over stdin/stdout (same protocol as sam3_worker). Replaces
GPT-Image-2 for occlusion removal (masked inpaint) and border completion (native
outpaint). LanPaint gives clean mask-region regeneration where GPT-Image-2 hallucinated
a new object in the hole.

Protocol (one JSON object per line on stdin; one per line on stdout):
  {"op":"inpaint","image":<png>,"mask":<png white=keep/black=edit>,"prompt":str,
   "neg":str,"out":<png>,"guidance":4.0,"steps":20,"seed":0}
  {"op":"outpaint","image":<png>,"pad":"l192","prompt":str,"neg":str,"out":<png>,...}
  {"cmd":"shutdown"}
Response: {"ok":true,"out":<png>,"width":W,"height":H} | {"ok":false,"error":str}

Model prints (hub load, tqdm) go to stderr; only JSON goes to the real stdout.
"""

import json
import sys


def main() -> None:  # pragma: no cover - subprocess entry
    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # model / tqdm output must not corrupt the JSON pipe

    from lanpaint_pipeline import LanPaintConfig, LanPaintInpaintPipeline
    from lanpaint_pipeline.registry import create_adapter

    adapter = create_adapter("qwen", device="cpu", model_id=None)
    adapter.pipe.enable_sequential_cpu_offload()
    lp = LanPaintInpaintPipeline(
        adapter,
        config=LanPaintConfig(
            n_steps=1, friction=15.0, chara_lambda=16.0, beta=1.0,
            step_size=0.2, early_stop=1, cfg_big=4.0, blend_overlap=9,
        ),
    )

    def emit(obj: dict) -> None:
        print(json.dumps(obj), file=real_stdout, flush=True)

    emit({"ready": True})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        if req.get("cmd") == "shutdown":
            break
        try:
            common = dict(
                prompt=req["prompt"],
                image=req["image"],
                negative_prompt=req.get("neg", ""),
                guidance_scale=req.get("guidance", 4.0),
                num_inference_steps=req.get("steps", 20),
                seed=req.get("seed", 0),
            )
            op = req["op"]
            if op == "inpaint":
                res = lp(mask_image=req["mask"], **common)
            elif op == "outpaint":
                res = lp(outpaint_padding=req["pad"], **common)
            else:
                raise ValueError(f"unknown op {op!r}")
            img = res.images[0]
            img.save(req["out"])
            emit({"ok": True, "out": req["out"], "width": img.width, "height": img.height})
        except Exception as exc:  # noqa: BLE001 - report over the protocol, keep serving
            emit({"ok": False, "error": str(exc)[:300]})


if __name__ == "__main__":
    main()
