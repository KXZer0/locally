"""QA harness for evaluating a VLM through locally's OpenAI API.

Runs a battery of synthetic-image tests plus any real images dropped into
./test-images/. Records timings, full responses, and auto-grades the
synthetic cases via keyword checks. Outputs a Markdown report to stdout.

Usage:
    python qa_vlm.py [--url http://localhost:8765] [--model gpu-model]
"""
import argparse
import base64
import io
import json
import os
import time
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def red_circle():
    img = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(img).ellipse((128, 128, 384, 384), fill="red", outline="black", width=4)
    return img


def two_circles():
    img = Image.new("RGB", (512, 512), "white")
    d = ImageDraw.Draw(img)
    d.ellipse((40, 180, 240, 380), fill="red", outline="black", width=3)
    d.ellipse((272, 180, 472, 380), fill="blue", outline="black", width=3)
    return img


def text_image():
    img = Image.new("RGB", (640, 200), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 96)
    except OSError:
        font = ImageFont.load_default()
    d.text((40, 30), "HELLO 42", fill="black", font=font)
    return img


def empty_image():
    return Image.new("RGB", (512, 512), "white")


def color_grid():
    img = Image.new("RGB", (512, 512), "white")
    d = ImageDraw.Draw(img)
    d.rectangle((0,   0,   256, 256), fill="red")
    d.rectangle((256, 0,   512, 256), fill="green")
    d.rectangle((0,   256, 256, 512), fill="blue")
    d.rectangle((256, 256, 512, 512), fill="yellow")
    return img


def post_chat(url, model, prompt, image_b64=None, max_tokens=120):
    msg_content = [{"type": "text", "text": prompt}]
    if image_b64:
        msg_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        })
    body = {
        "model": model,
        "stream": False,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": msg_content}],
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.loads(r.read())
    elapsed = time.perf_counter() - t0
    text = data["choices"][0]["message"]["content"]
    return text, elapsed


def keyword_check(response, must_have, must_not_have=()):
    rlow = response.lower()
    missing = [k for k in must_have if k.lower() not in rlow]
    forbidden = [k for k in must_not_have if k.lower() in rlow]
    if not missing and not forbidden:
        return "PASS", None
    notes = []
    if missing: notes.append(f"missing: {missing}")
    if forbidden: notes.append(f"unexpected: {forbidden}")
    return "FAIL", "; ".join(notes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8765")
    p.add_argument("--model", default="gpu-model")
    p.add_argument("--images-dir", default="test-images")
    args = p.parse_args()

    endpoint = args.url.rstrip("/") + "/v1/chat/completions"
    results = []

    print(f"# Qwen3-VL 8B QA report")
    print(f"\nEndpoint: `{endpoint}`")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ---------- Synthetic image tests (auto-graded) ----------
    synth = [
        ("Red circle",       red_circle(),
         "What shape and color is in this image? One short sentence.",
         ["red", "circle"], ()),
        ("Two circles",      two_circles(),
         "How many circles do you see, and what colors are they?",
         ["red", "blue"], ()),
        ("Rendered text",    text_image(),
         "What text appears in this image? Output only the text.",
         ["hello", "42"], ()),
        ("Empty (no objects)", empty_image(),
         "What objects do you see in this image?",
         [], ["circle", "person", "car", "dog", "cat", "building"]),
        ("4-color grid",     color_grid(),
         "How many distinct colors do you see, and which?",
         ["red", "green", "blue", "yellow"], ()),
    ]

    print("## Synthetic tests\n")
    print("| # | Test | Time | Result | Notes |")
    print("|---|---|---|---|---|")
    for i, (name, img, prompt, must, mustnot) in enumerate(synth, 1):
        try:
            resp, dt = post_chat(endpoint, args.model, prompt, png_b64(img))
            verdict, note = keyword_check(resp, must, mustnot)
            results.append({"name": name, "resp": resp, "dt": dt,
                            "verdict": verdict, "note": note})
            print(f"| {i} | {name} | {dt:.2f}s | {verdict} | {note or ''} |")
        except Exception as e:
            results.append({"name": name, "resp": None, "dt": None,
                            "verdict": "ERROR", "note": str(e)[:80]})
            print(f"| {i} | {name} | - | ERROR | {str(e)[:80]} |")

    print("\n### Synthetic responses (full text)\n")
    for r in results:
        if r["resp"] is not None:
            print(f"**{r['name']}** — {r['dt']:.2f}s — *{r['verdict']}*  ")
            print(f"> {r['resp'].strip()}\n")

    # ---------- Text-only sanity ----------
    print("\n## Text-only sanity (no image)\n")
    text_cases = [
        ("Math",      "What is 17 * 23? Just the number."),
        ("Reasoning", "If today is Tuesday, what day was it 10 days ago?"),
    ]
    print("| Test | Time | Response |")
    print("|---|---|---|")
    for name, prompt in text_cases:
        try:
            resp, dt = post_chat(endpoint, args.model, prompt, image_b64=None,
                                 max_tokens=60)
            results.append({"name": f"text:{name}", "resp": resp, "dt": dt,
                            "verdict": "INFO", "note": ""})
            short = resp.strip().replace("\n", " ")[:120]
            print(f"| {name} | {dt:.2f}s | {short} |")
        except Exception as e:
            print(f"| {name} | - | ERROR: {str(e)[:80]} |")

    # ---------- Real images (if provided) ----------
    images_dir = Path(args.images_dir)
    if images_dir.is_dir():
        real = sorted([p for p in images_dir.iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")])
    else:
        real = []

    if real:
        print(f"\n## Real images from `{images_dir}/` ({len(real)} files)\n")
        for path in real:
            print(f"### {path.name}\n")
            try:
                img = Image.open(path).convert("RGB")
                # Cap dimensions to keep prefill reasonable
                if max(img.size) > 1024:
                    img.thumbnail((1024, 1024))
                resp, dt = post_chat(
                    endpoint, args.model,
                    "Describe this image in 1-2 sentences. Note any text visible.",
                    png_b64(img), max_tokens=160,
                )
                results.append({"name": f"real:{path.name}", "resp": resp,
                                "dt": dt, "verdict": "INFO", "note": ""})
                print(f"*{dt:.2f}s* — {resp.strip()}\n")
            except Exception as e:
                print(f"ERROR: {str(e)[:200]}\n")
    else:
        print(f"\n## Real images\n\n(no images in `{images_dir}/` — synthetic-only run)\n")

    # ---------- Summary ----------
    passes = sum(1 for r in results if r.get("verdict") == "PASS")
    fails  = sum(1 for r in results if r.get("verdict") == "FAIL")
    errors = sum(1 for r in results if r.get("verdict") == "ERROR")
    info   = sum(1 for r in results if r.get("verdict") == "INFO")
    times  = [r["dt"] for r in results if r.get("dt") is not None]
    avg_t  = sum(times) / len(times) if times else 0

    print("\n## Summary\n")
    print(f"- **PASS:**  {passes}")
    print(f"- **FAIL:**  {fails}")
    print(f"- **ERROR:** {errors}")
    print(f"- **INFO:**  {info}")
    print(f"- Total tests timed: {len(times)}, avg {avg_t:.2f}s")


if __name__ == "__main__":
    main()
