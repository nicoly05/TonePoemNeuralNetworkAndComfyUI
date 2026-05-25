from flask import Flask, request, jsonify
import requests, json, time, uuid, traceback

app = Flask(__name__)
COMFY_URL = "http://127.0.0.1:8188"

NOTE_PROMPTS = {
    "C": "deep indigo slow waves",
    "D": "violet medium oscillations",
    "E": "blue calm sine waves",
    "F": "teal flowing energy",
    "G": "cyan bright waves",
    "A": "light blue electric pulses",
    "B": "white-blue intense vibrations",
}

def notes_to_prompt(notes):
    if not notes:
        return "sound wave visualization, blue waves, dark background"
    freqs = [n["freq"] for n in notes]
    avg_freq = sum(freqs) / len(freqs)
    count = len(notes)
    unique_notes = list(set(n["note"][0] for n in notes))
    wave_descriptions = [NOTE_PROMPTS.get(n, "cyan waves") for n in unique_notes[:3]]
    wave_str = ", ".join(wave_descriptions)
    if avg_freq > 470:
        intensity = "high frequency intense bright electric"
    elif avg_freq > 370:
        intensity = "medium frequency flowing calm"
    else:
        intensity = "low frequency deep slow pulsing"
    prompt = (
        f"sound wave visualization, {wave_str}, "
        f"{intensity}, {count} overlapping sine waves, "
        f"bioluminescent glow, dark navy background, "
        f"photorealistic, 8k, no text, abstract"
    )
    print(f"[PROMPT] {prompt}")
    return prompt

def build_workflow(prompt_text):
    return {
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "realisticVisionV60B1_v51HyperVAE.safetensors"}
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 768, "height": 384, "batch_size": 1}
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt_text, "clip": ["4", 1]}
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "blurry, low quality, text, watermark, people, faces",
                "clip": ["4", 1]
            }
        },
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
                "seed": int(time.time()),
                "steps": 25,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "karras",
                "denoise": 1.0
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]}
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": "kalimba_wave"
            }
        }
    }

@app.route("/generate", methods=["POST"])
def generate():
    try:
        raw = request.get_data(as_text=True)
        print(f"[RAW] {raw}")

        data = request.get_json(force=True)
        print(f"[TD] Recebido: {data}")

        if not data or 'notes' not in data:
            return jsonify({"status": "error", "message": "sem notas"}), 400

        notes = data['notes']
        print(f"[TD] {len(notes)} notas")

        prompt_text = notes_to_prompt(notes)
        workflow = build_workflow(prompt_text)

        client_id = str(uuid.uuid4())
        r = requests.post(
            f"{COMFY_URL}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=10
        )

        print(f"[COMFY] status: {r.status_code}")
        print(f"[COMFY] resposta: {r.text}")

        result = r.json()
        return jsonify({"status": "ok", "prompt_id": result.get("prompt_id")})

    except Exception as e:
        print(f"[ERRO COMPLETO]")
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"status": "bridge running"})

if __name__ == "__main__":
    print("Bridge a correr em http://localhost:5005")
    app.run(host="0.0.0.0", port=5005, debug=True)
