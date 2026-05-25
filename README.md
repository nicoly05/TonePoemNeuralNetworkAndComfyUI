# TonePoemNeuralNetworkAndComfyUI


# For the Neural Networking training the sound outputs
1) requirements 
   python3 --version  
   brew --version 
   brew install portaudio 

   if you don´t have Homebrew:
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

2) open the folder
   -> install the folder
   
   cd ~/folderPath
   source venv/bin/activate
   pip install -r requirements.txt
   python -c "import numpy; import soundfile; print('OK')"

3) fix the live.py code
   nano live.py
   
  // find and change yours input and output devices
   INPUT_DEVICE_HINT  = "Microfone" 
   OUTPUT_DEVICE_HINT = "Alto-falantes"
   
   // find "channels=2" and update to
   channels=1,

   // find and change inside the "_input_callback" function
   mono = indata[:, 0].astype(np.float32)

   // get out of the nano 
   Ctrl+O  →  Enter  →  Ctrl+X

4) run the scripts
   python prepare_audio.py
   python build_map.py
   python train.py
   python live.py --input-device 0 --output-device 1

-- in case of errors
1) ModuleNotFoundError (numpy, soundfile, etc.)
   venv is not active.
   Run:
   source venv/bin/activate

2) PortAudioError — Invalid number of channels
   The channels=2 error has not yet been corrected to channels=1 in live.py.

3) torch does not install (incompatible version)
   Run:
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt --ignore-installed torch torchaudio

4) View available audio devices
   Run:
     python live.py --list-devices

# TONE POEM

> A 17-key kalimba performance becomes a generative audiovisual experience in real time.

Each key played is analyzed spectrally, feeding a neural network that synthesizes new sounds and triggers AI-generated imagery — making every performance a unique, unrepeatable artwork.

---

## Pipeline

```
Kalimba (microphone)
       ↓
TouchDesigner — spectral analysis, note detection, real-time visual
       ↓  OSC port 9001
live_generative.py — neural network, audio synthesis
       ↓  HTTP port 5005
bridge.py + ComfyUI — AI image generation
       ↓
Projected output — synthesized sound + generative visuals
```

---

## Hardware Requirements

| Item | Specification |
|------|--------------|
| Kalimba | 17 keys in C major |
| Microphone | Condenser or audio interface (BlackHole 2ch recommended) |
| Computer | macOS 12+ or Windows 10+, min. 16GB RAM |
| GPU | NVIDIA 8GB+ VRAM or Apple Silicon (for ComfyUI) |
| Projector | Optional, for installation |

---

## Software Requirements

| Software | Version | Link |
|----------|---------|------|
| TouchDesigner | 2025.32280+ | [derivative.ca](https://derivative.ca) |
| Python | 3.11+ | [python.org](https://python.org) |
| ComfyUI | latest | [github.com/comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI) |
| VS Code | any | [code.visualstudio.com](https://code.visualstudio.com) |
| Git | any | [git-scm.com](https://git-scm.com) |

---

## Repository Structure

```
tone-poem/
├── touchdesigner/
│   └── tone_poem.toe          # TouchDesigner project
├── kalimba_bridge/
│   └── bridge.py              # Flask bridge to ComfyUI
├── main_project_third/
│   ├── live_generative.py     # Neural network + audio synthesis
│   ├── inference.py
│   └── models/                # Trained models
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/tone-poem.git
cd tone-poem
```

---

### 2. Python environment

```bash
# Enter the neural network folder
cd main_project_third

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS
# venv\Scripts\activate         # Windows

# Install dependencies
pip install numpy sounddevice scipy python-osc torch torchaudio
```

```bash
# Install bridge dependencies
cd ../kalimba_bridge
pip install flask requests
```

> **NVIDIA GPU?** Install PyTorch with CUDA:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

---

### 3. ComfyUI

```bash
# Clone ComfyUI
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI

# Create environment and install
python3 -m venv comfyui-env
source comfyui-env/bin/activate
pip install -r requirements.txt
```

**Download a model** and place it in `ComfyUI/models/checkpoints/`:

```bash
cd models/checkpoints
curl -L -o realisticVision.safetensors \
  "https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE/resolve/main/Realistic_Vision_V6.0_B1_noVAE.safetensors"
```

Open `kalimba_bridge/bridge.py` and confirm the model name matches:

```python
"ckpt_name": "realisticVision.safetensors"
```

---

### 4. TouchDesigner

1. Open TouchDesigner (version 2025.32280+)
2. File → Open → `touchdesigner/tone_poem.toe`
3. Click **Allow** if prompted about external scripts
4. Click on `audiodevIn1` and select your microphone under **Device**

**Initialize tables** — open Textport (ALT+T) and run line by line:

```python
op('session_control').clear()
op('session_control').appendRow(['recording', 'user_id', 'send_time'])
op('session_control').appendRow([0, 1, 0])
```

```python
op('kalimba_session').clear()
op('kalimba_session').appendRow(['nota', 'freq', 'timestamp'])
```

```python
t = op('color_control')
t.clear()
t.appendRow(['note','position','color_index','r','g','b'])
t.appendRow(['D6',0,0,0.05,0.3,1.0])
t.appendRow(['B5',1,0,0.05,0.3,1.0])
t.appendRow(['G5',2,0,0.05,0.3,1.0])
t.appendRow(['E5',3,0,0.05,0.3,1.0])
t.appendRow(['C5',4,0,0.05,0.3,1.0])
t.appendRow(['A4',5,0,0.05,0.3,1.0])
t.appendRow(['F4',6,0,0.05,0.3,1.0])
t.appendRow(['D4',7,0,0.05,0.3,1.0])
t.appendRow(['C4',8,0,0.05,0.3,1.0])
t.appendRow(['E4',9,0,0.05,0.3,1.0])
t.appendRow(['G4',10,0,0.05,0.3,1.0])
t.appendRow(['B4',11,0,0.05,0.3,1.0])
t.appendRow(['D5',12,0,0.05,0.3,1.0])
t.appendRow(['F5',13,0,0.05,0.3,1.0])
t.appendRow(['A5',14,0,0.05,0.3,1.0])
t.appendRow(['C6',15,0,0.05,0.3,1.0])
t.appendRow(['E6',16,0,0.05,0.3,1.0])
```

**Verify** — play a note and run in Textport:

```python
print(op('/project1/script1')['hz'][0])
print(op('/project1/script1')['position'][0])
```

`hz` should show a value between 200–1400. `position` should show 0.0 to 1.0.

---

## Running

Always start in this order — **3 separate terminals**:

**Terminal 1 — ComfyUI**
```bash
cd ComfyUI
source comfyui-env/bin/activate
python main.py --listen
# Wait for: To see the GUI go to: http://127.0.0.1:8188
```

**Terminal 2 — Bridge**
```bash
cd tone-poem/kalimba_bridge
python bridge.py
# Wait for: Bridge a correr em http://localhost:5005
```

**Terminal 3 — Neural Network**
```bash
cd tone-poem/main_project_third
source venv/bin/activate
python live_generative.py --list-devices
# Then run with your device indices:
python live_generative.py --input-device 2 --output-device 1 --osc-port 9001 --echo-delay 1.0
```

**TouchDesigner** — open `tone_poem.toe`, confirm microphone is active.

**Play** — click `btn_record`, play the kalimba, click `btn_stop_send` to generate image.

---

## Controls (live_generative.py)

| Key | Action |
|-----|--------|
| `q` | quit |
| `t` / `T` | raise / lower onset threshold |
| `f` / `F` | raise / lower flux sensitivity |
| `+` / `-` | increase / decrease K (neighbours) |
| `d` / `D` | increase / decrease echo delay |
| `i` | print current settings |

---

## Troubleshooting

**`Port 5000 already in use`**
Disable AirPlay Receiver: System Settings → General → AirDrop & Handoff → AirPlay Receiver → Off

**`Connection refused 8188`**
ComfyUI is not running. Start Terminal 1 first.

**`hz = 0.0` in TouchDesigner**
Microphone not selected. Click `audiodevIn1` and choose the correct device.

**OSC not reaching TouchDesigner**
Confirm the OSC In DAT is on port `9001` and `Active = On`.

**Model not found in ComfyUI**
Confirm the exact filename in `bridge.py` matches the file in `models/checkpoints/`.

---

## Color Palette (Neural Network output 0–6)

| Value | Color | RGB |
|-------|-------|-----|
| 0 | Deep Blue | (0.05, 0.30, 1.00) |
| 1 | Cyan | (0.00, 0.90, 0.90) |
| 2 | Violet | (0.50, 0.00, 1.00) |
| 3 | Neon Green | (0.00, 1.00, 0.40) |
| 4 | Orange | (1.00, 0.30, 0.00) |
| 5 | Magenta | (1.00, 0.00, 0.50) |
| 6 | White | (1.00, 1.00, 1.00) |

---

## License

MIT License — free to use, modify and distribute with attribution.

---

*Tone Poem — where music becomes image.*
