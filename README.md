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


# For the ComfyUI generation

--- main.py 

1) Download the local ComfyUI
--- brigde.py 
