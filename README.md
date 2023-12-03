# WhisperAi_Transcipt
Converts batches of mp3 in a directory into txt file transcript, using the whisper ai model, running on CPU 

If using a CUDA version of pytorch, with a compatabile Nvidia GPU, it can be run faster by setting the device to CUDA 
>> model = WhisperModel(model_size, device="cpu", compute_type="int8")

First install pytorch, choco, then whisper ai, then faster whisper ai 

Pytorch linK;
https://pytorch.org/get-started/locally/ 

Using chocolatey https://chocolatey.org/
>>choco install ffmpeg

pip install -U openai-whisper
pip install faster-whisper

Now cd into the directory with the mp3 or wav files in cmd you can type
>> whisper audio.mp3
to get a transcipt back
or you can run the python code here, using your own directory.


# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("audio.mp3", beam_size=5)
