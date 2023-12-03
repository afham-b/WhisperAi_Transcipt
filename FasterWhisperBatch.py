import os 
from faster_whisper import WhisperModel

model_size = "tiny"

# Initialize the model to run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Directory containing the MP3 files
audio_dir = "D:\\Folder"

# Loop through each file in the directory
for filename in os.listdir(audio_dir):
    if filename.endswith(".mp3"):
        # Full path to the audio file
        audio_file = os.path.join(audio_dir, filename)

        # Transcribe the audio file
        segments, info = model.transcribe(audio_file, beam_size=5)

        # Name for the output text file (same as audio file but with .txt extension)
        output_file = os.path.join(audio_dir, os.path.splitext(filename)[0] + ".txt")

        # Open the text file and write the transcription
        with open(output_file, "w") as file:
            file.write("Detected language '%s' with probability %f\n" % (info.language, info.language_probability))
            for segment in segments:
                file.write("[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text))

        print(f"Transcription for {filename} completed and saved to '{output_file}'")

print("All transcriptions completed.")
