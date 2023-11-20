from audiocraft.models import MusicGen
import torchaudio
from pydub import AudioSegment

class MusicGenerator:
    def __init__(self, model_name='facebook/musicgen-small', compress_audio=True):
        # Initialize MusicGen with a pre-trained model
        print("Loading model ", model_name, "...")
        self.model = MusicGen.get_pretrained(model_name)
        self.compress_audio = compress_audio

    def set_params(self, use_sampling=True, top_k=250, duration=15):
        # Set generation parameters
        self.model.set_generation_params(
            use_sampling=use_sampling,
            top_k=top_k,
            duration=duration
        )

    def generate_music(self, prompt, file_name="output_audio.wav"):
        # Generate music with a text prompt
        print("Generating music from prompt: ", prompt, "...")
        output = self.model.generate(
            descriptions=[prompt],
            progress=True
        )

        # Move the tensor to CPU and then save the generated audio to a file
        output_cpu = output[0].cpu()
        torchaudio.save(file_name, output_cpu, sample_rate=32000)
        print(f"Audio generated and saved as {file_name}")

        if self.compress_audio:
            self._compress_audio(file_name)

    def _compress_audio(self, file_name):
        # Compress the audio to ogg
        print("Compressing audio...")
        sound = AudioSegment.from_wav(file_name)
        compress_file_name = file_name.replace(".wav", ".ogg")
        sound.export(compress_file_name, format="ogg", codec="libvorbis")