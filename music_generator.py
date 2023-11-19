print("Loading audiocraft...")
from audiocraft.models import MusicGen
print("Loading torchaudio...")
import torchaudio

class MusicGenerator:
    def __init__(self, model_name='facebook/musicgen-small'):
      # Initialize MusicGen with a pre-trained model
      print("Loading model ", model_name, "...")
      self.model = MusicGen.get_pretrained(model_name)

    def set_params(self, use_sampling=True, top_k=250, duration=30):
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
