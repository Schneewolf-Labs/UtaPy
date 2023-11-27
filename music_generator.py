import logging
from audiocraft.models import MusicGen
import torchaudio
from pydub import AudioSegment


class MusicGenerator:
    """
    Music generator class using audiocraft.models.MusicGen for generating music from text prompts.
    """

    def __init__(self, model_name='facebook/musicgen-small', compress_audio=True):
        """
        Initialize the MusicGen model with a pre-trained model and set compression settings.
        """
        logging.info(f"Loading model {model_name}...")
        self.model = MusicGen.get_pretrained(model_name)
        self.compress_audio = compress_audio

    def set_params(self, use_sampling=True, top_k=250, top_p=0.0, temperature=1.0, duration=30.0, cfg_coef=3.0, two_step_cfg=False, extend_stride=18):
        """
        Set generation parameters for the music generation model.
        """
        self.model.set_generation_params(
            use_sampling=use_sampling,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            duration=duration,
            cfg_coef=cfg_coef,
            two_step_cfg=two_step_cfg,
            extend_stride=extend_stride
        )

    def generate_music(self, prompt, file_name="output_audio.wav"):
        """
        Generate music based on the given text prompt and save the audio to a file.
        """
        logging.info(f"Generating music from prompt: {prompt}...")
        try:
            output = self.model.generate(
                descriptions=[prompt],
                progress=True
            )

            output_cpu = output[0].cpu()
            torchaudio.save(file_name, output_cpu, sample_rate=32000)
            logging.info(f"Audio generated and saved as {file_name}")

            if self.compress_audio:
                self._compress_audio(file_name)

        except Exception as e:
            logging.error(f"Error in generating music: {e}")

    def _compress_audio(self, file_name):
        """
        Compress the generated audio file to ogg format.
        """
        logging.info("Compressing audio...")
        try:
            sound = AudioSegment.from_wav(file_name)
            compress_file_name = file_name.replace(".wav", ".ogg")
            sound.export(compress_file_name, format="ogg", codec="libvorbis")
            logging.info(f"Audio compressed and saved as {compress_file_name}")

        except Exception as e:
            logging.error(f"Error in compressing audio: {e}")


# Setup basic logging
logging.basicConfig(level=logging.INFO)