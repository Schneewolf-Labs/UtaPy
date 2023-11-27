from flask import Flask, request, send_file
from music_generator import MusicGenerator
import threading
import queue
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
music_gen = MusicGenerator(model_name='facebook/musicgen-stereo-medium', compress_audio=True)
task_queue = queue.Queue()

def worker():
    """
    Worker thread to process music generation tasks.
    """
    while True:
        task = task_queue.get()
        if task is None:
            break

        prompt, duration, event = task
        try:
            if duration:
                music_gen.set_params(duration=duration)
            music_gen.generate_music(prompt)
        except Exception as e:
            logging.error(f"Error in generating music: {e}")
        finally:
            event.set()
            task_queue.task_done()

# Start worker thread
threading.Thread(target=worker, daemon=True).start()

@app.route('/generate', methods=['POST'])
def generate():
    """
    Route to handle music generation requests.
    """
    try:
        data = request.json
        prompt = data.get('prompt')
        duration = data.get('duration')
        event = threading.Event()

        task_queue.put((prompt, duration, event))
        event.wait()

        # Ensure the file exists
        if not os.path.exists("output_audio.ogg"):
            return "File not found", 404

        return send_file("output_audio.ogg", as_attachment=True)
    except Exception as e:
        logging.error(f"Error in generate route: {e}")
        return "Internal Server Error", 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5002))  # Use environment variable for port, default to 5002
    logging.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
