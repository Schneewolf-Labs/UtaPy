from flask import Flask, request, send_file
from music_generator import MusicGenerator  # Assuming this is your class
import threading
import queue

app = Flask(__name__)
music_gen = MusicGenerator(model_name='facebook/musicgen-stereo-medium', compress_audio=True)
task_queue = queue.Queue()

def worker():
    while True:
        task = task_queue.get()
        if task is None:
            break
        prompt, duration, event = task
        if duration:
            music_gen.set_params(duration=duration)
        music_gen.generate_music(prompt)
        event.set()
        task_queue.task_done()
        
# Start worker thread
threading.Thread(target=worker, daemon=True).start()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt')
    duration = data.get('duration')
    event = threading.Event()
    task_queue.put((prompt, duration, event))
    event.wait()
    return send_file("output_audio.ogg", as_attachment=True)

if __name__ == '__main__':
    port = 5000
    print("Starting server on port", port)
    app.run(host='0.0.0.0', port=port, debug=False)
