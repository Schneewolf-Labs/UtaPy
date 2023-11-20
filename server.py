from flask import Flask, request, send_file
from music_generator import MusicGenerator  # Assuming this is your class

app = Flask(__name__)
music_gen = MusicGenerator(model_name='facebook/musicgen-stereo-medium', compress_audio=True)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt')
    duration = data.get('duration')
    if duration:
            music_gen.set_params(duration=duration)
    file_name = "output_audio.ogg"
    music_gen.generate_music(prompt)
    return send_file(file_name, as_attachment=True)

if __name__ == '__main__':
    port = 5000
    print("Starting server on port", port)
    app.run(host='0.0.0.0', port=port, debug=False)
