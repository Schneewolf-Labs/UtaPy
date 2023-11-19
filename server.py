from flask import Flask, request, send_file
from music_generator import MusicGenerator  # Assuming this is your class

app = Flask(__name__)
music_gen = MusicGenerator()

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt')
    file_name = "output_audio.wav"
    music_gen.generate_music(prompt, file_name)
    return send_file(file_name, as_attachment=True)

if __name__ == '__main__':
    port = 5000
    print("Starting server on port", port)
    app.run(host='0.0.0.0', port=port, debug=True)
