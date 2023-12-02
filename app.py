from flask import Flask
from yt_dlp import YoutubeDL

app = Flask(__name__)
# model = load_model('path/to/your/fused_model.h5')

@app.route('/classify', methods=['POST'])
def classify_video():
    data = request.json
    video_url = data['url']
    video_title = data['title']
    video_id = data['id']
    

    prediction = model.predict(frames)

    # Prepare and return the result
    result = {'classification': prediction.tolist()}
    return jsonify(result)



if __name__ == '__main__':
    app.run(debug=True)



