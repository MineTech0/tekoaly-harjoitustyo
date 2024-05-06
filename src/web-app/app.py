from flask import Flask, request, render_template
from services.prediction_service import PredictionService

app = Flask(__name__)

prediction_service = PredictionService()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Assume the song is passed as a file
        file = request.files['file']
        prediction, confidence = prediction_service.predict_genre_peer_model(file)

        return render_template('index.html', prediction=prediction, confidence=confidence)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
