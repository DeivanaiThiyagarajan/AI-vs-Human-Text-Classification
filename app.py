from flask import Flask, render_template, request, jsonify
from TextDetector import TextDetector

app = Flask(__name__)

detector = TextDetector()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form.get('inputText')
    if not input_text:
        return jsonify({'error': 'No input provided'}), 400

    # Call the detector's prediction method (you'll implement this later)
    prediction = detector.predict(input_text)

    return jsonify({'result': prediction})


if __name__ == '__main__':
    app.run(debug=True)
