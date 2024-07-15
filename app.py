from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

def model_output(f, stress, b, s):
    # if b < 0:
    #     return ((s**(1/b)) / (2 * f * (stress**(1/b)) * (0.683 + (0.271 * 2**(1/b)) + (0.043 * 3**(1/b)))))
    # else:
        return ((s**(-1/b)) / (2 * f * (stress**(-1/b)) * (0.683 + (0.271 * 2**(-1/b)) + (0.043 * 3**(-1/b)))))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stress = float(request.form['stress'])
    frequency = float(request.form['frequency'])
    # Assuming your model takes stress and frequency as input
    # and returns the predicted RUL
    predicted_rul = model_output(frequency, stress, *model)
    return render_template('result.html', predicted_rul=predicted_rul)

if __name__ == '__main__':
    app.run(debug=True)