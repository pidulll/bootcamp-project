from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import mysql.connector
from datetime import datetime

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model = load_model('models/skin_model.h5')
labels = ['Eczema', 'Melanoma', 'Normal']

# MySQL connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="skinscanai"
)
cursor = db.cursor()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle uploaded file
        file = request.files['file']
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)[0]
        class_idx = np.argmax(prediction)
        result = labels[class_idx]
        confidence = float(np.max(prediction))

        # Save to DB
        sql = "INSERT INTO uploads (image_path, result, confidence, upload_time) VALUES (%s, %s, %s, %s)"
        val = (filename, result, confidence, datetime.now())
        cursor.execute(sql, val)
        db.commit()

        return render_template('result.html', result=result, confidence=round(confidence*100, 2), image=filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
