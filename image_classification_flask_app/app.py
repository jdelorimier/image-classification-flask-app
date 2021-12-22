from flask import Flask

UPLOAD_FOLDER = 'image_classification_flask_app/static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024