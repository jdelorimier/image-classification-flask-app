import os
import json
import torch
from app import app
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from torchvision import transforms
from PIL import Image
from model.model import MResnet

model = MResnet(in_channels=3,num_classes=100)
model.load_state_dict(torch.load('./model/full_train.pth'))

with open('./model/classes.json', 'r') as infile:
    data = json.loads(infile.read())
classes = data['classes']

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_predict(image_path):
    model.eval()
    prod_transform = transforms.Compose(
       [
       transforms.Resize(35),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
       ])
    image = Image.open(image_path)
    img_tensor = prod_transform(image).to('cpu').unsqueeze(0)
    model_output = model(img_tensor)
    _, predicted = torch.max(model_output, 1)
    prediction = classes[predicted]
    return prediction
    
@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        print('upload_image filename: ' + filename)
        # flash('Image successfully uploaded and displayed below')
        prediction = image_predict(save_path)
        flash(f"*beep* *boop* *boop* is this a picture of {prediction}?")
        return render_template('upload.html', filename=filename, prediction=prediction)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()