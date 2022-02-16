from flask import Flask, render_template, redirect, url_for, request
from werkzeug.utils import secure_filename
import os
from os import listdir
from pred_utils import ModelPrediction

app = Flask(__name__)
app.config['IMAGE_UPLOADS'] = r'C:/Users/Asus/Desktop/RoadMent/Roadment/Web/static/Satellite Image/'

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/image-upload', methods=['POST', 'GET'])
def upload_image():
    PATH = r'C:/Users/Asus/Desktop/RoadMent/Roadment/Web/static/Satellite Image/'
    clear_dir(PATH)

    if request.method == 'POST':
        image = request.files['image']
        file_name = secure_filename(image.filename)

        basedir = os.path.abspath(os.path.dirname(__file__))
        image.save(os.path.join(basedir, app.config['IMAGE_UPLOADS'], file_name))
        prediction(modelh5, file_name)

        return render_template("index.html", filename = file_name, outname = 'out_plot.png')

    return render_template("index.html")


def clear_dir(PATH):
    filenames = os.listdir(PATH)
    for f in filenames:
        os.remove(PATH + f)

def prediction(modelh5,filename):
    PATH = app.config['IMAGE_UPLOADS'] + filename
    compute = model.compute(modelh5,PATH)


model = ModelPrediction(Image_Size=(512,512))
modelh5 = model.load_model()

def main():
    print("Server begin")

    app.run(debug=True, port=2800)


if __name__ == "__main__":
    main()
