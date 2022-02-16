from flask import Flask, render_template, redirect, url_for, request
from werkzeug.utils import secure_filename
from src.imports import *
from pred_utils import ModelPrediction
import requests, io
from PIL import Image

app = Flask(__name__)

app.config['IMAGE_UPLOADS'] = 'static/Satellite Images/'
IMAGE_SHAPE = (512, 512)
OUTPUT_IMG_NAME = 'out_plot.png'
INPUT_IMG_NAME = 'in_plot.png'

model = ModelPrediction(IMAGE_SHAPE)
modelh5 = model.load_model()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/image-upload', methods=['POST', 'GET'])
def upload_image():
    PATH = app.config['IMAGE_UPLOADS']
    clear_dir(PATH)

    if request.method == 'POST':
        image = request.files['image']
        file_name = secure_filename(image.filename)

        basedir = os.path.abspath(os.path.dirname(__file__))
        image.save(os.path.join(basedir, app.config['IMAGE_UPLOADS'], file_name))
        prediction(modelh5, file_name)

        return render_template("index.html", filename = file_name, outname = OUTPUT_IMG_NAME)

    return render_template("index.html")

@app.route('/url-upload', methods = ['POST', 'GET'])
def upload_url():

    clear_dir(app.config['IMAGE_UPLOADS'])

    url = request.form['url']
    img = load_image(url)

    plt.imsave(app.config['IMAGE_UPLOADS'] + 'in_plot.png', img)
    prediction(modelh5, 'in_plot.png')

    return render_template("index.html", filename = INPUT_IMG_NAME, outname = OUTPUT_IMG_NAME)

def load_image(url):
    response = requests.get(url)
    bytes_im = io.BytesIO(response.content)
    img = np.array(Image.open(bytes_im))[:, :, :3]
    return img

def clear_dir(PATH):
    filenames = os.listdir(PATH)
    for f in filenames:
        os.remove(PATH + f)

def prediction(modelh5, filename):
    PATH = app.config['IMAGE_UPLOADS'] + filename
    model.compute(modelh5, PATH)

def main():
    print("Server begin")

    app.run(debug=True, port=2800)


if __name__ == "__main__":
    main()
