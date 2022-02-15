from flask import Flask, render_template, redirect, url_for, request
from werkzeug.utils import secure_filename
from src.imports import *
import segmentation_models as sm
import numpy as np

app = Flask(__name__)
app.config['IMAGE_UPLOADS'] = 'D:/Machine learning/Road Extraction/Roadment/Web/static/Satellite Images/'

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/image-upload', methods=['POST', 'GET'])
def upload_image():
    PATH = 'D:/Machine learning/Road Extraction/Roadment/Web/static/Satellite Images/'
    clear_dir(PATH)

    if request.method == 'POST':
        image = request.files['image']
        file_name = secure_filename(image.filename)

        basedir = os.path.abspath(os.path.dirname(__file__))
        image.save(os.path.join(basedir, app.config['IMAGE_UPLOADS'], file_name))
        prediction(model, file_name)

        return render_template("index.html", filename = file_name, outname = 'out_plot.png')

    return render_template("index.html")


def clear_dir(PATH):
    filenames = os.listdir(PATH)
    for f in filenames:
        os.remove(PATH + f)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(self, y_true, y_pred):
    return 1 - self.dice_coef(y_true, y_pred)


def load_model():
    objects = {
        'dice_coef': dice_coef,
        'dice_coef_loss': dice_coef_loss,
        'iou_score': sm.metrics.iou_score
    }

    model = keras.models.load_model(
        'D:\Machine learning\Road Extraction\Roadment\Web\Model\model_main_loss=0.36_iou=0.47.h5',
        custom_objects=objects)
    return model


def placeMaskOnImg(img, mask):
    color = [66, 255, 73]
    color = [i / 255.0 for i in color]
    np.place(img[:, :, :], mask[:, :, :] >= 0.5, color)
    return img


def make_pred_good(pred):
    pred = pred[0][:, :, :]
    pred = np.repeat(pred, 3, 2)
    return pred


def prediction(model, filename):

    PATH = app.config['IMAGE_UPLOADS'] + filename
    img = op.imread(PATH)
    img = op.cvtColor(img, op.COLOR_BGR2RGB)

    img = img / 255.0
    img = op.resize(img, (512, 512))
    img = np.expand_dims(img, axis=0)
    img = img[:, :, :, :3]

    pred = make_pred_good(model(img))
    pred = placeMaskOnImg(img[0], pred)

    plt.axis('off')
    plt.grid(False)

    plt.imsave('D:\Machine learning\Road Extraction\Roadment\Web\static\Satellite Images\out_plot.png', pred)


model = load_model()

def main():
    print("Server begin")

    app.run(debug=True, port=2800)


if __name__ == "__main__":
    main()
