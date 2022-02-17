from src.imports import *

class ModelPrediction:
    def __init__(self, Image_Size):
        self.IMAGE_SIZE = Image_Size
        self.MODEL_PATH = '../Web/Model/model_loss=0.3458_iou=0.489.h5'
        self.OUT_IMAGE_PATH = '../Web/static/Satellite Images/'
        self.MASK_COLOR = [i / 255 for i in list([66, 255, 73])]

    @staticmethod
    def dice_coef(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

    def dice_coef_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def load_model(self):
        objects = {
            'dice_coef': self.dice_coef,
            'dice_coef_loss': self.dice_coef_loss,
            'iou_score': sm.metrics.iou_score,
            'binary_crossentropy_plus_jaccard_loss': sm.losses.bce_jaccard_loss
        }

        model = keras.models.load_model(
            self.MODEL_PATH,
            custom_objects=objects)
        return model

    def place_mask_on_img(self, img, mask):
        np.place(img[:, :, :], mask[:, :, :] >= 0.5, self.MASK_COLOR)
        return img

    def make_pred_good(self, pred):
        pred = pred[0][:, :, :]
        pred = np.repeat(pred, 3, 2)
        return pred

    def normalize_image(self, img):
        img = op.cvtColor(img, op.COLOR_BGR2RGB)
        img = img / 255.0
        img = op.resize(img, self.IMAGE_SIZE)
        img = np.expand_dims(img, axis=0)

        if len(img.shape) == 4:
            img = self.convert_png(img)

        return img

    def compute(self, model, PATH):
        img = op.imread(PATH)
        img = self.normalize_image(img)
        print()
        pred = self.make_pred_good(model(img))
        pred = self.place_mask_on_img(img[0], pred)

        plt.axis('off')
        plt.grid(False)

        plt.imsave(self.OUT_IMAGE_PATH + 'out_plot.png', pred)

    def convert_png(self, img):
        return img[:, :, :, :3]
