from main import *
from augmentation import Augmentaton

class DataManager:
    def __init__(self, img_shape):
        self.IMG_SHAPE = img_shape
        self.augmentor = Augmentaton(self.IMG_SHAPE)
        self.transform = self.augmentor.build_augmentation(h_flip_prob=0.5, blur_limit= 5, blur_prob=0.85)

    def build_data(self, PATH):
        data = []
        filenames = sorted(os.listdir(PATH))

        for name in filenames:
            data.append(PATH + '/' + name)
        return data

    def build_df(self):
        TRAIN_PATH, LABEL_PATH = ' ', ' '
        sat_image = self.build_data(TRAIN_PATH)
        mask_image = self.build_data(LABEL_PATH)

        df = pd.DataFrame({
            'Image': sat_image,
            'Mask': mask_image
        })

        return df

    def modify_mask(self, mask, threshold = 100):
        mask = np.expand_dims(mask, axis = 2)
        t_mask = np.zeros(mask.shape)
        np.place(t_mask[:, :, 0], mask[:, :, 0] >= threshold, 1)
        return t_mask

    def map_function(self, img, mask):
        img, mask = plt.imread(img.decode()), plt.imread(mask.decode())
        img = op.resize(img, self.IMG_SIZE)
        mask = self.modify_mask(op.resize(mask, self.IMG_SIZE))

        img = img/255.0
        transformed = self.transform(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

        return img.astype(np.float64), mask.astype(np.float64)

    def create_dataset(self, data, BATCH_SIZE, BUFFER_SIZE = 1000):
        dataset = tf.data.Dataset.from_tensor_slices((data['Image'], data['Mask']))
        dataset = dataset.shuffle(BUFFER_SIZE)
        dataset = dataset.map(lambda img, mask : tf.numpy_function(
                    self.map_function, [img, mask], [tf.float64, tf.float64]),
                    num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE)

        dataset = dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
        return dataset

