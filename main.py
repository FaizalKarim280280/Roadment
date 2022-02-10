from src.model_v1 import  Model
from src.dataset import DataManager
from src.imports import *

# plt.style.use('seaborn')

INPUT_SHAPE = (512, 512)
CLASSES = 2
BATCH_SIZE = 4
UNITS = 64
DROPOUT = 0.2
LEARNING_RATE = 1e-5

def train_model():
    model = Model(INPUT_SHAPE, CLASSES)
    data = DataManager(INPUT_SHAPE)

    unet = model.unet(UNITS, DROPOUT, LEARNING_RATE)
    callback = keras.callbacks.LearningRateSchedular(model.lr_scheduler)
    train_dataset = data.create_dataset(BATCH_SIZE, BUFFER_SIZE=1000)

    unet.fit(
        train_dataset,
        callback = [callback],
        epochs = 10
    )


def main():
    pass


if __name__ == "__main__":
    main()