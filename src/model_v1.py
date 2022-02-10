from imports import *

class Model:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    @staticmethod
    def dice_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

    def dice_coef_loss(self, y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)

    def unet(self, units, dropout, learning_rate):
        inp = Input(self.input_shape)
        # Block 1
        x = Conv2D(units, (3, 3), padding='same', name='block1_conv1')(inp)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(units, (3, 3), padding='same', name='block1_conv2')(x)
        x = BatchNormalization()(x)
        block_1_out = Activation('relu')(x)
        x = MaxPooling2D()(block_1_out)
        x = Dropout(dropout)(x)

        # Block 2
        x = Conv2D(units * 2, (3, 3), padding='same', name='block2_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(units * 2, (3, 3), padding='same', name='block2_conv2')(x)
        x = BatchNormalization()(x)
        block_2_out = Activation('relu')(x)
        x = MaxPooling2D()(block_2_out)
        x = Dropout(dropout)(x)

        # Block 3
        x = Conv2D(units * 4, (3, 3), padding='same', name='block3_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(units * 4, (3, 3), padding='same', name='block3_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(units * 4, (3, 3), padding='same', name='block3_conv3')(x)
        x = BatchNormalization()(x)
        block_3_out = Activation('relu')(x)
        x = MaxPooling2D()(block_3_out)
        x = Dropout(dropout)(x)

        # Block 4
        x = Conv2D(units * 8, (3, 3), padding='same', name='block4_conv1')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(units * 8, (3, 3), padding='same', name='block4_conv2')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(units * 8, (3, 3), padding='same', name='block4_conv3')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

        block_4_out = Activation('relu')(x)

        x = Conv2DTranspose(units * 8, (2, 2), strides=(2, 2), padding='same', name = 'Conv2DTranspose_UP2')(block_4_out)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Concatenate()([x, block_3_out])
        x = Conv2D(units * 8, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(units * 8, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout)(x)

        # UP 3
        x = Conv2DTranspose(units * 2, (2, 2), strides=(2, 2), padding='same', name = 'Conv2DTranspose_UP3')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Concatenate()([x, block_2_out])
        x = Conv2D(units * 2, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(units * 2, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout)(x)

        # UP 4
        x = Conv2DTranspose(units * 1, (2, 2), strides=(2, 2), padding='same', name = 'Conv2DTranspose_UP4')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Concatenate()([x, block_1_out])
        x = Conv2D(units * 1, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(units * 1, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout)(x)

        x = Conv2D(self.num_classes, (3, 3), activation='sigmoid', padding='same')(x)

        model = keras.models.Model(inputs=inp, outputs=x)
        model.compile(
            optimizer = keras.optimizers.Adam(learning_rate = learning_rate),
            loss = self.dice_coef_loss,
            metrics = [self.dice_coef, sm.metrics.iou_score])

        return model

    def lr_scheduler(self, epoch, lr):
        factor, step = 0.3, 5
        if epoch % step == 0 and epoch != 0:
            print("lr changed from {} to {}".format(lr, lr*factor))
            return lr * factor
        else:
            return lr


