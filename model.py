import tensorflow as tf

def generate_model_binary() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(64, 64, 2), dtype='float32')

    layer = tf.keras.layers.Conv2D(filters=16, strides=(1, 1), kernel_size=(3, 3), activation='relu')(inputs)
    #layer = tf.keras.layers.ActivityRegularization(l1=0.001)(layer)
    layer = tf.keras.layers.Lambda(tf.nn.local_response_normalization)(layer)
    layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), )(layer)

    layer = tf.keras.layers.Conv2D(filters=32, strides=(1, 1), kernel_size=(3, 3), activation='relu')(layer)
    #layer = tf.keras.layers.ActivityRegularization(l1=0.001)(layer)
    layer = tf.keras.layers.Lambda(tf.nn.local_response_normalization)(layer)
    layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(layer)

    layer = tf.keras.layers.Conv2D(filters=64, strides=(1, 1), kernel_size=(3, 3), activation='relu')(layer)
    #layer = tf.keras.layers.ActivityRegularization(l1=0.001)(layer)
    layer = tf.keras.layers.Lambda(tf.nn.local_response_normalization)(layer)
    layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(layer)

    layer = tf.keras.layers.Flatten()(layer)
    layer = tf.keras.layers.Dense(units=64, activation="relu")(layer)
    layer = tf.keras.layers.Dense(units=64, activation="relu")(layer)
    layer = tf.keras.layers.Dropout(0.3)(layer)
    #layer = tf.keras.layers.ActivityRegularization(l1=0.001)(layer)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(layer)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Nadam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"])
    return model