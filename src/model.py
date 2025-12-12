from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_model(config):
    img = config["img_size"]
    classes = config["num_classes"]

    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(img, img, 3)),
        MaxPooling2D(),

        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(),

        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D(),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
