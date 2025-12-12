import yaml
import os

from src.data_loader import load_data
from src.model import build_model

def train():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_gen, val_gen = load_data(config)
    model = build_model(config)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config["epochs"]
    )

    os.makedirs("outputs/models", exist_ok=True)
    model.save(config["model_path"])

    print(f"✅ Model saved at {config['model_path']}")

if __name__ == "__main__":
    train()
