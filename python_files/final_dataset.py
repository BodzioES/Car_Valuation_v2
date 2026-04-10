import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping


def train_valuation_model():
    df_equip = pd.read_parquet("../files_other/final_dataset.parquet")

    data_x = df_equip.drop(columns="price_scaled")
    data_y = df_equip["price_scaled"]

    train_x, test_x, train_y, test_y = train_test_split(
        data_x, data_y, test_size=0.2, random_state=42
    )

    model = Sequential([
        Dense(128, input_dim=train_x.shape[1], activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="linear")
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    print(f"Starting training on {len(train_x)} cars...")

    history = model.fit(
        train_x, train_y,
        validation_data=(test_x, test_y),
        epochs=100,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1
    )

    model.save('car_valuation_model_modern.keras')
    print("Model has been trained and saved")


if __name__ == "__main__":
    train_valuation_model()
