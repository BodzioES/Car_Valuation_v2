import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split


def train_valuation_model():
    df_equip = pd.read_parquet("final_dataset.parquet")

    data_x = df_equip.drop(columns="price")
    data_y = df_equip["price"]

    train_x, test_x, train_y, test_y = train_test_split(
        data_x, data_y, test_size=0.2, random_state=42
    )

    model = Sequential([
        Dense(64, input_dim=train_x.shape[1], activation="relu"),
        Dropout(0.2),

        Dense(32, activation="relu"),

        Dense(16, activation="relu"),

        Dense(1, activation="linear")
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=["mae"]
    )

    history = model.fit(
        train_x, train_y,
        validation_data=(test_x, test_y),
        epochs=100,
        batch_size=32,
        verbose=1
    )

    model.save('car_valuation_model.keras')


if __name__ == "__main__":
    train_valuation_model()
