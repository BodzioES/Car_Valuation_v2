import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def train_valuation_model():
    """
    Builds, trains, and saves a deep neural network for vehicle price estimation.
    Uses a funnel architecture with Dropout layers to ensure high generalization.
    """
    # Load the consolidated dataset
    df_equip = pd.read_parquet("../files_other/final_dataset.parquet")

    # Split features (X) from the target value (Y - scaled price)
    data_x = df_equip.drop(columns="price_scaled")
    data_y = df_equip["price_scaled"]

    # Divide data into Training (80%) and Validation (20%) sets
    train_x, test_x, train_y, test_y = train_test_split(
        data_x, data_y, test_size=0.2, random_state=42
    )

    # Neural Network Architecture: Sequential Deep Learning model
    model = Sequential([
        # Input layer: 256 neurons to capture initial feature correlations
        Dense(256, input_dim=train_x.shape[1], activation="relu"),
        Dropout(0.2),  # Prevents overfitting by randomly disabling 20% of neurons

        # Hidden layers: Funneling data to extract abstract patterns
        Dense(128, activation="relu"),
        Dropout(0.1),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),

        # Output layer: Linear activation for regression (continuous price value)
        Dense(1, activation="linear")
    ])

    # Compile the model using Adam optimizer and Mean Squared Error loss function
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]  # Mean Absolute Error for human-readable performance tracking
    )

    # Callback: Stop training if validation loss stops improving (prevents memorization)
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    # Callback: Dynamically reduce Learning Rate when performance plateaus
    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )

    print(f"Starting training on {len(train_x)} cars...")

    # Execute the training process
    history = model.fit(
        train_x, train_y,
        validation_data=(test_x, test_y),
        epochs=100,
        batch_size=128,  # Process 128 records per update for stable gradient descent
        callbacks=[early_stopping, lr_scheduler],
        verbose=1
    )

    model.save('car_valuation_model.keras')
    print("Model has been trained and saved")


if __name__ == "__main__":
    train_valuation_model()