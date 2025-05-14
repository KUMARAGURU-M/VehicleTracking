import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# ------------------------------
# 1. Load Preprocessed Data
# ------------------------------
X = np.load("X.npy", allow_pickle=True)
y = np.load("y.npy", allow_pickle=True)

print("✅ Loaded Data Successfully!")
print("📊 X shape:", X.shape)  # (num_sequences, SEQUENCE_LENGTH, num_features)
print("📊 y shape:", y.shape)  # (num_sequences, num_classes)

# ------------------------------
# 2. Train-Test Split
# ------------------------------
y_labels = np.argmax(y, axis=1)  # Convert one-hot to labels for stratification

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y_labels, random_state=42
)

print("📌 Training Set Shape:", X_train.shape, y_train.shape)
print("📌 Testing Set Shape:", X_test.shape, y_test.shape)

# ------------------------------
# 3. Build the LSTM Model
# ------------------------------
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),

    # First LSTM layer
    LSTM(128, return_sequences=True),
    Dropout(0.3),

    # Second LSTM layer
    LSTM(64, return_sequences=True),
    Dropout(0.3),

    # Third LSTM layer (final output)
    LSTM(32, return_sequences=False),
    Dropout(0.2),

    # Dense layer
    Dense(64, activation='relu'),
    Dropout(0.2),

    # Output layer (softmax for classification)
    Dense(y_train.shape[1], activation='softmax')
])

# ------------------------------
# 4. Compile the Model
# ------------------------------
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------------
# 5. Define Callbacks
# ------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# ------------------------------
# 6. Train the Model
# ------------------------------
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr]
)

# ------------------------------
# 7. Evaluate Model
# ------------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print(f"🎯 Test Accuracy: {accuracy * 100:.2f}%")

# ------------------------------
# 8. Save Trained Model
# ------------------------------
model.save("final_route_prediction_model.h5")
print("✅ Model saved as 'final_route_prediction_model.h5'")
