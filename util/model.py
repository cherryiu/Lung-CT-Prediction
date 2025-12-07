from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling3D, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import tensorflow as tf

# Turn off XLA JIT globally to addr kernel error
tf.config.optimizer.set_jit(False)

# build instance of model
def build_3d_model(target_shape, num_classes):
  model = Sequential([
        # Input Layer: Must match the shape of your tensors
        tf.keras.Input(shape=target_shape),

        # Block 1
        Conv3D(filters=32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling3D(pool_size=2),
        Dropout(0.2),

        # Block 2
        Conv3D(filters=64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling3D(pool_size=2),
        Dropout(0.2),

        # Block 3
        Conv3D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling3D(pool_size=2),
        Dropout(0.3),

        # Classification/Segmentation Head
        GlobalAveragePooling3D(),

        #Flatten(), # Convert 3D feature map to 1D vector
        Dense(512, activation='relu'),
        Dropout(0.5),

        # Output Layer:
        # # For Binary Classification (e.g., tumor/no tumor)
        # Dense(1, activation='sigmoid')

        # OR for Multi-class Classification:
        Dense(num_classes, activation='softmax')

        # OR for Segmentation (U-Net style, requires TransposeConv3D/UpSampling3D layers)
    ])
  return model

def compile_model(model):
  # Compile the Model
   model.compile(
      optimizer=Adam(learning_rate=0.0001),
      loss='categorical_crossentropy',
      metrics=['accuracy']
   )

def train_model(model, params, train_size, train_dataset_batched, test_dataset_batched, val_dataset_batched):
  print("Starting training on the training split...")
  
  try:
    callbacks = [
      keras.callbacks.TensorBoard(log_dir='./logs')
    ]

    steps_per_epoch = train_size // params['batch_size']

    history = model.fit(
      # Train only on the batched training dataset
      train_dataset_batched,
      steps_per_epoch=steps_per_epoch,

      # Use the batched validation dataset for monitoring
      validation_data=val_dataset_batched,

      epochs=params['epochs'],
      verbose=1,
      callbacks=callbacks
    )

    # You will typically evaluate the final model performance
    # on the test set *after* training is complete
    loss, accuracy = model.evaluate(test_dataset_batched, verbose=0)
    print(f"\nFinal Test Set Loss: %.2f" % loss)
    print(f"\nFinal Test Set Accuracy: {accuracy*100:.2f}%")
  
  except Exception as e: print(e) 

