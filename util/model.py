from tensorflow.keras import Sequential
from tensorflow.keras.layers import GlobalAveragePooling3D, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import regularizers

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
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.5),

        # Output Layer:
        Dense(num_classes, activation='softmax')


        ])
  return model

def compile_model(model):
  # Compile the Model
   model.compile(
      optimizer=Adam(learning_rate=1e-4),
      loss='categorical_crossentropy',
      metrics=['accuracy']
   )

def train_model(model, params, train_size, train_dataset_batched, test_dataset_batched, val_dataset_batched):
  print("Starting training on the training split...")
  
  try:
      # add scheduler to start aggressive then fine tune
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
    	factor=0.5,
    	patience=4,
    	min_lr=1e-6
	  )

    callbacks = [
    	keras.callbacks.TensorBoard(log_dir='./logs'),
 		lr_scheduler,
	  	tf.keras.callbacks.EarlyStopping(
        	monitor='val_loss',
        	patience=6,
        	restore_best_weights=True
    	)    
	  ]

    steps_per_epoch = train_size // params['batch_size']
    
    # address class imbalance (hard coded, fix later)
    N0 = 266
    N1 = 44
    N2 = 62
    total = N0 + N1 + N2
    # clip weights
    class_weight = {
      0: total / (params['num_classes'] * N0),
      1: total / (params['num_classes'] * N1),
      2: total / (params['num_classes'] * N2)
    }

    history = model.fit(
      # Train only on the batched training dataset
      train_dataset_batched,
      steps_per_epoch=steps_per_epoch,

      # Use the batched validation dataset for monitoring
      validation_data=val_dataset_batched,

      epochs=params['epochs'],
      verbose=1,
      callbacks=callbacks,
	    class_weight=class_weight
    )

    # You will typically evaluate the final model performance
    # on the test set *after* training is complete
    loss, accuracy = model.evaluate(test_dataset_batched, verbose=0)
    print(f"\nFinal Test Set Loss: %.2f" % loss)
    print(f"\nFinal Test Set Accuracy: {accuracy*100:.2f}%")
    
    return history 

  except Exception as e: print("Training failed with exception: ", e) 

