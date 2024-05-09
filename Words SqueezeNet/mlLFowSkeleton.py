#Skeleton to use in the code at your own convenience
import mlflow
import mlflow.tensorflow
from SqueezeNetTrainer import *

# Now you can use SqueezeNet and prepare_data directly
train_dir = 'data/'
validation_dir = 'validate/'
model = SqueezeNet(input_shape=(224, 224, 3), num_classes=33)
train_generator, validation_generator = prepare_data(train_dir, validation_dir)

# Continue with your setup and training logic


mlflow.tensorflow.autolog()  # Automatically logs metrics, parameters, and models
mlflow.start_run(run_name="SqueezeNet_Training")

# mlflow.set_tracking_uri('http://your-tracking-server:port')

# Log batch size and number of epochs as parameters manually
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("epochs", 10)
mlflow.log_param("num_classes", num_classes)

# For any additional non-automated metrics or artifacts
# mlflow.log_artifact("your_additional_file.txt")

# Model training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=callbacks
)

# End the MLflow run after the training is complete
mlflow.end_run()

# Manually log models
mlflow.keras.log_model(model, "model")

# Save the confusion matrix as an image and log it
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig("confusion_matrix.png")
plt.close()

mlflow.log_artifact("confusion_matrix.png")
