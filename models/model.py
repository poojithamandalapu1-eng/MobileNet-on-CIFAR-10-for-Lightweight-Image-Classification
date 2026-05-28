import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import layers,models

def build_mobilenet_model(input_shape=(96,96,3), num_classes=10):
   """
Build MobileNet model fine-tuned for CIFAR-10.
"""
print("Building MobileNet model...")

base_model = MobileNet(
input_shape = input_shape,
alpha=1.0,
include_top = False,
weights = 'imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256,activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

print("Model built successfully!")
return model,base_model



def unfreeze_model(model, base_model, unfreeze_layers=30):
    """
    print(f"unfreezing last{unfreeze_layers} layers for fine-tuning...")
    """
    base_model.trainable = True

    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False

        print("Model ready for fine-tuning!")

        return model
    
    def get_model_size(model):
        """
        Print model parameter count and size.
        """
        total_params = model.count_params()
        size_mb = (total_params * 4) / (1024 * 1024)

        print(f"\n Model Summary: ")
        print(f"   Total Parameters:  {total_params:,}")
        print(f"    Estimated Size  :  {size_mb:.2f} MB")

        return total_params, size_mb
    

    if __name__ == "__main__":
        model, base_model = build_mobilenet_model()
        model.summary()
        get_model_size(model)
        print("\n model.py is working correctly!")