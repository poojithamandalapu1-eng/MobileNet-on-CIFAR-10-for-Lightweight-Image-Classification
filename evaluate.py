import os
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from data_loader import load_and_preprocess_data, get_test_dataset

#CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile','bird','cat','deer','dog','frog','horse','ship','truck']
MODEL_PATH = "mobilenet_cifar10.h5"
def load_model():
    print("Loading saved model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("model loaded!")
    return model

def evaluate_accuracy(model, test_ds):
    print("\n Evaluating Accuracy..")
    loss, accuracy = model.evaluate(test_ds, verbose=0)
    print(f" Test Accuracy : {accuracy*100:.2f}%")
    print(f" Test Loss:{loss:.4f}")
    return accuracy, loss

def measure_inference_time(model, x_test):
    print("\n Measuring inference time(edge device simulation)...")

    #Warm Up
    _=model.predict(x_test[:1], verbose=0)

    #Time 100 single predictions 
    start = time.time()
    for i in range(100):
        _=model.predict(x_test[i:i+1],verbose=0)
        end =  time.time()
        avg_ms = ((end-start)/100)*1000
        print(f" Average inference time per image:{avg_ms:.2f}ms")
        return avg_ms
    
    def get_model_size_mb(path=MODEL_PATH):
        size_bytes = os.path.getsize(path)
        size_mb = size_bytes/(1024*1024)
        print(f"\n Model file size: {size_mb:.2f} MB")
        return size_mb
    
    def plot_confusion_matrix(model, x_test, y_test):
        print("\n Generating confusion matrix ....")
        y_pred = model.predict(x_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_true_classes, y_pred_classes)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklables=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title('Confusion Matrix - MobileNet on CIFAR-10')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150)
        plt.show()
        print(" Saved: confusion_matrix.png")

        def plot_accuracy_vs_size(accuracy, size_mb):
            print("\n Plotting Accuracy vs Model Size...")
            #Comparision with a simple baseline CNN(approximate values)
            models_data = {
                'Simple CNN': {'accuracy': 70, 'size_mb':5},
                'MobileNet': {'accuarcy': round(accuracy*100,2),
                              'size_mb': size_mb},
                 'ResNet-50\n(reference)': {'accuracy': 93, 'size_mb': 98},            
            }
            names = list(models_data.keys())
            accuracies =[v['accuracy'] for v in model_data.values()]
            sizes = [v['size_mb'] for v in models_data.values()]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            fig, axes = plt.subplots(1,2, figsize=(14,5))
            #Bar chart-Accuracy
            axes[0].bar(names, accuracies, color=colors, edgecolor='black',width=0.5)
            axes[0].set_title('Model Size Comparision', fontsize=14, fontweight='bold')
            axes[0].set_ylabel('Accuracy(%)')
            axes[0].set_ylim(0,100)
            for i,v in enumerate(accuracies):
                axes[0].text(i,v+1,f'{v}%', ha='center', fontweight='bold')
                
                #Bar Chart-Model Size
                axes[1].bar(names, sizes, color=colors, edgecolor='black',width=0.5)
                axes[1].set_title('Model Size Comparision', fontsize=14, fontweight='bold')
                axes[1].set_ylabel('Size(MB)')
                for i,v in enumerate(sizes):
                    axes[1].text(i, v+0.5, f'{v} MB', ha='center',fontweight='bold')

                    plt.suptitle('MobileNet vs other Models-Accuracy vs Size',fontsize=16,fontweight='bold',y=1.02)
                    plt.tight_layout()
                    plt.show()
                    print(" Saved: accuracy_vs_size.png")

                    def print_classification_report(model, x_test, y_test):
                        print("\n Classification Report:")
                        y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
                        y_true = np.argmax(y_test, axis=1)
                        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

                        #MAIN
                        if __name__=="__main__":
                            #Load model
                            model = load_model()
                            #Run all evaluations
                            accuracy, loss =evaluate_accuracy(model, test_ds)
                            avg_inference_ms = measure_inference_time(model, x_test)
                            size_mb = get_model_size_mb()

                            print_classification_report(model, x_test, y_test)
                            plot_confusion_matrix(model, x_test, y_test)
                            plot_accuracy_vs_size(accuarcy, size_mb)

                            print("\n Evaluation complete!")
                            print(f" Accuracy: {accuracy*100:.2f}%")
                            print(f"  Model Size: {size_mb:.2f}MB")
                            print(f" Inference : {avg_inference_ms:.2f} ms/image")

                    