import json
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np

# Carica i dati di training salvati
with open('training_history.json', 'r') as f:
    history = json.load(f)

# Estrai i dati
train_loss = history['train_loss']
test_loss = history['test_loss']
train_accuracy = history['train_accuracy']
test_accuracy = history['test_accuracy']
epochs = np.array(range(1, len(train_loss) + 1))

# Funzione per creare una linea spline liscia che passa attraverso i vertici
def smooth_line(x, y):
    spline = make_interp_spline(x, y, k=3)  # k=3 per spline cubica
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

# Stampa delle percentuali finali di perdita e accuratezza
final_train_loss = train_loss[-1]
final_test_loss = test_loss[-1]
final_train_accuracy = train_accuracy[-1] * 100
final_test_accuracy = test_accuracy[-1] * 100

print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Testing Loss: {final_test_loss:.4f}")
print(f"Final Training Accuracy: {final_train_accuracy:.2f}%")
print(f"Final Testing Accuracy: {final_test_accuracy:.2f}%")

# Grafico della perdita con spline, senza griglia
x_smooth, y_smooth = smooth_line(epochs, train_loss)
plt.figure(figsize=(12, 6))
plt.plot(x_smooth, y_smooth, 'b-', label='Training Loss')
x_smooth, y_smooth = smooth_line(epochs, test_loss)
plt.plot(x_smooth, y_smooth, 'r-', label='Testing Loss')
plt.title('Training and Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Grafico dell'accuratezza con spline, senza griglia
x_smooth, y_smooth = smooth_line(epochs, train_accuracy)
plt.figure(figsize=(12, 6))
plt.plot(x_smooth, y_smooth, 'b-', label='Training Accuracy')
x_smooth, y_smooth = smooth_line(epochs, test_accuracy)
plt.plot(x_smooth, y_smooth, 'r-', label='Testing Accuracy')
plt.title('Training and Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

