"""
ML Training Results Demo
Simple demonstration of image classification model training across 5 epochs.
"""

import matplotlib.pyplot as plt

# Hardcoded training data
epochs = [1, 2, 3, 4, 5]
train_accuracy = [0.7450, 0.7650, 0.7625, 0.7850, 0.7950]
val_accuracy = [0.8350, 0.8100, 0.8250, 0.8025, 0.8275]

# Print epoch-by-epoch results
print("\n" + "=" * 50)
print("Model Training Progress")
print("=" * 50)

for epoch, train_acc, val_acc in zip(epochs, train_accuracy, val_accuracy):
    print(f"Epoch {epoch}: Train = {train_acc*100:.2f}%, Val = {val_acc*100:.2f}%")

# Calculate and print final results
final_val_accuracy = val_accuracy[-1]
print("=" * 50)
print(f"Final Validation Accuracy: ~{final_val_accuracy*100:.0f}%")
print("=" * 50 + "\n")

# Create a clean plot
plt.figure(figsize=(10, 6))

# Plot both curves
plt.plot(epochs, train_accuracy, marker='o', linewidth=2, label='Training Accuracy', color='#3b82f6')
plt.plot(epochs, val_accuracy, marker='s', linewidth=2, label='Validation Accuracy', color='#ef4444')

# Formatting
plt.xlabel('Epoch', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
plt.title('Model Training Performance: Training vs Validation Accuracy', fontsize=14, fontweight='bold')
plt.xticks(epochs)
plt.ylim([0.70, 0.85])
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=11, loc='best')

# Save and show
plt.tight_layout()
plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
print("Graph saved as: training_results.png\n")
plt.show()
