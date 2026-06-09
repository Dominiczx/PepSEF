import re
import matplotlib.pyplot as plt

losses = []
accuracies = []

with open('/home/dataset-local/chenzixu/PepSEF/01_pretraining/bert_pretraining/outputs/nohup2.out', 'r') as f:
    for line in f:
        match = re.search(r'Epoch\s+\d+/\d+,\s+Loss:\s+([\d.]+),\s+Accuracy:\s+([\d.]+)%', line)
        if match:
            loss = float(match.group(1))
            acc = float(match.group(2))
            losses.append(loss)
            accuracies.append(acc)

epochs = range(1, len(losses) + 1)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs, losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(epochs, accuracies, label='Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy')
plt.grid(True)

plt.tight_layout()
plt.savefig('nohup2_training_curve.png')
plt.show()