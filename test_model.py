import main
import torch
from main import model, dataloader, train_model
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support
import torch
from main import model, dataloader, train_model
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


output_dir = 'test_images'
os.makedirs(output_dir, exist_ok=True)
label_names = {0: 'No Finding', 1: 'Pleural Effusion', 2: 'Pneumonia', 3: 'Pneumothorax', 4: 'Atelectasis'}

model.load_state_dict(torch.load('model.pth'))
model.eval()

correct = 0
total = 0
losses = []
true_labels = []
predicted_probs = []

with torch.no_grad():
    for data in dataloader:
        images, labels = data['image'], data['labels']
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # Calculate batch accuracy
        correct_preds = (predicted == labels.max(1)[1])
        correct += correct_preds.sum().item()
        total += labels.size(0)
        
        # Calculate batch loss
        loss = torch.nn.functional.cross_entropy(outputs, labels.max(1)[1])
        losses.append(loss.item())

        predicted_probs.extend(torch.nn.functional.softmax(outputs,dim=1).cpu())
        true_labels.extend(labels.cpu().numpy())

        print("Tested batch of images:")
        for i in range(images.size(0)):  # Loop through each image in the batch
            fig, ax = plt.subplots()
            ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())  # Adjust for proper visualization
            expected_label = label_names[labels.max(1)[1][i].item()]
            predicted_label = label_names[predicted[i].item()]
            ax.set_title(f"Expected: {expected_label}, Predicted: {predicted_label}")
            # Save the figure
            fig.savefig(f"{output_dir}/{expected_label}_{predicted_label}.jpg")
            plt.close(fig)  # Close the plot to free up memory
        
        print(f"Batch Accuracy: {100 * correct_preds.float().mean():.2f}%")
        print(f"Batch Loss: {loss.item():.4f}\n")

true_labels = np.argmax(true_labels, axis=1) 
predicted_probs = np.array([p.numpy() for p in predicted_probs])
predicted_labels = predicted_probs.argmax(axis=1)
# conf_matrix = confusion_matrix(true_labels, predicted_labels)
# true_labels_binary = label_binarize(true_labels, classes=[0, 1, 2, 3, 4])
# # predicted_probs = np.array(predicted_probs)
# precision, recall, f1_score, _ = classification_report(true_labels.argmax(axis=1), predicted_probs.argmax(axis=1), output_dict=True)['macro avg'].values()
# roc_auc = roc_auc_score(true_labels, predicted_probs, average='macro')
# roc_auc = roc_auc_score(true_labels_binary, predicted_probs, average='macro')
precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, predicted_probs.argmax(axis=1), average='macro')
# roc_auc = roc_auc_score(true_labels_binary, predicted_probs, average='macro', multi_class='ovr')
overall_accuracy = 100 * correct / total
average_loss = sum(losses) / len(losses)
print(f'Final Accuracy of the model on the test data: {overall_accuracy:.2f}%')
print(f'Average Loss: {average_loss:.4f}')
print(f"Precision of model:", precision)
print("Recall ability:", recall)
print("F1-score:", f1_score)
# print("ROC AUC:", roc_auc)

# plt.figure(figsize=(10, 7))
# sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=label_names.values(), yticklabels=label_names.values())
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()
