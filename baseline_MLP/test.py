# test.py
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from preprocess import load_and_preprocess
from model import MLPClassifier
from config import BATCH_SIZE, THRESHOLD, DEVICE

# Load preprocessed data
X_train, y_train, X_test, y_test = load_and_preprocess()

# Convert to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(DEVICE)

# Create DataLoader
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load trained model
input_dim = X_train.shape[1]
model = MLPClassifier(input_dim).to(DEVICE)
model.load_state_dict(torch.load("mlp_nslkdd.pth", map_location=DEVICE))
model.eval()

# Evaluate
y_pred = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        outputs = model(X_batch)
        preds = (outputs >= THRESHOLD).int()
        y_pred.extend(preds.cpu().numpy())

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
