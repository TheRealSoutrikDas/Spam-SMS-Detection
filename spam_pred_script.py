import torch
import torch.nn as nn
import joblib
import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', text).strip()

class SpamClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super(SpamClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

vectorizer = joblib.load('tfidf_vectorizer.pkl')
input_dim = len(vectorizer.get_feature_names_out())

model = SpamClassifier(input_dim=input_dim)
model.load_state_dict(torch.load('spam_classifier.pt', map_location=torch.device('cpu')))
model.eval()


def predict_spam(message):
    clean = clean_text(message)
    tfidf_vector = vectorizer.transform([clean]).toarray()
    input_tensor = torch.tensor(tfidf_vector, dtype=torch.float32)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        return 'Spam' if predicted.item() == 1 else 'Ham'

print("-----------------------------")
print("My Spam Identifier!!!")
print("-----------------------------")
if __name__ == "__main__":
    msg = input("Enter SMS message: ")
    result = predict_spam(msg)
    print(f"\nPrediction: {result}")
