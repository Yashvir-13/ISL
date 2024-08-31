import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the data
data_dict = pickle.load(open('data.pickle', 'rb'))

data_list = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Adjust this to handle 84 features (2 hands)
target_feature_count = 84

# Adjust features to match the target length (84)
def adjust_features(features, target_length):
    if len(features) > target_length:
        return features[:target_length]
    else:
        return np.pad(features, (0, target_length - len(features)), 'constant')

# Process the data to ensure consistent feature length
data = np.array([adjust_features(x, target_feature_count) for x in data_list])

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, y_encoded, test_size=0.2, shuffle=True, stratify=y_encoded, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# Make predictions and evaluate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f'{score * 100}% of samples were classified correctly!')

# Save the model and label encoder
with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model}, f)

with open('label_encoder.pkl', 'wb') as f:
    joblib.dump(label_encoder, f)

print(data.shape)
