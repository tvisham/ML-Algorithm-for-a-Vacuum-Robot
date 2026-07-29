import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('robot_energy_data.csv')

df['Battery_Decrease_Rate'] = pd.to_numeric(df['Battery_Decrease_Rate'], errors='coerce')
df['Acceleration_Decrease_Rate'] = pd.to_numeric(df['Acceleration_Decrease_Rate'], errors='coerce')

df = df.dropna(subset=['Battery_Decrease_Rate', 'Acceleration_Decrease_Rate'])

features = df.groupby('Experiment_ID').agg({
    'Battery_Decrease_Rate': ['mean', 'max', 'std'],
    'Acceleration_Decrease_Rate': ['mean', 'std'],
    'Battery_Percentage': ['first', 'last'],
    'Time': lambda x: df.loc[x.idxmax(), 'Time']
})

features.columns = ['_'.join(col).strip() for col in features.columns.values]
features.reset_index(inplace=True)

features['Total_Battery_Used'] = features['Battery_Percentage_first'] - features['Battery_Percentage_last']

labels = df[['Experiment_ID', 'Classification']].drop_duplicates()
final_df = pd.merge(features, labels, on='Experiment_ID')

le = LabelEncoder()
final_df['Classification'] = le.fit_transform(final_df['Classification'])

X = final_df.drop(columns=['Experiment_ID', 'Classification'])
y = final_df['Classification']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

