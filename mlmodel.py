import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv(r"C:\Users\rahul\OneDrive\Desktop\AyurGuru\ML model data -Filtered.csv")
X = data.drop('class', axis=1)
y = data['class']
categorical_features = [col for col in X.columns if X[col].dtype == 'object']
encoder = OneHotEncoder(drop='first', sparse=False)
X_encoded = encoder.fit_transform(X[categorical_features])
X_numeric = pd.concat([X.drop(categorical_features, axis=1), pd.DataFrame(X_encoded)], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.25, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
new_individual = {
    'Gender': 'Male',
    'eye_Color': 'Black',
    'lips_Color': 'Dark',
    'hair_Color': 'Black',
    'skin_Color': 'Dark',
    'lips_Nature': 'Cracked',
    'skin_pimple': 'Pimples'
}

new_individual_df = pd.DataFrame([new_individual])

new_individual_encoded = encoder.transform(new_individual_df[categorical_features])
new_individual_numeric = pd.concat([new_individual_df.drop(categorical_features, axis=1), pd.DataFrame(new_individual_encoded)], axis=1)

prakriti_prediction = rf_classifier.predict(new_individual_numeric)

print("Predicted Prakriti:", prakriti_prediction[0])



