import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load Data
df = pd.read_csv("fraudTest.csv")

# Step 2: Drop Unnecessary Columns
drop_cols = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'first', 'last',
             'street', 'city', 'state', 'zip', 'job', 'dob', 'trans_num']
df.drop(columns=drop_cols, inplace=True)

# Step 3: Encode Categorical Columns
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Step 4: Balance and Sample the Data
fraud_df = df[df['is_fraud'] == 1]
non_fraud_df = df[df['is_fraud'] == 0].sample(n=45000, random_state=42)
sampled_df = pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=42)

# Step 5: Split Features and Labels
X = sampled_df.drop(columns='is_fraud')
y = sampled_df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 6: Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluate the Model
y_pred = model.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
