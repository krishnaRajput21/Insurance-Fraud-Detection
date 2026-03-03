import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

np.random.seed(42)

# Data Generation
print("Generating dummy dataset...")
n_samples = 1000
fraud_reported = np.array(['Y'] * 247 + ['N'] * 753)
np.random.shuffle(fraud_reported)

months_as_customer = np.zeros(n_samples)
age = np.zeros(n_samples)
policy_annual_premium = np.zeros(n_samples)
injury_claim = np.zeros(n_samples)
property_claim = np.zeros(n_samples)
vehicle_claim = np.zeros(n_samples)

for i in range(n_samples):
    if fraud_reported[i] == 'Y':
        # Fraudulent claims: Tend to be newer customers, higher claims, higher premiums
        months_as_customer[i] = np.random.randint(0, 50)
        age[i] = np.random.randint(18, 40)
        policy_annual_premium[i] = np.random.uniform(2000, 3000)
        injury_claim[i] = np.random.randint(15000, 30000)
        property_claim[i] = np.random.randint(15000, 30000)
        vehicle_claim[i] = np.random.randint(40000, 80000)
    else:
        # Genuine claims: Longer tenure, normal claims
        months_as_customer[i] = np.random.randint(50, 500)
        age[i] = np.random.randint(30, 65)
        policy_annual_premium[i] = np.random.uniform(500, 1500)
        injury_claim[i] = np.random.randint(0, 15000)
        property_claim[i] = np.random.randint(0, 15000)
        vehicle_claim[i] = np.random.randint(0, 40000)

months_as_customer = months_as_customer.astype(int)
age = age.astype(int)
injury_claim = injury_claim.astype(int)
property_claim = property_claim.astype(int)
vehicle_claim = vehicle_claim.astype(int)

# Make total_claim_amount highly correlated with claims
total_claim_amount = injury_claim + property_claim + vehicle_claim

df = pd.DataFrame({
    'months_as_customer': months_as_customer,
    'age': age,
    'policy_annual_premium': policy_annual_premium,
    'injury_claim': injury_claim,
    'property_claim': property_claim,
    'vehicle_claim': vehicle_claim,
    'total_claim_amount': total_claim_amount,
    'fraud_reported': fraud_reported
})

df.to_csv('insurance_claims.csv', index=False)
print("Saved insurance_claims.csv")

# EDA & Preprocessing
print("\nPerforming EDA & Preprocessing...")
# Use IQR method to detect outliers in policy_annual_premium
Q1 = df['policy_annual_premium'].quantile(0.25)
Q3 = df['policy_annual_premium'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Capping outliers
df['policy_annual_premium'] = np.where(df['policy_annual_premium'] > upper_bound, upper_bound,
                                       np.where(df['policy_annual_premium'] < lower_bound, lower_bound, df['policy_annual_premium']))

# Log transformation
df['policy_annual_premium'] = np.log1p(df['policy_annual_premium'])

# Drop highly correlated features (>0.9)
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr().abs()

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
print(f"Dropping highly correlated columns: {to_drop}")
df.drop(columns=to_drop, inplace=True, errors='ignore')

# Handle LabelEncoding for object datatypes
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale using StandardScaler (Fit ONLY on training data to prevent leakage)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SMOTE to training data ONLY
print("Applying SMOTE to balance the dataset...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Original training shape: {X_train.shape}, {y_train.value_counts().to_dict()}")
print(f"Resampled training shape: {X_train_resampled.shape}, {pd.Series(y_train_resampled).value_counts().to_dict()}")

# Model Training
print("\nTraining and Evaluating Models...")

# Define parameter grid for SVM
svm_param_grid = {
    'C': [0.1, 1, 10], 
    'kernel': ['linear', 'rbf']
}

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(random_state=0),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(random_state=0),
    "Naïve Bayes": GaussianNB(),
    "SVM (GridSearch)": GridSearchCV(SVC(probability=True, random_state=0), svm_param_grid, cv=5, verbose=1, n_jobs=-1)
}

best_model = None
best_acc = 0

for name, model in models.items():
    print(f"\n--- {name} ---")
    
    # Train model
    model.fit(X_train_resampled, y_train_resampled)
    
    if name == "SVM (GridSearch)":
       print(f"Best Parameters for SVM: {model.best_params_}")
       # We use the best estimator from grid search for further evaluation
       eval_model = model.best_estimator_
    else:
       eval_model = model
       
    y_pred = eval_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    cv_scores = cross_val_score(eval_model, X_train_resampled, y_train_resampled, cv=5)
    
    print(f"Accuracy Score: {acc:.4f}")
    print(f"Cross Validation Score (cv=5): {cv_scores.mean():.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = eval_model

print(f"\nBest Model is {best_model.__class__.__name__} with Accuracy: {best_acc:.4f}")

# Save the best model and preprocessors
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
    
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("Saved model.pkl, scaler.pkl, and model_columns.pkl successfully.")
