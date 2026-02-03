# =====================================
# FINAL SAMPLING ASSIGNMENT WITH RESULTS
# =====================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Sampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

# ------------------------------
# Load Dataset
# ------------------------------
url = "https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv"
df = pd.read_csv(url)

print("\nClass Distribution Before Sampling:\n")
print(df['Class'].value_counts())

# ------------------------------
# Features & Target
# ------------------------------
X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# Sampling Techniques
# ------------------------------
sampling_methods = {
    "Random UnderSampling": RandomUnderSampler(random_state=42),
    "Random OverSampling": RandomOverSampler(random_state=42),
    "SMOTE": SMOTE(k_neighbors=2, random_state=42),
    "ADASYN": ADASYN(n_neighbors=2, random_state=42)
}

# ------------------------------
# Models
# ------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=1
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=1),
    "SVM": SVC(kernel='rbf')
}

results = []

# ------------------------------
# Train Models
# ------------------------------
for sampling_name, sampler in sampling_methods.items():

    X_res, y_res = sampler.fit_resample(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res,
        y_res,
        test_size=0.2,
        stratify=y_res,
        random_state=42
    )

    for model_name, model in models.items():

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        results.append([sampling_name, model_name, round(acc*100, 2)])

# ------------------------------
# Stratified Sampling
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

for model_name, model in models.items():

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    results.append(["Stratified Sampling", model_name, round(acc*100, 2)])

# ------------------------------
# Create Results DataFrame
# ------------------------------
results_df = pd.DataFrame(
    results,
    columns=["Sampling Technique", "Model", "Accuracy (%)"]
)

pivot_table = results_df.pivot(
    index="Model",
    columns="Sampling Technique",
    values="Accuracy (%)"
)

print("\n🔥 FINAL ACCURACY TABLE:\n")
print(pivot_table)

# ------------------------------
# Save Files ⭐⭐⭐
# ------------------------------

# Save raw results
results_df.to_csv("sampling_results.csv", index=False)

# Save pivot table
pivot_table.to_csv("accuracy_table.csv")

# Save Excel (VERY PROFESSIONAL)
with pd.ExcelWriter("sampling_results.xlsx") as writer:
    results_df.to_excel(writer, sheet_name="Raw Results", index=False)
    pivot_table.to_excel(writer, sheet_name="Accuracy Table")

# ------------------------------
# Best Sampling Per Model
# ------------------------------
best_results = pivot_table.idxmax(axis=1).reset_index()
best_results.columns = ["Model", "Best Sampling Technique"]

best_results.to_csv("best_sampling_per_model.csv", index=False)

print("\n✅ Files Saved Successfully:")
print("✔ sampling_results.csv")
print("✔ accuracy_table.csv")
print("✔ sampling_results.xlsx")
print("✔ best_sampling_per_model.csv")
