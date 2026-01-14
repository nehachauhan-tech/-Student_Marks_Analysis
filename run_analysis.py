# Complete Student Performance Analysis Script
# This script runs the entire analysis and shows output

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

print('=' * 60)
print('🎓 STUDENT PERFORMANCE PREDICTION ANALYSIS')
print('=' * 60)

# ============================================
# 1. Load Dataset
# ============================================
print('\n📂 Loading Dataset...')
df = pd.read_csv('data/Student_Performance.csv')
print('✅ Dataset loaded successfully!')

# ============================================
# 2. Data Exploration (EDA)
# ============================================
print('\n' + '=' * 60)
print('📊 DATA EXPLORATION')
print('=' * 60)

print(f'\n📏 Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns')

print('\n📋 Column Names:')
for col in df.columns:
    print(f'   - {col}')

print('\n📈 First 5 Rows:')
print(df.head().to_string())

print('\n📈 Statistical Summary:')
print(df.describe().to_string())

print('\n❓ Missing Values:')
missing = df.isnull().sum()
print(missing)
print(f'\n✅ Total missing values: {missing.sum()}')

print(f'\n🔄 Duplicate rows: {df.duplicated().sum()}')

# ============================================
# 3. Data Visualization (Save plots to images folder)
# ============================================
print('\n' + '=' * 60)
print('📊 CREATING VISUALIZATIONS')
print('=' * 60)

# 3.1 Performance Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Performance Index'], kde=True, color='#3498db', bins=30)
plt.title('Distribution of Student Performance Index', fontsize=14, fontweight='bold')
plt.xlabel('Performance Index')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('images/performance_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print('✅ Saved: images/performance_distribution.png')

# 3.2 Correlation Heatmap
df_encoded = df.copy()
if 'Extracurricular Activities' in df_encoded.columns:
    df_encoded['Extracurricular Activities'] = df_encoded['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

plt.figure(figsize=(10, 8))
correlation = df_encoded.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, fmt='.2f')
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print('✅ Saved: images/correlation_heatmap.png')

# 3.3 Hours Studied vs Performance
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Hours Studied', y='Performance Index', alpha=0.6, color='#e74c3c')
plt.title('Hours Studied vs Performance Index', fontsize=14, fontweight='bold')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.tight_layout()
plt.savefig('images/hours_vs_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print('✅ Saved: images/hours_vs_performance.png')

# 3.4 Previous Scores vs Performance
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Previous Scores', y='Performance Index', alpha=0.6, color='#2ecc71')
plt.title('Previous Scores vs Performance Index', fontsize=14, fontweight='bold')
plt.xlabel('Previous Scores')
plt.ylabel('Performance Index')
plt.tight_layout()
plt.savefig('images/previous_scores_vs_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print('✅ Saved: images/previous_scores_vs_performance.png')

# 3.5 Extracurricular Activities Impact
if 'Extracurricular Activities' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Extracurricular Activities', y='Performance Index', 
                palette=['#3498db', '#e74c3c'])
    plt.title('Performance by Extracurricular Activities', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('images/extracurricular_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✅ Saved: images/extracurricular_impact.png')

# ============================================
# 4. Data Preprocessing
# ============================================
print('\n' + '=' * 60)
print('🔧 DATA PREPROCESSING')
print('=' * 60)

df_model = df.copy()
if 'Extracurricular Activities' in df_model.columns:
    df_model['Extracurricular Activities'] = df_model['Extracurricular Activities'].map({'Yes': 1, 'No': 0})
    print('✅ Encoded Extracurricular Activities: Yes=1, No=0')

# Define Features (X) and Target (y)
X = df_model.drop('Performance Index', axis=1)
y = df_model['Performance Index']

print(f'\n📊 Features shape: {X.shape}')
print(f'🎯 Target shape: {y.shape}')
print(f'\n📋 Features used: {list(X.columns)}')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'\n📚 Training set size: {X_train.shape[0]} samples')
print(f'🧪 Testing set size: {X_test.shape[0]} samples')

# ============================================
# 5. Model Training
# ============================================
print('\n' + '=' * 60)
print('🤖 MODEL TRAINING')
print('=' * 60)

model = LinearRegression()
model.fit(X_train, y_train)
print('✅ Linear Regression model trained successfully!')

print('\n📊 Model Coefficients:')
print('-' * 40)
for feature, coef in zip(X.columns, model.coef_):
    print(f'   {feature}: {coef:.4f}')
print(f'\n   Intercept: {model.intercept_:.4f}')

# ============================================
# 6. Model Evaluation
# ============================================
print('\n' + '=' * 60)
print('📈 MODEL EVALUATION')
print('=' * 60)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('\n📊 Performance Metrics:')
print('-' * 40)
print(f'   Mean Absolute Error (MAE): {mae:.4f}')
print(f'   Mean Squared Error (MSE): {mse:.4f}')
print(f'   Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'   R² Score: {r2:.4f} ({r2*100:.2f}%)')

# Actual vs Predicted Plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='#3498db')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Performance Index')
plt.ylabel('Predicted Performance Index')
plt.title('Actual vs Predicted Performance Index', fontsize=14, fontweight='bold')
plt.legend()
plt.tight_layout()
plt.savefig('images/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.close()
print('\n✅ Saved: images/actual_vs_predicted.png')

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='#9b59b6', bins=30)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('images/residuals_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print('✅ Saved: images/residuals_distribution.png')

# ============================================
# 7. Sample Prediction
# ============================================
print('\n' + '=' * 60)
print('🔮 SAMPLE PREDICTION')
print('=' * 60)

sample_student = pd.DataFrame({
    'Hours Studied': [7],
    'Previous Scores': [75],
    'Extracurricular Activities': [1],
    'Sleep Hours': [7],
    'Sample Question Papers Practiced': [5]
})

prediction = model.predict(sample_student)

print('\n📋 Student Details:')
print('   - Hours Studied: 7')
print('   - Previous Scores: 75')
print('   - Extracurricular Activities: Yes')
print('   - Sleep Hours: 7')
print('   - Sample Papers Practiced: 5')
print(f'\n🎯 Predicted Performance Index: {prediction[0]:.2f}')

# ============================================
# 8. Save Model
# ============================================
print('\n' + '=' * 60)
print('💾 SAVING MODEL')
print('=' * 60)

with open('models/linear_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print('✅ Model saved to: models/linear_regression_model.pkl')

# ============================================
# 9. Final Summary
# ============================================
print('\n' + '=' * 60)
print('📊 PROJECT SUMMARY')
print('=' * 60)
print(f'\n📁 Dataset: {df.shape[0]} students, {df.shape[1]} features')
print(f'🎯 Target Variable: Performance Index')
print(f'📈 Model: Linear Regression')
print(f'✅ Model Accuracy (R² Score): {r2*100:.2f}%')
print(f'📉 Average Error (MAE): {mae:.2f} points')

print('\n🔑 Key Insights:')
print('   1. Hours Studied has strong positive correlation with performance')
print('   2. Previous Scores are a good predictor of future performance')
print('   3. Extracurricular activities show positive impact')

print('\n📁 Generated Files:')
print('   - images/performance_distribution.png')
print('   - images/correlation_heatmap.png')
print('   - images/hours_vs_performance.png')
print('   - images/previous_scores_vs_performance.png')
print('   - images/extracurricular_impact.png')
print('   - images/actual_vs_predicted.png')
print('   - images/residuals_distribution.png')
print('   - models/linear_regression_model.pkl')

print('\n' + '=' * 60)
print('✅ ANALYSIS COMPLETED SUCCESSFULLY!')
print('=' * 60)
