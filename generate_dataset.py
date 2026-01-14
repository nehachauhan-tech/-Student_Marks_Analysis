Script to generate Student Performance dataset

This creates a realistic dataset similar to the Kaggle dataset

import pandas as pd
import numpy as np


Set random seed for reproducibility
np.random.seed(42)

Number of students
n_students = 10000

 Generate features
hours_studied = np.random.randint(1, 10, n_students)   1-9 hours
previous_scores = np.random.randint(40, 100, n_students)  40-99 scores
extracurricular = np.random.choice(['Yes', 'No'], n_students)   Yes or No
sleep_hours = np.random.randint(4, 10, n_students)  4-9 hours
sample_papers = np.random.randint(0, 10, n_students)   0-9 papers

Calculate Performance Index based on features (with some randomness)

This creates a realistic relationship between features and target
performance_index = (
    hours_studied * 2.85 +  # Hours studied has high impact
    previous_scores * 1.01 +  # Previous scores matter
    np.where(extracurricular == 'Yes', 1, 0) * 0.5 +  # Small positive effect
    sleep_hours * 0.48 +  # Sleep helps a bit
    sample_papers * 0.67 +  # Practice helps
    np.random.normal(0, 2, n_students)  # Add some noise
)


 Normalize to 0-100 range
performance_index = (performance_index - performance_index.min()) / (performance_index.max() - performance_index.min()) * 100
performance_index = np.round(performance_index, 2)


 Create DataFrame
df = pd.DataFrame({
    'Hours Studied': hours_studied,
    'Previous Scores': previous_scores,
    'Extracurricular Activities': extracurricular,
    'Sleep Hours': sleep_hours,
    'Sample Question Papers Practiced': sample_papers,
    'Performance Index': performance_index
})


 Save to CSV
df.to_csv('data/Student_Performance.csv', index=False)

print('✅ Dataset created successfully!')
print(f'📊 Shape: {df.shape}')
print(f'📁 Saved to: data/Student_Performance.csv')
print('\nFirst 5 rows:')
print(df.head())
