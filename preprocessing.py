import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv('train.csv')

# 2. Basic info
print(f"Total samples: {len(df)}")
print("-" * 30)

# 3. Class distribution (summing the 1s for each label)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
print("Class counts:")
print(df[label_cols].sum())
print("-" * 30)

# 4. Comments with NO labels (Clean comments)
# A comment is clean if the sum of all label columns is 0
df['clean'] = (df[label_cols].sum(axis=1) == 0).astype(int)
print(f"Clean comments: {df['clean'].sum()}")
print(f"Toxic comments (at least one label): {len(df) - df['clean'].sum()}")

# 5. Multi-label analysis
# How many labels does each comment have?
df['label_count'] = df[label_cols].sum(axis=1)
print("-" * 30)
print("Distribution of number of labels per comment:")
print(df['label_count'].value_counts().sort_index())

# 6. Comment length analysis
df['comment_length'] = df['comment_text'].apply(lambda x: len(str(x).split()))
print("-" * 30)
print(f"Average word count: {df['comment_length'].mean():.2f}")
print(f"Max word count: {df['comment_length'].max()}")

# 7. Visualization
plt.figure(figsize=(10, 5))
sns.histplot(df['comment_length'], bins=50, kde=False)
plt.title('Distribution of Comment Lengths (in words)')
plt.show()

##I use iterative stratification library as it is best for this paetituclar task for multi label split of dataset 

from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Get the indices
for train_index, val_index in msss.split(df['comment_text'], df[label_cols]):
    train_df = df.iloc[train_index]
    val_df = df.iloc[val_index]

print("Train toxic ratio:", train_df['toxic'].mean())
print("Val toxic ratio:", val_df['toxic'].mean())

train_df.to_csv('train_split.csv', index=False)
val_df.to_csv('val_split.csv', index=False)


