from datasets import load_dataset
import pandas as pd

# Load the dataset from HuggingFace
print("Downloading dataset...")
dataset = load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge", "default")

# Extract the train split and convert to CSV
print("Converting to CSV...")
train_df = dataset['train'].to_pandas()

# Save to train.csv
train_df.to_csv('train.csv', index=False)

print(f"\n✓ Downloaded {len(train_df)} samples")
print(f"✓ Saved to train.csv")
print(f"\nColumns: {list(train_df.columns)}")
print(f"\nLabel distribution:")
print(train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].sum())
