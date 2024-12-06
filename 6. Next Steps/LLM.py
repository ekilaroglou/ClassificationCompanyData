from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import wandb
import os
from time import time
from tools.basic import get_data

# Initialize WandB (Weights & Biases)
# Set to offline mode for local debugging or disable WandB entirely
wandb.init(project="my_project", mode="offline")
os.environ["WANDB_DISABLED"] = "true"

# Step 1: Load data from SQLite3
df = get_data()

# Step 2: Combine Relevant Features into a Single Text Column
# Function to combine row data into a single text string for model input
def row_to_text(row):
    return f"Year Founded: {row['year_founded']}; Country: {row['country']}; Employee Range: {row['employee_range']}; " \
           f"Industry: {row['industry']}; Homepage Text: {row['homepage_text']}; Homepage Keywords: {row['homepage_keywords']}; " \
           f"Meta Description: {row['meta_description']}"

df['text'] = df.apply(row_to_text, axis=1)

# Step 3: Encode Target Labels
# Use a label encoder to transform categorical labels into numerical values
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['category'])

# Step 4: Split Data into Training and Test Sets
# Reserve 20% of the data for testing
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Step 5: Tokenize Text Data Using DistilBERT Tokenizer
# Load DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Define a function for tokenization
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

# Convert data to Hugging Face Dataset format
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

# Time tokenization process
t0 = time()
print('Tokenize...')

# Apply tokenization to training and testing datasets
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

t1 = time()
print(f'Tokenization completed in {int(t1 - t0)} seconds.')

# Format datasets for PyTorch
train_dataset = train_dataset.with_format("torch")
test_dataset = test_dataset.with_format("torch")

# Step 6: Load Pre-Trained DistilBERT Model
# Define the number of output labels
num_labels = len(label_encoder.classes_)

# Load DistilBERT model with the specified number of output labels
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# Step 7: Set Training Arguments
# Configure training parameters
training_args = TrainingArguments(
    output_dir="./results",               # Directory to save model checkpoints
    evaluation_strategy="epoch",         # Evaluate model after each epoch
    save_strategy="epoch",               # Save model checkpoints after each epoch
    logging_dir="./logs",                # Directory for training logs
    num_train_epochs=3,                  # Number of epochs
    per_device_train_batch_size=4,       # Batch size per device for training
    per_device_eval_batch_size=4,        # Batch size per device for evaluation
    learning_rate=5e-5,                  # Learning rate
    weight_decay=0.01,                   # Weight decay for regularization
    save_total_limit=2,                  # Limit on the number of saved checkpoints
    gradient_accumulation_steps=4,       # Accumulate gradients to simulate a larger batch size
    load_best_model_at_end=True          # Load the best model after training
)

# Step 8: Initialize the Trainer
# Define the trainer with model, arguments, and datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Step 9: Train the Model
t2 = time()
print('Training started...')
trainer.train()
t3 = time()
print(f'Training completed in {int(t3 - t2)} seconds.')

# Step 10: Evaluate the Model
print('Evaluating the model...')
results = trainer.evaluate()
t4 = time()
print(f'Evaluation completed in {int(t4 - t3)} seconds.')

# Display evaluation results
print("Evaluation results:", results)
