# %% [markdown]
# # Fine-tuning Whisper for Stuttering Speech
# 
# This notebook demonstrates how to fine-tune the OpenAI Whisper model on the timestamped FluencyBank dataset, which contains stuttered speech. The process includes:
# 
# 1. Loading and preprocessing the CSV and audio data
# 2. Creating training and testing splits
# 3. Evaluating the base Whisper model's performance on stuttered speech
# 4. Fine-tuning Whisper using transfer learning
# 5. Evaluating the fine-tuned model

# %% [markdown]
# ## Setup and Dependencies
# 
# First, we need to install the necessary libraries.

# %%
# # Install required packages
# !pip install openai-whisper
# !pip install transformers
# !pip install datasets
# !pip install evaluate
# !pip install librosa
# !pip install jiwer
# !pip install accelerate
# !pip install soundfile
# !pip install tqdm

# %%
import os
import glob
import json
import random
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import whisper
import torch
# from tqdm.notebook import tqdm
from tqdm.auto import tqdm  # automatically picks best available tqdm (notebook or CLI)
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset, Audio
from jiwer import wer

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# %%
TRAIN_NUM=4

# %% [markdown]
# ## Data Loading and Preprocessing
# 
# First, let's load the CSV files and process them to create a dataset with audio segments and their corresponding transcriptions.

# %%
# Paths to data
csv_dir = os.path.expanduser("~/TimeStamped/cleaned_csvs")
audio_dir = os.path.expanduser("~/TimeStamped-wav/combined_wavs")

# Get all CSV files
csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
print(f"Found {len(csv_files)} CSV files.")

# %%
def extract_segments_from_csv(csv_file):
    import pandas as pd
    import os

    df = pd.read_csv(csv_file)
    segments = []

    # Extract base filename (without extension), e.g. "24fa_000_combined"
    base_name = os.path.splitext(os.path.basename(csv_file))[0]

    # Build the corresponding WAV file path
    wav_file = os.path.join(audio_dir, f"{base_name}.wav")

    if not os.path.exists(wav_file):
        print(f"Warning: Audio file {wav_file} not found. Skipping file {csv_file}.")
        return segments

    # Use the base_name as segid
    segid = base_name

    start_time = df['wordstart'].min()
    end_time = df['wordend'].max()

    words = df.sort_values('wordstart')['word'].tolist()
    transcript = ' '.join(words)

    has_stutter = any(df['fp'] == 1) or any(df['rp'] == 1) or any(df['rv'] == 1) or any(df['pw'] == 1)

    segments.append({
        'segid': segid,
        'wav_file': wav_file,
        'start_time': start_time,
        'end_time': end_time,
        'transcript': transcript,
        'has_stutter': has_stutter
    })

    return segments

# %%
# Extract segments from all CSV files
all_segments = []
for csv_file in tqdm(csv_files, desc="Processing CSV files"):
    segments = extract_segments_from_csv(csv_file)
    all_segments.extend(segments)

print(f"Total segments extracted: {len(all_segments)}")

# Display sample segments
print("\nSample segments:")
for segment in all_segments[:3]:
    print(f"Segment ID: {segment['segid']}")
    print(f"WAV file: {segment['wav_file']}")
    print(f"Time range: {segment['start_time']} - {segment['end_time']} seconds")
    print(f"Transcript: {segment['transcript']}")
    print(f"Has stutter: {segment['has_stutter']}")
    print("---")

# %% [markdown]
# ## Creating Train and Test Splits
# 
# Next, let's split the data into training and testing sets.

# %%
# Shuffle the segments
random.shuffle(all_segments)

# Split into train, validation, and test sets (70% train, 15% validation, 15% test)
total_segments = len(all_segments)
val_count = max(1, int(total_segments * 0.15))
test_count = max(1, int(total_segments * 0.15))
train_count = total_segments - val_count - test_count

# Create the splits
test_segments = all_segments[:test_count]
val_segments = all_segments[test_count:test_count + val_count]
train_segments = all_segments[test_count + val_count:]

print(f"Train segments: {len(train_segments)}")
print(f"Validation segments: {len(val_segments)}")
print(f"Test segments: {len(test_segments)}")

# %%
# # Split segments into train and test sets based on speaker ID
# train_segments = []
# test_segments = []

# for segment in processed_segments:
#     speaker_id = segment['segid'].split('_')[0]
#     if speaker_id in test_speakers:
#         test_segments.append(segment)
#     else:
#         train_segments.append(segment)

# print(f"Train segments: {len(train_segments)}")
# print(f"Test segments: {len(test_segments)}")

# %%
# Save the split information for later reference
split_info = {
    'train_segment_ids': [seg['segid'] for seg in train_segments],
    'val_segment_ids': [seg['segid'] for seg in val_segments],
    'test_segment_ids': [seg['segid'] for seg in test_segments],
    'train_count': len(train_segments),
    'val_count': len(val_segments),
    'test_count': len(test_segments)
}

with open(f'data_split_info-{TRAIN_NUM}.json', 'w') as f:
    json.dump(split_info, f, indent=2)

# %% [markdown]
# ## Prepare Datasets for HuggingFace
# 
# Now, let's prepare the data for training with the HuggingFace Transformers library.

# %%
def prepare_dataset(segments):
    data = {
        'audio': [],
        'text': []
    }
    
    for segment in segments:
        data['audio'].append(segment['wav_file'])
        data['text'].append(segment['transcript'])
    
    return data

# %%
# Prepare the train, validation, and test datasets
train_data = prepare_dataset(train_segments)
val_data = prepare_dataset(val_segments)
test_data = prepare_dataset(test_segments)

# Create HuggingFace datasets
train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)
test_dataset = Dataset.from_dict(test_data)

# Add audio loading capability
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
val_dataset = val_dataset.cast_column("audio", Audio(sampling_rate=16000))
test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=16000))

print(f"Train dataset: {train_dataset}")
print(f"Validation dataset: {val_dataset}")
print(f"Test dataset: {test_dataset}")

# %% [markdown]
# ## Evaluate Base Whisper Model
# 
# Let's first evaluate the base Whisper model on our test set to establish a baseline performance.

# %%
# Function to evaluate the base Whisper model
def evaluate_base_whisper(test_dataset, model_size="base"):
    # Load the Whisper model
    model = whisper.load_model(model_size)
    print(f"Loaded Whisper {model_size} model.")
    
    # Extract audio paths and transcripts from the dataset
    audio_paths = test_dataset['audio']
    references = test_dataset['text']
    
    hypotheses = []
    
    # Process each audio file
    for idx, audio_path in enumerate(tqdm(audio_paths, desc=f"Evaluating {model_size} Whisper")):
        try:
            # Get the actual file path from the dataset info
            audio_file = audio_path['path'] if isinstance(audio_path, dict) else audio_path
            
            # Transcribe the audio with explicit settings
            result = model.transcribe(
                audio_file,
                language='en',  # Force English language
                task='transcribe'  # Ensure transcription instead of translation
            )
            
            # Get the transcription
            transcription = result["text"].strip()
            hypotheses.append(transcription)
            
            # Print progress every 10 files
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(audio_paths)} files")
                print(f"Reference: {references[idx]}")
                print(f"Hypothesis: {transcription}")
                print("---")
                
        except Exception as e:
            print(f"Error processing file {audio_file}: {e}")
            hypotheses.append("")
    
    # Calculate WER
    error_rate = wer(references, hypotheses)
    
    # Save results
    results = {
        'references': references,
        'hypotheses': hypotheses,
        'wer': error_rate
    }
    
    with open(f'base_whisper_{model_size}_results.json-{TRAIN_NUM}', 'w') as f:
        json.dump(results, f, indent=2)
    
    return error_rate, results

# %%
# Evaluate the base Whisper model
model_size="small" #changing to small to see better results
# model_size="base" #FIXME
base_wer, base_results = evaluate_base_whisper(test_dataset, model_size)
print(f"{model_size} Whisper WER: {base_wer:.4f}")

# %% [markdown]
# ## Fine-tune Whisper
# 
# Now, let's fine-tune the Whisper model on our stuttering speech dataset using the HuggingFace Transformers library.

# %%
# Load the Whisper processor and model for fine-tuning
# model_id = "openai/whisper-base"
model_id = "openai/whisper-small" #FIXME - change model based on ability
# model_id = "openai/whisper-medium"
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)

# %%
# Function to prepare the dataset for fine-tuning
def prepare_dataset_for_finetuning(dataset, processor):
    # Define a preprocessing function
    def prepare_example(example):
        # Load and resample the audio data
        audio = example["audio"]
        
        # Process the audio input
        input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features[0]
        
        # Process the text output
        example["labels"] = processor(text=example["text"]).input_ids
        example["input_features"] = input_features
        
        return example
    
    # Process the dataset
    processed_dataset = dataset.map(prepare_example, remove_columns=["audio", "text"])
    
    return processed_dataset

# %%
# Prepare the datasets for fine-tuning
print("Preparing train dataset...")
processed_train_dataset = prepare_dataset_for_finetuning(train_dataset, processor)

print("Preparing validation dataset...")
processed_val_dataset = prepare_dataset_for_finetuning(val_dataset, processor)

print("Preparing test dataset...")
processed_test_dataset = prepare_dataset_for_finetuning(test_dataset, processor)

print(f"Processed train dataset: {processed_train_dataset}")
print(f"Processed validation dataset: {processed_val_dataset}")
print(f"Processed test dataset: {processed_test_dataset}")

# %% [markdown]
# ## Training Iterations - Including for Insight into Progression

# %%
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./whisper-fine-tuned-stuttering",
#     per_device_train_batch_size=4,  # Reduced further from 8 to 4
#     gradient_accumulation_steps=4,  # Increased from 2 to 4 to maintain effective batch size
#     learning_rate=1e-5,
#     warmup_steps=125,
#     max_steps=1000,
#     gradient_checkpointing=True,  # Already enabled - saves memory
#     fp16=True,  # Already enabled - uses half precision
#     eval_strategy="steps",
#     eval_steps=125,
#     save_strategy="steps",
#     save_steps=125,
#     logging_steps=50,
#     report_to=["tensorboard"],
#     load_best_model_at_end=True,
#     metric_for_best_model="wer",
#     greater_is_better=False,
#     push_to_hub=False,
#     dataloader_num_workers=0,  # Avoid multiprocessing overhead
#     optim="adafactor",  # More memory-efficient optimizer
#     eval_accumulation_steps=8,  # Accumulate gradients during evaluation
#     prediction_loss_only=False,
#     group_by_length=True,  # Efficient batching by length
#     label_smoothing_factor=0.1,  # Regularization that can help training
# )

# # Additional memory optimizations
# model.config.use_cache = False  # Disable KV cache
# torch.cuda.empty_cache()  # Clear CUDA cache before training

# %%
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./whisper-fine-tuned-stuttering-test",
#     per_device_train_batch_size=1,          # small batch for quick test
#     gradient_accumulation_steps=1,
#     learning_rate=1e-4,                     # slightly higher LR for quick learning
#     warmup_steps=5,
#     max_steps=100,                           # 🔥 just 10 steps
#     eval_strategy="no",                     # skip evaluation for test run
#     save_strategy="no",                     # no saving checkpoints
#     logging_steps=1,
#     gradient_checkpointing=False,
#     fp16=torch.cuda.is_available(),         # only use fp16 if CUDA is available
#     report_to=[],                           # no TensorBoard in test
#     push_to_hub=False
# )

# %%
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./whisper-fine-tuned-stuttering-test",
#     per_device_train_batch_size=2,
#     gradient_accumulation_steps=1,
#     learning_rate=1e-4,
#     warmup_steps=100,
#     max_steps=500,  
#     eval_strategy="no",
#     logging_steps=10,
#     gradient_checkpointing=False,
#     save_strategy="no",
#     report_to=[],
#     push_to_hub=False,
#     fp16=torch.cuda.is_available()
# )


# %%
# # Training arguments optimized for heavy training without evaluation **DOING GREAT!
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./whisper-fine-tuned-stuttering2",
#     per_device_train_batch_size=4,          # Larger batch size for faster training
#     gradient_accumulation_steps=2,          # Effective batch size of 8
#     learning_rate=5e-5,
#     warmup_steps=200,
#     max_steps=1000,                         # More steps for thorough training
#     eval_strategy="no",                     # Disable evaluation completely
#     # save_strategy="steps",                  
#     # save_steps=1000,                        # Save checkpoints less frequently
#     logging_steps=20,                       # Regular logging for monitoring
#     # gradient_checkpointing=True,            # Enable for memory efficiency
#     fp16=torch.cuda.is_available(),                             # Use mixed precision
#     # dataloader_num_workers=4,              # Use multiple workers for data loading
#     optim="adamw_torch",                   
#     push_to_hub=False,
#     # save_total_limit=3,                    # Keep only 3 most recent checkpoints
#     report_to=["tensorboard"],             # Still log to tensorboard for monitoring
#     prediction_loss_only=True,             # Only compute loss, not other metrics
#     # group_by_length=True,                  # Efficient batching
#     # label_smoothing_factor=0.1,            # Regularization
# )

# # Memory optimizations
# model.config.use_cache = False
# torch.cuda.empty_cache()

# %% [markdown]
# # MAKE SURE TO CHANGE TRAIN_NUM
# this is main training block

# %%
# Training arguments optimized for heavy training without evaluation BEST MODEL
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./whisper-fine-tuned-stuttering-{TRAIN_NUM}",
    # per_device_train_batch_size=8,          # changed this
    per_device_train_batch_size=4,          # changed to 4 for medium model!
    gradient_accumulation_steps=4,          # Changed this
    learning_rate=5e-5,
    warmup_steps=500,                       # Changed this
    max_steps=2000,                         # More steps for thorough training
    eval_strategy="no",                     # Disable evaluation completely
    # save_strategy="steps",                  
    # save_steps=1000,                        
    logging_steps=40,                       # changed
    # gradient_checkpointing=True,          
    fp16=torch.cuda.is_available(),                            
    # dataloader_num_workers=4,         
    optim="adamw_torch",                   
    push_to_hub=False,
    # save_total_limit=3,                    
    report_to=["tensorboard"],        
    prediction_loss_only=True,          
    # group_by_length=True,             
    # label_smoothing_factor=0.1,        
)

# Memory optimizations
model.config.use_cache = False
torch.cuda.empty_cache()

# %%
# # Enhanced training arguments for potentially better results with ~700 clips dataset
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./whisper-fine-tuned-stuttering-final-{TRAIN_NUM}",
#     # Batch size and accumulation
#     per_device_train_batch_size=4,           # Good balance for available memory
#     gradient_accumulation_steps=4,           # Effective batch size of 16
    
#     # Learning rate schedule
#     learning_rate=2e-5,                      # Slightly lower learning rate for stability
#     lr_scheduler_type="cosine",              # Cosine schedule works well for speech models
#     warmup_ratio=0.1,                        # Warmup over 10% of training
    
#     # Training length
#     max_steps=3000,                          # More steps for a dataset of this size
    
#     # Regularization
#     weight_decay=0.01,                       # Help prevent overfitting (for small dataset)
    
#     # No evaluation to save memory
#     eval_strategy="no",
    
#     # Memory settings
#     fp16=torch.cuda.is_available(),
#     fp16_full_eval=False,
#     gradient_checkpointing=True,             # Enable for memory efficiency
    
#     # Logging
#     logging_steps=50,                        # Log progress more frequently
#     # save_strategy="steps",                   # Save at regular intervals
#     # save_steps=500,                          # Save every 500 steps
#     # save_total_limit=3,                      # Keep only 3 most recent checkpoints
    
#     # Better mixing of samples
#     # group_by_length=True,                    # Group similar length audios together
#     # length_column_name="input_features",     # Group based on input features
    
#     # Reporting
#     report_to=["tensorboard"],
#     prediction_loss_only=True,
#     push_to_hub=False,
    
#     # Optimize dataloader
#     # dataloader_num_workers=0,                # Avoid multiprocessing issues
#     # dataloader_pin_memory=True,              # Faster data transfer to GPU
# )

# # Memory optimizations
# model.config.use_cache = False
# torch.cuda.empty_cache()

# %%
# Define the data collator
class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        # Split inputs and labels since they need to be treated differently
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # Convert input features to tensors
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Convert labels to tensors with appropriate padding
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos_token_id exists, remove it
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# Create the data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# %%
# Define the compute metrics function
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    # Convert ids to strings
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    # Compute WER
    error = wer(label_str, pred_str)
    
    return {"wer": error}

# %%
# Initialize the trainer with the callback
print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"GPU available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

# %%
# Train the model with progress monitoring
print("Starting training with progress monitoring...")
print(f"Number of training samples: {len(processed_train_dataset)}")  # ADDED
print(f"Number of validation samples: {len(processed_val_dataset)}")  # ADDED

# Train the model
trainer.train()

# %%
# Save the fine-tuned model
trainer.save_model(f"./whisper-fine-tuned-stuttering-final-{TRAIN_NUM}")
processor.save_pretrained(f"./whisper-fine-tuned-stuttering-final-{TRAIN_NUM}")

# %% [markdown]
# ## Evaluate Fine-tuned Model
# 
# Now, let's evaluate the fine-tuned model on our test set to see how it performs compared to the base model.

# %%
# Load the fine-tuned model
fine_tuned_processor = WhisperProcessor.from_pretrained(f"./whisper-fine-tuned-stuttering-final-{TRAIN_NUM}")
fine_tuned_model = WhisperForConditionalGeneration.from_pretrained(f"./whisper-fine-tuned-stuttering-final-{TRAIN_NUM}")

# %%
# Function to evaluate the fine-tuned model
def evaluate_fine_tuned_whisper(test_dataset, processor, model, test=1):
    # Extract audio paths and transcripts from the dataset
    audio_paths = test_dataset['audio']
    references = test_dataset['text']
    
    hypotheses = []
    
    # Process each audio file
    for idx, audio_path in enumerate(tqdm(audio_paths, desc="Evaluating fine-tuned Whisper")):
        try:
            # Get the actual file path from the dataset info
            audio_file = audio_path['path'] if isinstance(audio_path, dict) else audio_path
            
            # Load audio
            audio, sr = librosa.load(audio_file, sr=16000)
            
            # Process audio for input
            input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
            
            # Clear the forced_decoder_ids and set language and task
            model.generation_config.forced_decoder_ids = None

            # Generate transcription with explicit arguments
            predicted_ids = model.generate(
                input_features,
                language='en',
                task='transcribe',
                use_cache=True
            )
            
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            hypotheses.append(transcription)
            
            # Print progress every 10 files
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(audio_paths)} files")
                print(f"Reference: {references[idx]}")
                print(f"Hypothesis: {transcription}")
                print("---")
                
        except Exception as e:
            print(f"Error processing file {audio_file}: {e}")
            hypotheses.append("")
    
    # Calculate WER
    error_rate = wer(references, hypotheses)
    
    # Save results
    results = {
        'references': references,
        'hypotheses': hypotheses,
        'wer': error_rate
    }
    
    if test==1:
        with open(f'fine_tuned_whisper_results_test-{TRAIN_NUM}.json', 'w') as f:
            json.dump(results, f, indent=2)
    elif test==0:
        with open(f'fine_tuned_whisper_results_val-{TRAIN_NUM}.json', 'w') as f:
            json.dump(results, f, indent=2)       
    
    return error_rate, results

# %%
# Evaluate the fine-tuned model
fine_tuned_wer, fine_tuned_results = evaluate_fine_tuned_whisper(test_dataset, fine_tuned_processor, fine_tuned_model, test=1)
print(f"Fine-tuned Whisper WER: {fine_tuned_wer:.4f}")

# %% [markdown]
# ## Compare Results
# 
# Let's compare the performance of the base and fine-tuned models.

# %%
print(f"Base Whisper WER: {base_wer:.4f}")
print(f"Fine-tuned Whisper WER: {fine_tuned_wer:.4f}")
print(f"Improvement: {(base_wer - fine_tuned_wer) / base_wer * 100:.2f}%")

# %% [markdown]
# ## Additional Analysis: Validation Set Performance
#  
# Let's also check how the model performed on the validation set during training.

# %%
# Evaluate the fine-tuned model on the validation set
print("Evaluating on validation set...")
val_wer, val_results = evaluate_fine_tuned_whisper(val_dataset, fine_tuned_processor, fine_tuned_model, test=0)
print(f"Fine-tuned Whisper WER on validation set: {val_wer:.4f}")

# %% [markdown]
# ## Summary of Performance
# 
# Let's create a summary of the model's performance across all datasets.

# %%
# Create performance summary
performance_summary = {
    'base_model': {
        'test_wer': base_wer
    },
    'fine_tuned_model': {
        'validation_wer': val_wer,
        'test_wer': fine_tuned_wer
    },
    'improvement': {
        'absolute': base_wer - fine_tuned_wer,
        'relative': (base_wer - fine_tuned_wer) / base_wer * 100
    }
}

# Save performance summary
with open(f'performance_summary-{TRAIN_NUM}.json', 'w') as f:
    json.dump(performance_summary, f, indent=2)

# Print summary
print("\nPerformance Summary:")
print(f"Base Model Test WER: {base_wer:.4f}")
print(f"Fine-tuned Model Validation WER: {val_wer:.4f}")
print(f"Fine-tuned Model Test WER: {fine_tuned_wer:.4f}")
print(f"Improvement: {(base_wer - fine_tuned_wer) / base_wer * 100:.2f}%")


