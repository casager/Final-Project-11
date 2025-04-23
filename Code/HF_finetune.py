from datasets import Dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

def fine_tune_whisper(dataset):
    """
    Fine-tune the Whisper model on stuttering data.
    
    Args:
        dataset: The prepared dataset
    
    Returns:
        The fine-tuned model and processor
    """
    # Convert to HuggingFace dataset
    hf_dataset = Dataset.from_list(dataset)
    
    # Split dataset
    hf_dataset = hf_dataset.train_test_split(test_size=0.1)
    
    # Load processor and model
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-fine-tuned-stuttering",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )
    
    # Preprocess the data
    def prepare_dataset(examples):
        audio_arrays = [
            whisper.load_audio(example["audio"]) 
            for example in examples["audio"]
        ]
        inputs = processor(
            audio_arrays, 
            sampling_rate=16000, 
            return_tensors="pt"
        )
        with processor.as_target_processor():
            labels = processor(examples["text"]).input_ids
        return {
            "input_features": inputs.input_features, 
            "labels": labels
        }
    
    # Prepare datasets
    train_dataset = hf_dataset["train"].map(
        prepare_dataset, 
        batched=True, 
        remove_columns=hf_dataset["train"].column_names
    )
    eval_dataset = hf_dataset["test"].map(
        prepare_dataset, 
        batched=True, 
        remove_columns=hf_dataset["test"].column_names
    )
    
    # Define trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train model
    trainer.train()
    
    # Save model
    model.save_pretrained("./whisper-fine-tuned-stuttering-final")
    processor.save_pretrained("./whisper-fine-tuned-stuttering-final")
    
    return model, processor