from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Processor,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
import numpy as np
import evaluate
import torch

def main():
    # Load the prepared dataset
    dataset = load_from_disk("your_data_path/prepared_dataset")

    # Define number of labels and load processor
    num_labels = len(set(dataset["train"]["label"]))  # Ensure labels are numeric
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    # Load Wav2Vec2 model for classification
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base",
        num_labels=num_labels,
    )

    # Freeze feature extractor layers (optional: unfreeze for larger datasets)
    model.freeze_feature_extractor()

    # Define metrics
    metric = evaluate.load("accuracy")  # For imbalanced datasets, consider F1-score

    def compute_metrics(pred):
        """
        Compute evaluation metrics.
        """
        predictions = np.argmax(pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=pred.label_ids)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="your_model_path/wav2vec2_finetuned",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,  # Adjusted for RTX 4090
        per_device_eval_batch_size=32,
        num_train_epochs=10,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="your_model_path/logs",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",  # Select best model based on validation accuracy
        dataloader_num_workers=8,  # Optimize data loading
        greater_is_better=True,
        fp16=torch.cuda.is_available(),  # Enable mixed precision if CUDA is available
        gradient_checkpointing=True,  # Useful for large models to save memory
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],  # Train split
        eval_dataset=dataset["test"],  # Test split
        tokenizer=processor,  # Use processor for tokenization
        compute_metrics=compute_metrics,
    )

    # Fine-tune the model
    print("Starting fine-tuning...")
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model("your_model_path/wav2vec2_finetuned_model")
    print("Model saved successfully!")

    # Evaluate the model on the test set
    print("Evaluating on the test set...")
    metrics = trainer.evaluate(dataset["test"])
    print(f"Test metrics: {metrics}")

if __name__ == "__main__":
    main()
