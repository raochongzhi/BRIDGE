import argparse
from pjfdataset import UserJobMatchingDataset
from pjfmodel import UserJobMatchingModel
import torch
import torch.optim as optim
from transformers import Trainer, TrainingArguments
from metrics import compute_metrics
from transformers import get_linear_schedule_with_warmup
from getdataset import getDataset
import os
import shutil


def train_model(args):
    if torch.cuda.is_available():
        print('CUDA available')
    else:
        print('CUDA not available')

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        max_grad_norm=1.0,
        metric_for_best_model='combined_score',
        greater_is_better=True,
        report_to="tensorboard",
    )

    train_data = getDataset(args.train_dir, args.user_dir, args.job_dir)
    val_data = getDataset(args.val_dir, args.user_dir, args.job_dir)

    for idx, data in enumerate(train_data):
        if isinstance(data, torch.Tensor):
            assert not torch.isnan(data).any(), f"Train data at index {idx} contains NaN"

    for idx, data in enumerate(val_data):
        if isinstance(data, torch.Tensor):
            assert not torch.isnan(data).any(), f"Eval data at index {idx} contains NaN"

    train_dataset = UserJobMatchingDataset(train_data, args.max_history_length, args.d_model1)
    val_dataset = UserJobMatchingDataset(val_data, args.max_history_length, args.d_model1)

    model = UserJobMatchingModel(args.d_model1, args.num_heads1, args.dropout1,
                                 args.d_model2, args.num_heads2, args.dropout2,
                                 args.dropout3, args.model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    num_training_steps = len(train_dataset) // args.per_device_train_batch_size * args.num_train_epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
    )

    trainer.train()

    train_metrics = trainer.evaluate(eval_dataset=train_dataset)
    print("Train Metrics:", train_metrics)
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    print("Val Metrics:", val_metrics)

    trainer.save_model(args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a User-Job Matching Model")
    parser.add_argument("--d_model1", type=int, default=4096, help="Dimension of user model")
    parser.add_argument("--d_model2", type=int, default=4096, help="Dimension of job model")
    parser.add_argument("--num_heads1", type=int, default=4, help="Number of attention heads for user model")
    parser.add_argument("--num_heads2", type=int, default=4, help="Number of attention heads for job model")
    parser.add_argument("--dropout1", type=float, default=0.2, help="Dropout rate for user model")
    parser.add_argument("--dropout2", type=float, default=0.2, help="Dropout rate for job model")
    parser.add_argument("--dropout3", type=float, default=0.1, help="Dropout rate for the final layer")
    parser.add_argument("--max_history_length", type=int, default=10, help="Max history length")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training")
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64, help="Evaluation batch size per device")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Number of steps between logging")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Save train data path")
    parser.add_argument("--output_dir", type=str, default="results", help="Save output path")
    parser.add_argument("--train_dir", type=str, default="train.pt", help="Train dataset path")
    parser.add_argument("--val_dir", type=str, default="train.pt", help="Val dataset path")
    parser.add_argument("--user_dir", type=str, default="train.pt", help="user embedding dataset path")
    parser.add_argument("--job_dir", type=str, default="train.pt", help="job embedding dataset path")
    parser.add_argument("--model_type", type=str, default="model1", help="Number of model type")
    args = parser.parse_args()

    train_model(args)
