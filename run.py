import json
from dataclasses import dataclass
from transformers import (
    HfArgumentParser, 
    TrainingArguments,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)
import datasets
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase
from typing import Optional, List, Dict, Union
from utils import * # prepare_dataset_yn, convert_boolq_to_yn, convert_contrast_to_yn, convert_pubmed_to_yn, compute_metrics


@dataclass
class ModelArguments:
    model: str = "google/electra-small-discriminator"
    max_length: int = 128
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    dataset: Optional[str] = None
    datasubset: Optional[str] = None

def main():
    parser = HfArgumentParser((TrainingArguments, ModelArguments))
    training_args, model_args = parser.parse_args_into_dataclasses()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model,
        num_labels=2
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model)

    # Load dataset
    if model_args.dataset == 'boolq':
        dataset = datasets.load_dataset('boolq')
        dataset = convert_boolq_to_yn(dataset)
        eval_split = 'validation'
    elif model_args.dataset == 'contrast':
        with open('boolq_perturbed.json', 'r') as f:
            json_data = json.load(f)
        dataset = convert_bundles_to_yn(json_data)
        eval_split = 'train'
    elif model_args.dataset == 'qiaojin/PubMedQA':
        dataset = datasets.load_dataset(model_args.dataset, model_args.datasubset)
        if model_args.datasubset == 'pqa_labeled':
            with open('pubmed_test.json', 'r') as f:
                pubmed_test = json.load(f)
            dataset = convert_pubmed_to_yn(dataset, list(map(int, list(pubmed_test.keys()))))
            eval_split = 'test'
        else:
            dataset = convert_pubmed_to_yn(dataset, [])
            eval_split = 'train'
    elif model_args.dataset.endswith('.json') or model_args.dataset.endswith('.jsonl'):
        dataset = datasets.load_dataset('json', data_files=model_args.dataset)
        eval_split = 'validation' if 'validation' in dataset else 'test' if 'test' in dataset else 'train'
    elif model_args.datasubset:
        dataset = datasets.load_dataset(model_args.dataset, model_args.datasubset)
        eval_split = 'validation'
    else:
        dataset = datasets.load_dataset(model_args.dataset)
        eval_split = 'validation'

    train_dataset = None
    eval_dataset = None

    prepare_fn = prepare_dataset_yn
    trainer_class = Trainer
    if model_args.dataset == 'contrast':
        prepare_fn = prepare_bundle_features
        trainer_class = BundleTrainer
        training_args.remove_unused_columns=False
    
    if training_args.do_train:
        train_dataset = dataset['train']
        if model_args.max_train_samples:
            train_dataset = train_dataset.select(range(model_args.max_train_samples))
        
        train_dataset = train_dataset.map(
            lambda x: prepare_fn(x, tokenizer, model_args.max_length),
            batched=True,
            remove_columns=train_dataset.column_names
        )

    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if model_args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(model_args.max_eval_samples))
        
        eval_dataset = eval_dataset.map(
            lambda x: prepare_fn(x, tokenizer, model_args.max_length),
            batched=True,
            remove_columns=eval_dataset.column_names
        )


    # Initialize trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda eval_preds: compute_metrics(
                eval_preds,
                dataset[eval_split],
                output_dir=training_args.output_dir
            ),
        tokenizer=tokenizer,
    )
    
    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()

    # Evaluation
    if training_args.do_eval:
        trainer.evaluate()

if __name__ == "__main__":
    main()