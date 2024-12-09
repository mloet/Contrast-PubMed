import os
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
import evaluate
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

@dataclass
class ModelArguments:
    model: str = "google/electra-small-discriminator"
    max_length: int = 128
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    dataset: Optional[str] = None

def prepare_dataset_ynm(examples, tokenizer, max_length):
    """Convert yes/no/maybe examples to model inputs"""
    tokenized = tokenizer(
        examples['question'],
        examples['context'],
        truncation=True,
        max_length=max_length,
        padding='max_length'
    )
    
    label_map = {'yes': 0, 'no': 1}
    tokenized['labels'] = [label_map[label] for label in examples['label']]
    
    return tokenized

def convert_boolq_to_ynm(dataset):
    def convert_example(example):
        return {
            'question': example['question'],
            'context': example['passage'],
            'label': 'yes' if example['answer'] else 'no'
        }
    
    converted_dataset = {}
    for split in dataset.keys():
        converted_dataset[split] = dataset[split].map(convert_example)
    
    return datasets.DatasetDict(converted_dataset)

def convert_contrast_to_ynm(json_data):
    questions = []
    contexts = []
    labels = []
    
    for item in json_data['data']:
        questions.append(item['question'])
        contexts.append(item['paragraph'])
        labels.append('yes' if item['answer'].upper() == 'TRUE' else 'no')
        
        for perturbed in item['perturbed_questions']:
            questions.append(perturbed['perturbed_q'])
            contexts.append(item['paragraph'])
            labels.append('yes' if perturbed['answer'].upper() == 'TRUE' else 'no')
    
    return {
        'train': datasets.Dataset.from_dict({
            'question': questions[2:],
            'context': contexts[2:],
            'label': labels[2:]
        })
    }

def convert_pubmed_to_ynm(dataset):
    def convert_example(example):
        return {
            'question': example['question'],
            'context': ' '.join(example['context']['contexts']),
            'label': example['final_decision'],
            'pubid': example['pubid'],
            'original_answer': example['long_answer']
        }
    
    converted_dataset = {}
    for split in dataset.keys():
        # converted_dataset[split] = dataset[split].map(convert_example)

        ds = dataset[split].map(convert_example)
        converted_dataset[split] = ds.filter(lambda x: x['label'] != 'maybe')
        
        labels = converted_dataset[split]['label']
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"\nLabel distribution in {split} split:")
        for label, count in label_counts.items():
            print(f"{label}: {count} ({count/len(labels)*100:.2f}%)")
    
    return datasets.DatasetDict(converted_dataset)

def compute_metrics(eval_preds, dataset, output_dir: str = "evaluation_results"):
    predictions, labels = eval_preds
    # predictions = predictions.argmax(-1)
    predictions = np.array(predictions.argmax(-1))  # Ensure predictions are numpy arrays
    labels = np.array(labels)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert numeric predictions/labels back to text for analysis
    label_map = {0: 'yes', 1: 'no'}
    text_preds = [label_map[p] for p in predictions]
    text_labels = [label_map[l] for l in labels]
    print("Unique labels in dataset:", np.unique(labels))
    print("Label map:", label_map)
    
    # Initialize result dictionaries
    metrics = {}
    error_analysis = {
        'correct': [],
        'false_positives': [],
        'false_negatives': []
    }
    
    # Calculate overall metrics
    metrics['total_f1'] = f1_score(labels, predictions, average='macro')
    metrics['total_exact_match'] = np.mean(predictions == labels)
    
    # Calculate per-class metrics
    for class_idx, class_name in label_map.items():
        # Create binary arrays for this class
        true_binary = (labels == class_idx)
        pred_binary = (predictions == class_idx)
        
        # Calculate metrics
        metrics[f'{class_name}_f1'] = f1_score(true_binary, pred_binary)
        metrics[f'{class_name}_precision'] = precision_score(true_binary, pred_binary)
        metrics[f'{class_name}_recall'] = recall_score(true_binary, pred_binary)
        metrics[f'{class_name}_exact_match'] = np.mean(true_binary == True) #(predictions[labels == class_idx] == labels[labels == class_idx]).mean()
        
        # Count examples
        # metrics[f'{class_name}_total'] = true_binary.sum()
        # metrics[f'{class_name}_correct'] = (true_binary & pred_binary).sum()
    
    print(dataset)
    # Generate error analysis
    for i, (pred, label) in enumerate(zip(text_preds, text_labels)):
        example = {
            'question': dataset[i]['question'],
            'predicted_label': pred,
            'correct_label': label,
            'context': dataset[i]['context']
        }
        
        if pred == label:
            error_analysis['correct'].append(example)
        elif label == 'yes':  # Missed a yes (false negative)
            error_analysis['false_negatives'].append(example)
        else:  # Predicted yes when it was no (false positive)
            error_analysis['false_positives'].append(example)
    
    # Write error analysis files
    for category, examples in error_analysis.items():
        output_file = os.path.join(output_dir, f'{category}.jsonl')
        with open(output_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
    
    # Write summary metrics
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Total Examples: {len(labels)}")
    print(f"Overall F1: {metrics['total_f1']:.3f}")
    print(f"Overall Exact Match: {metrics['total_exact_match']:.3f}")
    print("\nPer-class metrics:")
    for class_name in label_map.values():
        print(f"\n{class_name.upper()}:")
        print(f"F1: {metrics[f'{class_name}_f1']:.3f}")
        print(f"Precision: {metrics[f'{class_name}_precision']:.3f}")
        print(f"Recall: {metrics[f'{class_name}_recall']:.3f}")
        print(f"Exact Match: {metrics[f'{class_name}_exact_match']:.3f}")
        # print(f"Correct: {metrics[f'{class_name}_correct']}/{metrics[f'{class_name}_total']}")
    
    return metrics

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
        dataset = convert_boolq_to_ynm(dataset)
        eval_split = 'validation'
    elif model_args.dataset == 'pubmed':
        dataset = datasets.load_dataset("qiaojin/PubMedQA", "pqa_artificial")
        dataset = convert_pubmed_to_ynm(dataset)
        eval_split = 'validation' if 'validation' in dataset else 'test' if 'test' in dataset else 'train'
    elif model_args.dataset == 'contrast':
        with open('boolq_perturbed.json', 'r') as f:
          json_data = json.load(f)
        dataset = convert_contrast_to_ynm(json_data)
        eval_split = 'validation' if 'validation' in dataset else 'test' if 'test' in dataset else 'train'
    elif model_args.dataset.endswith('.json') or model_args.dataset.endswith('.jsonl'):
        dataset = datasets.load_dataset('json', data_files=model_args.dataset)
        eval_split = 'train'  # If using custom jsonl, all data is in 'train' split
    else:
        # Load from Hugging Face datasets hub
        dataset = datasets.load_dataset(model_args.dataset)
        eval_split = 'validation'

    data_list = [example for example in dataset['train']]

    # Write to JSON file
    with open(f'datasets/{model_args.dataset}.json', 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)

    # Preprocess the datasets
    train_dataset = None
    eval_dataset = None
    
    if training_args.do_train:
        train_dataset = dataset['train']
        if model_args.max_train_samples:
            train_dataset = train_dataset.select(range(model_args.max_train_samples))
        
        train_dataset = train_dataset.map(
            lambda x: prepare_dataset_ynm(x, tokenizer, model_args.max_length),
            batched=True,
            remove_columns=train_dataset.column_names
        )

    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if model_args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(model_args.max_eval_samples))
        
        eval_dataset = eval_dataset.map(
            lambda x: prepare_dataset_ynm(x, tokenizer, model_args.max_length),
            batched=True,
            remove_columns=eval_dataset.column_names
        )

    # Initialize trainer
    trainer = Trainer(
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
        results = trainer.evaluate()
        print('Evaluation results:')
        print(results)

        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, 'eval_metrics.json'), 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()