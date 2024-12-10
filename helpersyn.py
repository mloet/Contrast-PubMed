import os
import json
import datasets
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def prepare_dataset_yn(examples, tokenizer, max_length):
    """Convert yes/no examples to model inputs"""
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

def convert_boolq_to_yn(dataset):
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

def convert_contrast_to_yn(json_data):
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

def convert_pubmed_to_yn(dataset):
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
    
    predictions = np.array(predictions.argmax(-1))  
    labels = np.array(labels)
    
    os.makedirs(output_dir, exist_ok=True)
    
    label_map = {0: 'yes', 1: 'no'}
    text_preds = [label_map[p] for p in predictions]
    text_labels = [label_map[l] for l in labels]
    print("Unique labels in dataset:", np.unique(labels))
    print("Label map:", label_map)
    
    metrics = {}
    error_analysis = {
        'correct': [],
        'false_positives': [],
        'false_negatives': []
    }
    
    # Calculate overall metrics
    metrics['total_f1'] = f1_score(labels, predictions, average='macro')
    metrics['total_exact_match'] = np.mean(predictions == labels)
    metrics['total_precision'] = precision_score(labels, predictions, average='macro')
    metrics['total_recall'] = recall_score(labels, predictions, average='macro')
    
    # Calculate per-class metrics
    for class_idx, class_name in label_map.items():
        true_binary = (labels == class_idx)
        pred_binary = (predictions == class_idx)
        
        metrics[f'{class_name}_f1'] = f1_score(true_binary, pred_binary)
        metrics[f'{class_name}_precision'] = precision_score(true_binary, pred_binary)
        metrics[f'{class_name}_recall'] = recall_score(true_binary, pred_binary)
    
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
    
    return metrics

