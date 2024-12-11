import os
import json
import datasets
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import Trainer
import torch
import torch.nn.functional as F


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

def prepare_bundle_features(examples, tokenizer, max_length):
    """Prepare features for bundle training"""
    features = tokenizer(
        examples['question'],
        examples['context'],
        max_length=max_length,
        truncation=True,
        padding='max_length'
    )
    
    label_map = {'yes': 0, 'no': 1}
    features['labels'] = [label_map[label] for label in examples['label']]
    features['bundle_ids'] = examples['bundle_id']
    
    return features

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

def convert_bundles_to_yn(json_data):
    questions = []
    contexts = []
    labels = []
    bundle_ids = []
    
    id = 0
    for item in json_data['data']:
        questions.append(item['question'])
        contexts.append(item['paragraph'])
        labels.append('yes' if item['answer'].upper() == 'TRUE' else 'no')
        bundle_ids.append(id)
        for perturbed in item['perturbed_questions']:
            questions.append(perturbed['perturbed_q'])
            contexts.append(item['paragraph'])
            labels.append('yes' if perturbed['answer'].upper() == 'TRUE' else 'no')
            bundle_ids.append(id)
        id+=1
    return {
        'train': datasets.Dataset.from_dict({
            'question': questions[2:],
            'context': contexts[2:],
            'label': labels[2:],
            'bundle_id': bundle_ids[2:]
        })
    }

def convert_pubmed_to_yn(dataset, test_split):
    def convert_example(example):
        return {
            'question': example['question'],
            'context': ' '.join(example['context']['contexts']),
            'label': example['final_decision'],
            'pubid': example['pubid']
        }

    examples = dataset['train'].map(convert_example)
    examples = examples.filter(lambda x: x['label'] != 'maybe')

    converted_dataset = {}
    if test_split:
        examples = list(examples)
        test_examples = [ex for ex in examples if int(ex['pubid']) in test_split]
        train_examples = [ex for ex in examples if int(ex['pubid']) not in test_split]
        
        converted_dataset['train'] = datasets.Dataset.from_list(train_examples)
        converted_dataset['test'] = datasets.Dataset.from_list(test_examples)
    else:
        no_examples = list(examples.filter(lambda x: x['final_decision'] == 'no'))[:1600]
        yes_examples = list(examples.filter(lambda x: x['final_decision'] == 'yes'))[:2400]
        converted_dataset['train'] = datasets.Dataset.from_list(no_examples + yes_examples).shuffle(seed=0)
        
    for split in converted_dataset.keys():
        labels = converted_dataset[split]['label']
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        print(f"\nLabel distribution in {split} split:")
        for label, count in label_counts.items():
            print(f"{label}: {count} ({count/len(labels)*100:.2f}%)")
        print(f"Total examples in {split}: {len(labels)}")
    
    return datasets.DatasetDict(converted_dataset)

def compute_metrics(eval_preds, dataset, output_dir: str = "evaluation_results"):
    predictions, labels = eval_preds
    
    predictions = np.array(predictions.argmax(-1))  
    labels = np.array(labels)
    
    os.makedirs(output_dir, exist_ok=True)
    
    label_map = {0: 'yes', 1: 'no'}
    text_preds = [label_map[p] for p in predictions]
    text_labels = [label_map[l] for l in labels]
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
        exact_matches = (true_binary & pred_binary)
        metrics[f'{class_name}_exact_match'] = np.mean(exact_matches)
    
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
        print(f"Exact Match: {metrics[f'{class_name}_recall']:.3f}")
    
    return metrics

class BundleTrainer(Trainer):
    """
    Trainer specifically designed for instance bundles with question conditional loss.
    Should only be used with datasets that have bundle information.
    """
    def __init__(self, *args, bundle_args= {'temperature': 0.1, 'mle_weight': 1.0, 'ce_weight': 1.0}, **kwargs):
        kwargs['data_collator'] = self.default_bundle_collator
        super().__init__(*args, **kwargs)
        self.bundle_args = bundle_args 

    def default_bundle_collator(self, features):
        """Default collate function that ensures bundle_ids are included"""
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        
        batch["bundle_ids"] = torch.tensor([f["bundle_ids"] for f in features], dtype=torch.long)
        
        return batch

    def compute_question_conditional_loss(self, model_output, inputs):
        """Compute question-conditional contrastive loss over bundles."""
        logits = model_output.logits  # Predicted logits from the model
        bundle_ids = inputs["bundle_ids"]  # Bundle IDs
        labels = inputs["labels"]  # Ground-truth labels
        
        qc_loss = 0.0
        device = logits.device
        unique_bundles = torch.unique(bundle_ids)
        
        for bundle_id in unique_bundles:
            # Mask to extract examples in the current bundle
            bundle_mask = (bundle_ids == bundle_id)
            bundle_logits = logits[bundle_mask]
            bundle_labels = labels[bundle_mask]
            
            if len(bundle_logits) <= 1:
                continue  # Skip bundles with only one example
            
            # Identify positive examples
            positive_mask = (bundle_labels == 1)
            if positive_mask.sum() == 0:
                continue  # Skip bundles without any positive examples
            
            positive_logits = bundle_logits[positive_mask]  # Positive logits
            all_logits = bundle_logits  # All logits in the bundle
            
            # Compute similarity scores
            similarities = torch.matmul(all_logits, positive_logits.t())  # [N, P]
            similarities /= self.bundle_args["temperature"]  # Scale by temperature
            
            # Create target labels (row indices of `positive_logits`)
            # Each row in `similarities` should target its corresponding column in `positive_logits`
            targets = torch.arange(positive_logits.size(0), device=device).repeat(all_logits.size(0))
            
            # Compute cross-entropy loss
            qc_loss += F.cross_entropy(similarities, targets[:similarities.size(0)])
        
        # Normalize by the number of bundles
        return qc_loss / len(unique_bundles) if len(unique_bundles) > 0 else torch.tensor(0.0, device=device)


    # def compute_question_conditional_loss(self, model_output, inputs):
    #     """Compute question conditional loss over bundles"""
    #     logits = model_output.logits
    #     bundle_ids = inputs["bundle_ids"]
    #     labels = inputs["labels"]
        
    #     qc_loss = torch.tensor(0.0, device=logits.device)
    #     unique_bundles = torch.unique(bundle_ids)
        
    #     for bundle_id in unique_bundles:
    #         # Get questions from this bundle
    #         bundle_mask = (bundle_ids == bundle_id)
    #         bundle_logits = logits[bundle_mask]
    #         bundle_labels = labels[bundle_mask]
            
    #         if len(bundle_logits) <= 1:
    #             continue
                
    #         # Get positive examples (where label is 1)
    #         positive_mask = (bundle_labels == 1)
    #         if not torch.any(positive_mask):
    #             continue
                
    #         positive_logits = bundle_logits[positive_mask]
            
    #         # Compute similarities between questions
    #         similarities = torch.matmul(bundle_logits, positive_logits.t())
    #         similarities = similarities / self.bundle_args['temperature']
            
    #         # Create target distribution
    #         targets = torch.zeros_like(similarities)
    #         targets[positive_mask] = 1.0 / positive_mask.sum()
            
    #         # Compute cross entropy loss
    #         qc_loss += F.cross_entropy(similarities, targets)
            
    #     return qc_loss / len(unique_bundles) if len(unique_bundles) > 0 else qc_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        model_inputs = {k: v for k, v in inputs.items() 
                       if k in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
        outputs = model(**model_inputs)
        
        # Standard classification loss (MLE)
        mle_loss = outputs.loss
        
        # Question conditional contrastive loss
        qc_loss = self.compute_question_conditional_loss(outputs, inputs)
        
        # Combine losses
        total_loss = (self.bundle_args['mle_weight'] * mle_loss + 
                     self.bundle_args['ce_weight'] * qc_loss)
        
        return (total_loss, outputs) if return_outputs else total_loss

