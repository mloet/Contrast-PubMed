import os
import json
import datasets
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import Trainer, PreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass

def prepare_dataset_features(examples, tokenizer, max_length):
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

def prepare_stride_features(examples, tokenizer, max_length):
    """Convert examples to model inputs using document stride"""
    
    # Tokenize with stride
    tokenized = tokenizer(
        examples['question'],
        examples['context'],
        truncation='only_second',
        max_length=max_length,
        stride=128,
        return_overflowing_tokens=True,
        padding='max_length'
    )
    
    # Map labels and track example IDs
    label_map = {'yes': 0, 'no': 1}
    
    # For each chunk, get the ID and label of its source example
    chunk_ids = [examples['id'][i] for i in tokenized['overflow_to_sample_mapping']]
    chunk_labels = [label_map[examples['label'][i]] for i in tokenized['overflow_to_sample_mapping']]
    
    # Add to tokenized output
    tokenized['example_id'] = chunk_ids
    tokenized['labels'] = chunk_labels
    
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

def convert_bioasq_to_yn(dataset, golden):
    questions = []
    contexts = []
    labels = []
    ids = []
    yes = 0

    for item in dataset['questions']:
        if item['type'] == 'yesno':
            if item['exact_answer'] == 'yes' and yes >= 600:
                continue
            else:
                yes+=1

            questions.append(item['body'])
            ids.append(item['id'])
            labels.append(item['exact_answer'])
            context = ''
            for snippet in item['snippets']:
                context += snippet['text']
            contexts.append(context)

    for set in golden:
        for item in set['questions']:
            if item['type'] == 'yesno':
                if item['exact_answer'] == 'yes' and yes >= 600:
                    continue
                else:
                    yes+=1
                questions.append(item['body'])
                labels.append(item['exact_answer'])
                ids.append(item['id'])
                context = ''
                for snippet in item['snippets']:
                    context += snippet['text']
                contexts.append(context)

    return {
        'train': datasets.Dataset.from_dict({
            'question': questions,
            'context': contexts,
            'label': labels,
            'id': ids
        })
    }

def convert_pubmed_to_yn(dataset, test_split):
    def convert_example(example):
        return {
            'question': example['question'],
            'context': ' '.join(example['context']['contexts']),
            'label': example['final_decision'],
            'id': example['pubid']
        }

    examples = dataset['train'].map(convert_example)
    examples = examples.filter(lambda x: x['label'] != 'maybe')

    converted_dataset = {}
    if test_split:
        examples = list(examples)
        test_examples = [ex for ex in examples if int(ex['id']) in test_split]
        train_examples = [ex for ex in examples if int(ex['id']) not in test_split]
        
        converted_dataset['train'] = datasets.Dataset.from_list(train_examples).remove_columns(['pubid', 'long_answer', 'final_decision'])
        converted_dataset['test'] = datasets.Dataset.from_list(test_examples).remove_columns(['pubid', 'long_answer', 'final_decision'])
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
    
    # print(dataset)

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
        kwargs['data_collator'] = self.id_collator
        super().__init__(*args, **kwargs)
        self.bundle_args = bundle_args 

    def id_collator(self, features):
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

class StrideTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        kwargs['data_collator'] = self.id_collator
        super().__init__(*args, **kwargs)

    def id_collator(self, features):
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        
        batch["example_id"] = torch.tensor([f["example_id"] for f in features], dtype=torch.long)
        
        return batch
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get model outputs for all chunks
        outputs = model(**{
            k: v for k, v in inputs.items() 
            if k not in ['labels', 'example_id', 'overflow_to_sample_mapping']
        })
        logits = outputs.logits
        
        # Aggregate predictions by example ID
        unique_ids = inputs['example_id'].unique()
        aggregated_logits = []
        aggregated_labels = []
        
        for id in unique_ids:
            mask = inputs['example_id'] == id
            # Average predictions for all chunks of this example
            example_logits = logits[mask].mean(dim=0)
            aggregated_logits.append(example_logits)
            # Get the label (same for all chunks of one example)
            aggregated_labels.append(inputs['labels'][mask][0])
        
        aggregated_logits = torch.stack(aggregated_logits)
        aggregated_labels = torch.stack(aggregated_labels)
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(aggregated_logits, aggregated_labels)
        
        if return_outputs:
            outputs.logits = aggregated_logits
            return loss, outputs
        return loss
    
def prepare_stride_features2(examples, tokenizer, max_length=None):
    """Optimized feature preparation for yes/no classification with document stride"""
    max_length = max_length or tokenizer.model_max_length
    
    # Tokenize with stride
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=min(max_length // 2, 128),
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors=None  # Return python lists for easier batch processing
    )

    # Get mapping from chunks to original examples
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    
    # Map IDs and labels to chunks
    tokenized["id"] = [examples["id"][i] for i in sample_mapping]
    tokenized["labels"] = [0 if examples["label"][i].lower() == "yes" else 1 for i in sample_mapping]
    
    return tokenized


class YesNoTrainer(Trainer):
    """Optimized trainer for yes/no classification with document stride"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get logits for each chunk
        chunk_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids", None)
        )
        chunk_logits = chunk_outputs.logits  # Shape: (total_chunks, 2)
        
        # Group chunks by example_id and average their predictions
        unique_ids = inputs["id"].unique()
        aggregated_logits = []
        aggregated_labels = []
        
        for ex_id in unique_ids:
            mask = inputs["id"] == ex_id
            example_logits = chunk_logits[mask].mean(dim=0)  # Average logits for this example
            aggregated_logits.append(example_logits)
            aggregated_labels.append(inputs["labels"][mask][0])  # Take first label (all chunks have same label)
        
        # Stack for batch processing
        aggregated_logits = torch.stack(aggregated_logits)  # Shape: (num_examples, 2)
        aggregated_labels = torch.stack(aggregated_labels)  # Shape: (num_examples,)
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(aggregated_logits, aggregated_labels)
        
        return (loss, {"loss": loss, "logits": aggregated_logits}) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None,):
        """Optimized prediction step that handles chunk aggregation"""
        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only)
            
        with torch.no_grad():
            # Get and aggregate chunk predictions
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            aggregated_logits = outputs["logits"]
            
            # Get one label per example (labels are same for all chunks of an example)
            unique_ids = inputs["id"].unique()
            labels = torch.tensor([
                inputs["labels"][inputs["id"] == ex_id][0]
                for ex_id in unique_ids
            ], device=aggregated_logits.device)
            
        return loss, aggregated_logits, labels
    
def compute_metrics2(eval_preds, dataset, output_dir: str = "evaluation_results"):
    predictions, labels = eval_preds
    
    # Since predictions come from YesNoTrainer, they're already aggregated per example
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
    
    # Calculate overall metrics - no change needed here since predictions are already aggregated
    metrics['total_f1'] = f1_score(labels, predictions, average='macro')
    metrics['total_exact_match'] = np.mean(predictions == labels)
    metrics['total_precision'] = precision_score(labels, predictions, average='macro')
    metrics['total_recall'] = recall_score(labels, predictions, average='macro')
    
    # Calculate per-class metrics - no change needed
    for class_idx, class_name in label_map.items():
        true_binary = (labels == class_idx)
        pred_binary = (predictions == class_idx)
        
        metrics[f'{class_name}_f1'] = f1_score(true_binary, pred_binary)
        metrics[f'{class_name}_precision'] = precision_score(true_binary, pred_binary)
        metrics[f'{class_name}_recall'] = recall_score(true_binary, pred_binary)
        exact_matches = (true_binary & pred_binary)
        metrics[f'{class_name}_exact_match'] = np.mean(exact_matches)
    
    # For error analysis, we need to map back to original examples
    # Get unique example IDs to avoid duplicates from chunks
    unique_ids = []
    seen_ids = set()
    for i, example_id in enumerate(dataset['id']):
        if example_id not in seen_ids:
            unique_ids.append(i)
            seen_ids.add(example_id)
    
    # Generate error analysis using unique examples
    # for i, (pred, label) in enumerate(zip(text_preds, text_labels)):
    #     example_idx = unique_ids[i]  # Map back to original example
    #     example = {
    #         'id': dataset['id'][example_idx],
    #         'question': dataset['question'][example_idx],
    #         'predicted_label': pred,
    #         'correct_label': label,
    #         'context': dataset['context'][example_idx]
    #     }
        
    #     if pred == label:
    #         error_analysis['correct'].append(example)
    #     elif label == 'yes':  # Missed a yes (false negative)
    #         error_analysis['false_negatives'].append(example)
    #     else:  # Predicted yes when it was no (false positive)
    #         error_analysis['false_positives'].append(example)
    
    # # Write error analysis files
    # for category, examples in error_analysis.items():
    #     output_file = os.path.join(output_dir, f'{category}.jsonl')
    #     with open(output_file, 'w') as f:
    #         for example in examples:
    #             f.write(json.dumps(example) + '\n')
    
    # Add stride-specific metrics
    metrics['num_total_chunks'] = len(dataset['id'])
    metrics['num_unique_examples'] = len(unique_ids)
    metrics['avg_chunks_per_example'] = metrics['num_total_chunks'] / metrics['num_unique_examples']
    
    # Write summary metrics
    metrics_file = os.path.join(output_dir, 'metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Total Examples: {len(labels)}")
    print(f"Total Chunks: {metrics['num_total_chunks']}")
    print(f"Average Chunks per Example: {metrics['avg_chunks_per_example']:.2f}")
    print(f"Overall F1: {metrics['total_f1']:.3f}")
    print(f"Overall Exact Match: {metrics['total_exact_match']:.3f}")
    print("\nPer-class metrics:")
    for class_name in label_map.values():
        print(f"\n{class_name.upper()}:")
        print(f"F1: {metrics[f'{class_name}_f1']:.3f}")
        print(f"Precision: {metrics[f'{class_name}_precision']:.3f}")
        print(f"Recall: {metrics[f'{class_name}_recall']:.3f}")
        print(f"Exact Match: {metrics[f'{class_name}_exact_match']:.3f}")
    
    return metrics