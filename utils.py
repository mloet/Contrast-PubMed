import os
import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_metrics(eval_preds, dataset, output_dir):
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
        logits = model_output.logits 
        bundle_ids = inputs["bundle_ids"] 
        labels = inputs["labels"] 
        
        qc_loss = 0.0
        device = logits.device
        unique_bundles = torch.unique(bundle_ids)
        
        for bundle_id in unique_bundles:
            bundle_mask = (bundle_ids == bundle_id)
            bundle_logits = logits[bundle_mask]
            bundle_labels = labels[bundle_mask]
            
            if len(bundle_logits) <= 1:
                continue  
            
            positive_mask = (bundle_labels == 1)
            if positive_mask.sum() == 0:
                continue  
            
            positive_logits = bundle_logits[positive_mask]
            all_logits = bundle_logits 
            
            # Compute similarity scores
            similarities = torch.matmul(all_logits, positive_logits.t()) 
            similarities /= self.bundle_args["temperature"] 
            
            targets = torch.arange(positive_logits.size(0), device=device).repeat(all_logits.size(0))
            
            # Compute cross-entropy loss
            qc_loss += F.cross_entropy(similarities, targets[:similarities.size(0)])
        
        # Normalize by the number of bundles
        return qc_loss / len(unique_bundles) if len(unique_bundles) > 0 else torch.tensor(0.0, device=device)

    def compute_loss(self, model, inputs, return_outputs=False):
        model_inputs = {k: v for k, v in inputs.items() 
                       if k in ["input_ids", "attention_mask", "token_type_ids", "labels"]}
        outputs = model(**model_inputs)
        
        # MLE
        mle_loss = outputs.loss
        
        # Question conditional contrastive loss
        qc_loss = self.compute_question_conditional_loss(outputs, inputs)
        
        # Combine losses
        total_loss = (self.bundle_args['mle_weight'] * mle_loss + 
                     self.bundle_args['ce_weight'] * qc_loss)
        
        return (total_loss, outputs) if return_outputs else total_loss

class StrideTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        chunk_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids", None)
        )
        chunk_logits = chunk_outputs.logits  # Shape: (total_chunks, 2)
        
        unique_ids = inputs["id"].unique()
        aggregated_logits = []
        aggregated_labels = []
        
        for ex_id in unique_ids:
            mask = inputs["id"] == ex_id
            example_logits = chunk_logits[mask].mean(dim=0) 
            aggregated_logits.append(example_logits)
            aggregated_labels.append(inputs["labels"][mask][0])  
        
        aggregated_logits = torch.stack(aggregated_logits)  # Shape: (num_examples, 2)
        aggregated_labels = torch.stack(aggregated_labels)  # Shape: (num_examples,)
        
        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(aggregated_logits, aggregated_labels)
        
        return (loss, {"loss": loss, "logits": aggregated_logits}) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only, **kwargs):
        if prediction_loss_only:
            return super().prediction_step(model, inputs, prediction_loss_only)
            
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            aggregated_logits = outputs["logits"]
            
            unique_ids = inputs["id"].unique()
            labels = torch.tensor([
                inputs["labels"][inputs["id"] == ex_id][0]
                for ex_id in unique_ids
            ], device=aggregated_logits.device)
            
        return loss, aggregated_logits, labels