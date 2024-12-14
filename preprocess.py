import datasets

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

def prepare_stride_features(examples, tokenizer, max_length=None):
    max_length = max_length or tokenizer.model_max_length
    
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=max_length,
        stride=min(max_length // 2, 128),
        return_overflowing_tokens=True,
        padding="max_length",
        return_tensors=None  
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    
    tokenized["id"] = [examples["id"][i] for i in sample_mapping]
    tokenized["labels"] = [0 if examples["label"][i].lower() == "yes" else 1 for i in sample_mapping]
    
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
    id = 0

    for item in dataset['questions']:
        if item['type'] == 'yesno':
            # if item['exact_answer'] == 'yes' and yes >= 600:
            #     continue
            # else:
            #     yes+=1

            questions.append(item['body'])
            ids.append(id)
            labels.append(item['exact_answer'])
            context = ''
            for snippet in item['snippets']:
                context += snippet['text']
            contexts.append(context)
            id+=1

    for set in golden:
        for item in set['questions']:
            if item['type'] == 'yesno':
                # if item['exact_answer'] == 'yes' and yes >= 600:
                #     continue
                # else:
                #     yes+=1
                questions.append(item['body'])
                labels.append(item['exact_answer'])
                # ids.append(int(item['id'], 16))
                ids.append(id)
                context = ''
                for snippet in item['snippets']:
                    context += snippet['text']
                contexts.append(context)
                id+=1

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
