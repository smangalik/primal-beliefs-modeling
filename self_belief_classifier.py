# Run as python3 roberta_self_belief_classifier.py --train or python3 roberta_self_belief_classifier.py --inference

# Before importing torch, set os params
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["WORLD_SIZE"] = "1"

import time
import pandas as pd
from transformers import EarlyStoppingCallback, RobertaTokenizerFast, RobertaForSequenceClassification,Trainer, TrainingArguments
import torch
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys  # noqa: E401

huggingface_model = "roberta-base"

data_path = "/chronos_data/smangalik/self-belief-classifiers/"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the datasets
shared_columns = ['message_id', 'message', 'self_belief_explicit']

# I am _ person
df_i_am = pd.read_excel("/home/smangalik/primal-beliefs-modeling/data/i_am_person_annotated.csv.xlsx")
df_i_am = df_i_am[df_i_am['self_belief_explicit'].notna()]
df_i_am['self_belief_explicit'] = df_i_am['self_belief_explicit'].astype(int)
df_i_am['message_id'] = df_i_am.apply(lambda x: hash(x['yearweek_userid'] + x['message']), axis=1)
print("I am _ person:", df_i_am.shape, df_i_am.columns)
print(df_i_am[shared_columns])

# <whatever> I <VBD> ___  <whatever>
df_self_beliefs = pd.read_excel("/home/smangalik/primal-beliefs-modeling/data/annotated_self_beliefs.csv.xlsx")
df_self_beliefs = df_self_beliefs[df_self_beliefs['self_belief_explicit'].notna()]
print("Self beliefs:", df_self_beliefs.shape, df_self_beliefs.columns)
print(df_self_beliefs[shared_columns])

# I <VBD> ___
df_self_beliefs_candidate = df_self_beliefs.copy(deep=True)
df_self_beliefs_candidate.drop(columns=['message'], inplace=True)
df_self_beliefs_candidate.rename(columns={'better_candidate': 'message'}, inplace=True)
df_self_beliefs_candidate['message_id'] = df_self_beliefs_candidate.apply(lambda x: hash(x['message']), axis=1)
print("Self beliefs: [Better Candidate]", df_self_beliefs_candidate.shape, df_self_beliefs_candidate.columns)
print(df_self_beliefs_candidate[shared_columns])


df = pd.concat([
    df_i_am[shared_columns], 
    df_self_beliefs_candidate[shared_columns],
    df_self_beliefs[shared_columns]
    ])
df.rename(columns={'message': 'text', 'self_belief_explicit': 'label'}, inplace=True)
print("\nFinal Data:", df.shape, df.columns)
print(df)

df.to_csv('/home/smangalik/primal-beliefs-modeling/data/all_data.csv', index=False, encoding='utf-8')

# Print some statistics
num_labels = len(df['label'].unique())
print("Label counts:")
print(df['label'].value_counts())
print("Number of labels:", num_labels)
print("Baseline accuracy:", df['label'].value_counts().max() / df.shape[0])

mode = sys.argv[1]
if mode == "--train":
    
    print("Training Data:")
    print(df[['text','label']].sample(25))
    print(df['label'].value_counts())
    
    # List the available devices
    print("Available Devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print("-> Device Name:", torch.cuda.get_device_name(i))
    
    # Set the device to GPU (cuda) if available, otherwise stick with CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Training Device:", device)
    
    
    dataset = Dataset.from_pandas(df).train_test_split(shuffle=True, seed=25, test_size=0.2)
    train_data, test_data = dataset['train'], dataset['test']
    
    # write train/test data to disk
    train_data.to_csv('data/train_data.csv', index=False, encoding='utf-8')
    test_data.to_csv('data/test_data.csv', index=False, encoding='utf-8')
    
    # Define the tokenizer and model
    model = RobertaForSequenceClassification.from_pretrained(huggingface_model, num_labels=num_labels)
    tokenizer = RobertaTokenizerFast.from_pretrained(huggingface_model, max_length = 256)
    
    # Tokenize the input data
    def tokenization(batched_text):
        return tokenizer(batched_text['text'], padding = True, truncation=True, max_length=256)
    train_data = train_data.map(tokenization, batched=True, batch_size=len(train_data))
    test_data = test_data.map(tokenization, batched=True, batch_size=len(test_data))

    # Set up data
    train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # define accuracy metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    training_args = TrainingArguments(
        output_dir='./self-belief-classifier',
        num_train_epochs=50,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,    
        per_device_eval_batch_size= 4,
        evaluation_strategy = "steps",
        save_strategy="steps",
        #disable_tqdm = False, 
        load_best_model_at_end=True,
        warmup_steps=100,
        eval_steps = 50,
        learning_rate=1e-5,
        metric_for_best_model = 'f1',
        weight_decay=0,
        logging_steps = 8,
        fp16 = True,
        logging_dir='./self-belief-classifier-logs',
        dataloader_num_workers = 8,
        run_name = 'self-belief-classification'
    )
    
    print("Training Arguments:")
    print(training_args)
    
    early_stop_callback = EarlyStoppingCallback(
        early_stopping_patience=3, 
        early_stopping_threshold=0.0001
        )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=test_data,
        callbacks=[early_stop_callback]
    )
        
    # Train the model
    trainer.train()
    # Save the model
    trainer.save_model('./self-belief-classifier-{}'.format(time.time()))
    # Evaluate the model
    final_metrics = trainer.evaluate()
    print("Final Metrics:\n", final_metrics)
    
elif mode == "--inference":
    # Load the tokenizer and model
    model = RobertaForSequenceClassification.from_pretrained('./self-belief-classifier')
    tokenizer = RobertaTokenizerFast.from_pretrained(huggingface_model, max_length = 256)
    
    # Define the function to predict the class of new messages
    def predict_class(text):
        try:
            inputs = tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors='pt')
            outputs = model(**inputs)
            logits = outputs.logits
            soft_logits = torch.softmax(logits, dim=1).tolist()
            predicted_class = np.argmax(soft_logits, axis=1)
            #print(logits,soft_logits,predicted_class)
            return int(predicted_class[0])
        except Exception as _:
            return -1
        
    def predict_classes(texts):
        try:
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
            outputs = model(**inputs)
            logits = outputs.logits
            soft_logits = torch.softmax(logits, dim=1).tolist()
            predicted_classes = np.argmax(soft_logits, axis=1)
            #print(logits,soft_logits,predicted_class)
            return predicted_classes
        except Exception as e:
            print(e)
            return [-1] * len(texts)

    # Example usage
    print("Predicting classes for some example texts:")
    texts = [
        "I think I am the coolest",
        "I think I am not the coolest",
        "I hate the sound of my voice",
        "I don't hate the sound of my voice",
        "I am a hard worker",
        "I ain't a hard worker",
        "I am angry",
        "I am never angry",
        "I am a lazy person",
        "I am not a lazy person",
        "I feel like a loser",
        "I don't feel like a loser",
        "I am a doctor",
        "I am not a doctor",
        "I love you",
        "I don't love you",
        "I like pizza",
        "I don't like pizza",
        "I miss you a lot",
        "I don't miss you a lot",
        "I know she wants me",
        "I know she doesn't want me",
        "I think this is easy",
        "I don't think this is easy",
    ]
    text_preds = predict_classes(texts)
    for text,pred in zip(texts,text_preds):
        print(f"Predicted class for: '{text}' = {pred}.")
    
    # Load test data
    df_test = pd.read_csv('data/test_data.csv')
    
    # Predict all values at once
    print("\nPredicting labels for test data:")
    y_pred = predict_classes(df_test['text'].tolist())
    print("Predicted labels:", y_pred)
    df_test['predicted_label'] = y_pred
    
    # Print the results
    print("Test Data:")    
    print(df_test[['text','label','predicted_label']].sample(25))
    print("Predicted label counts:")
    print(df_test['predicted_label'].value_counts())
    print("True label counts:")
    print(df_test['label'].value_counts())
    
    print("\nMetrics:")
    acc = accuracy_score(df_test['label'], df_test['predicted_label'])
    precision, recall, f1, _ = precision_recall_fscore_support(df_test['label'], df_test['predicted_label'], average='weighted', zero_division=0)
    print("Accuracy:", acc)
    print("F1:", f1)
    
    
    
else:
    print("Invalid mode. Please use --train or --inference")


