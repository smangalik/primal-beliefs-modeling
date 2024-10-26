# Run as python3 roberta_self_belief_classifier.py --train or python3 roberta_self_belief_classifier.py --inference

# Before importing torch, set os params
import os
import time
import pandas as pd
from transformers import EarlyStoppingCallback, RobertaTokenizerFast, RobertaForSequenceClassification,Trainer, TrainingArguments
import torch
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import sys  # noqa: E401

huggingface_model = "roberta-base"

data_path = "/chronos_data/smangalik/beliefs_modeling/"

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Run Parameters
mode = sys.argv[1] # --train or --inference
data_type = 1
if len(sys.argv) > 2:
    data_type = int(sys.argv[2]) # 1 (human only) or 2 (llm only) or 3 (human + llm) or 4 (llm then human)

# Read in the inter-annotator agreement set
df = pd.read_excel('data/llm_annotations.xlsx')
df['text'] = df['message']
df['label'] = df['annotator_3'].replace(3,2) # Code 3 (I am the only...) = implicit self-belief
print("\nInterrater Annotations:", df.shape)
print(df.head())

# Train / Test split
train = df.sample(frac=0.5, random_state=25)
test = df.drop(train.index)

# Human annotations NOT from the interrater agreement set
df_human_1 = pd.read_excel("data/askreddit_self_belief_candidates_annotated.csv.xlsx")
df_human_1['text'] = df_human_1['self_belief_candidate']
df_human_1['label'] = df_human_1['self_belief_explicit'].replace(3,2)
df_human_1 = df_human_1[~df_human_1['text'].isin(df['text'])] # Remove the overlapping messages
print("\nReddit Human Annotations:", df_human_1.shape)
df_human_2 = pd.read_excel("data/twitter_i_am_person_annotatedINTERRATER.csv.xlsx", sheet_name="i-am--persona-za-z")
df_human_2['text'] = df_human_2['message']
df_human_2['label'] = df_human_2['Abby Rating'].replace(3,2)
df_human_2 = df_human_2[df_human_2['Sid self_belief_explicit'].isna()]
df_human = pd.concat([df_human_1, df_human_2])
df_human.drop(columns=['index','yearweek_userid','Abby Rating','Sid self_belief_explicit'], inplace=True)
print("Twitter Human Annotations:", df_human_2.shape)
print("Human Annotations:", df_human.shape)
print(df_human.head())

# ChatGPT Annotations
df_llm = pd.read_excel('data/self_belief_candidates_annotated_10k.xlsx')
df_llm['text'] = df_llm['message']
df_llm['label'] = df_llm['annotator_chatgpt']
print("\nLLM Annotations:", df_llm.shape)
print(df_llm.head())

# Alter the data based on the data type
if data_type == 1: # Human Only
    # Concatenate the human data
    train = pd.concat([ train[['text','label']], df_human[['text','label']] ])
elif data_type == 2: # LLM Only
    # train is replaced with the LLM data
    train = df_llm[['text','label']]
elif data_type == 3: # LLM + Human
    # Concatenate the LLM data with the human data
    train = pd.concat([ train[['text','label']], df_llm[['text','label']], df_human[['text','label']] ])
elif data_type == 4: # fetch the best model from the LLM data and then train on the human data
    pass

# Print some statistics
print("\nTrain Stats:")
print(train['label'].value_counts())
print("Number of labels:", train['label'].nunique())
print("Baseline accuracy:", train['label'].value_counts().max() / train.shape[0])

print("\nTest Stats:")
print(test['label'].value_counts())
print("Number of labels:", test['label'].nunique())
print("Baseline accuracy:", test['label'].value_counts().max() / test.shape[0])
    
assert(train['label'].nunique() == test['label'].nunique())
num_labels = train['label'].nunique()
    
print("STOPPING EARLY")
sys.exit()
    
if mode == "--train":
    
    print("\nTraining Data:")
    print(df[['text','label']].sample(25))
    print(df['label'].value_counts())
    
    # List the available devices
    print("Available Devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print("-> Device Name:", torch.cuda.get_device_name(i))
    
    # Set the device to GPU (cuda) if available, otherwise stick with CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Training Device:", device)
    
    
    train_data = Dataset.from_pandas(train)
    test_data = Dataset.from_pandas(test)
    
    # write train/test data to disk
    train_data.to_csv(data_path + 'data/train_data_{}.csv'.format(data_type), index=False, encoding='utf-8')
    test_data.to_csv(data_path + 'data/test_data_{}.csv'.format(data_type), index=False, encoding='utf-8')
    
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
        output_dir=data_path + 'self-belief-classifier-{}'.format(data_type),
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
        logging_dir=data_path + '/self-belief-classifier-logs',
        dataloader_num_workers = 8,
        run_name = 'self-belief-classification'
    )
    
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
    
    print("Trainer Params:")
    print(training_args)
        
    # Train the model
    trainer.train()
    # Save the model
    trainer.save_model(data_path + 'self-belief-classifier-{}-{}'.format(data_type,time.time()))
    # Evaluate the model
    final_metrics = trainer.evaluate()
    print("Final Metrics:\n", final_metrics)
    
elif mode == "--inference":
    # Load the tokenizer and model
    model = RobertaForSequenceClassification.from_pretrained(data_path + 'self-belief-classifier-{}'.format(data_type))
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
    df_test = pd.read_csv(data_path + 'data/test_data.csv')
    
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


