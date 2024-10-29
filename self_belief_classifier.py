# Run as python3 roberta_self_belief_classifier.py --train or python3 roberta_self_belief_classifier.py --inference

# Before importing torch, set os params
import os
import pandas as pd
from transformers import EarlyStoppingCallback, RobertaTokenizerFast, RobertaForSequenceClassification,Trainer, TrainingArguments
import torch
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import sys  # noqa: E401
from matplotlib import pyplot as plt
import json

huggingface_model = "roberta-base"

data_path = "/chronos_data/smangalik/beliefs_modeling/"

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["WORLD_SIZE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
print("Twitter Human Annotations:", df_human_2.shape)
df_human = pd.concat([df_human_1, df_human_2, train[['text','label']]])
df_human.drop(columns=['index','yearweek_userid','Abby Rating','Sid self_belief_explicit'], inplace=True)
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
    print("\nHuman Only") 
    # Train + Remaining Human
    train = df_human[['text','label']]
elif data_type == 2: # LLM Only
    print("\nLLM Only") 
    # LLM
    train = df_llm[['text','label']]
elif data_type == 3: # LLM + Human
    print("\nLLM + Human")
    train = pd.concat([ df_llm[['text','label']], df_human[['text','label']] ]) 
    # LLM + Train + Remaining Human
elif data_type == 4: # fetch the best model from the LLM Only data and then finetune on Human Only
    print("\nLLM then Human") 
    # Train + Remaining Human
    train = df_human[['text','label']]
    
# Drop duplicates and NA values
train.drop_duplicates(inplace=True)
train.dropna(inplace=True)
train = train[train['text'].str.len() > 0]

# Create an evaluation set from 30% of the training data
eval = train.sample(frac=0.3, random_state=25)
train = train.drop(eval.index)

# Print some statistics
print("\nTrain Stats:")
print(train['label'].value_counts())
print("Number of labels:", train['label'].nunique())
print("Baseline accuracy:", train['label'].value_counts().max() / train.shape[0])

print("\nEval Stats:")
print(eval['label'].value_counts())
print("Number of labels:", eval['label'].nunique())
print("Baseline accuracy:", eval['label'].value_counts().max() / eval.shape[0])

print("\nTest Stats:")
print(test['label'].value_counts())
print("Number of labels:", test['label'].nunique())
print("Baseline accuracy:", test['label'].value_counts().max() / test.shape[0])
    
assert(train['label'].nunique() == test['label'].nunique() == eval['label'].nunique())
num_labels = train['label'].nunique()
    
def plot_metrics_over_epochs(metrics):
    
    # Plot metrics over epoches
    # plt.plot(metrics['epoch'], metrics['eval_accuracy'], label='Eval Accuracy')
    metrics_to_plot = ['eval_loss','loss']
    for metric in metrics_to_plot:
        metrics_plot = metrics[~metrics[metric].isna()]
        plt.plot(metrics_plot['epoch'], metrics_plot[metric], label=metric)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.show()    
    plt.clf()
    
    metrics_plot = metrics[~metrics['eval_f1'].isna()]
    plt.plot(metrics_plot['epoch'], metrics_plot['eval_f1'], label='Eval F1')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.show()  

       
if mode == "--train":
    
    print("\nTraining Data:")
    print(train[['text','label']].sample(25))
    print(train['label'].value_counts())
    
    # List the available devices
    print("Available Devices:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print("-> Device Name:", torch.cuda.get_device_name(i))
    
    # Set the device to GPU (cuda) if available, otherwise stick with CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Training Device:", device)
    
    
    train_data = Dataset.from_pandas(train)
    eval_data = Dataset.from_pandas(eval)
    
    # write train/test data to disk
    train_data.to_csv(data_path + 'data/train_data_{}.csv'.format(data_type), index=False, encoding='utf-8')
    eval_data.to_csv(data_path + 'data/eval_data_{}.csv'.format(data_type), index=False, encoding='utf-8')
    
    # Define the tokenizer and mod
    if data_type == 4: # Finetuning
        print("Finetuning from the LLM Only model")
        trained_model = data_path + 'models/self-belief-classifier-{}'.format(2)
        model = RobertaForSequenceClassification.from_pretrained(trained_model)
    else: # Training from scratch
        print("Training from scratch")
        model = RobertaForSequenceClassification.from_pretrained(huggingface_model, num_labels=num_labels)
    tokenizer = RobertaTokenizerFast.from_pretrained(huggingface_model, max_length = 256)
    
    # Tokenize the input data
    def tokenization(batched_text):
        return tokenizer(batched_text['text'], padding = True, truncation=True, max_length=256)
    train_data = train_data.map(tokenization, batched=True, batch_size=len(train_data))
    eval_data = eval_data.map(tokenization, batched=True, batch_size=len(eval_data))

    # Set up data
    train_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    eval_data.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    
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
    
    output_dir = data_path + 'models/self-belief-classifier-{}'.format(data_type)

    # len of batch size rounded to the nearest multiple of 10
    batch_size = 8
    if data_type in [1,4]: # human-only Tuning
        step_size = 50
    else: # LLM Tuning
        step_size = 500
    print("\nStep Size:", step_size)
    print("Batch Size:", batch_size)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=50,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = 2,    
        per_device_eval_batch_size= batch_size,
        evaluation_strategy = "steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        warmup_steps=step_size,
        learning_rate=1e-5,
        metric_for_best_model = 'f1',
        weight_decay=0.00001,
        eval_steps = step_size,
        logging_steps = step_size,
        save_steps = step_size,
        fp16 = True,
        logging_dir=data_path + '/self-belief-classifier-logs',
        dataloader_num_workers = 8,
        run_name = 'self-belief-classification'
    )
    
    early_stop_callback = EarlyStoppingCallback(
        early_stopping_patience=3, 
        early_stopping_threshold=0.001
        )

    # Define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=eval_data,
        callbacks=[early_stop_callback]
    )
    
    print('\n',str(training_args).replace('\n',' '),'\n')
        
    # Train the model
    trainer.train()
    # Save the model
    trainer.save_model(data_path + 'models/self-belief-classifier-{}'.format(data_type))
    # Evaluate the model
    final_metrics = trainer.evaluate()
    print("Final Metrics:\n", final_metrics)
    
elif mode == "--inference":
    # Load the tokenizer and model
    model_path = data_path + 'models/self-belief-classifier-{}'.format(data_type)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
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
    print("\nPredicting classes for some example texts:")
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
    
    # Predict all values at once
    print("\nPredicting labels for test data...")
    test['predicted_label'] = predict_classes(test['text'].tolist())
    
    # Print the results
    print("\nTest Data:")    
    print(test[['text','label','predicted_label']].sample(5))
    print("Predicted label counts:")
    print(test['predicted_label'].value_counts())
    print("True label counts:")
    print(test['label'].value_counts())
    mfq = test['label'].value_counts().idxmax()
    mfq_count = test['label'].value_counts().max()
    acc = accuracy_score(test['label'], test['predicted_label'])
    precision, recall, f1, _ = precision_recall_fscore_support(test['label'], test['predicted_label'], average='weighted', zero_division=0)
    acc_baseline = mfq_count / test.shape[0]
    precision_b, recall_b, f1_b, _ = precision_recall_fscore_support(test['label'], [mfq] * test.shape[0], average='weighted', zero_division=0)
    print("Accuracy:", acc, "Baseline:", acc_baseline)
    print("F1:", f1, "Baseline:", f1_b)
    
    # Predict all values at once
    print("\nPredicting labels for eval data...")
    eval['predicted_label'] = predict_classes(eval['text'].tolist())
    
    # Print the results
    print("\nEval Data:")    
    print(eval[['text','label','predicted_label']].sample(5))
    print("Predicted label counts:")
    print(eval['predicted_label'].value_counts())
    print("True label counts:")
    print(eval['label'].value_counts())
    mfq = eval['label'].value_counts().max()
    acc = accuracy_score(eval['label'], eval['predicted_label'])
    precision, recall, f1, _ = precision_recall_fscore_support(eval['label'], eval['predicted_label'], average='weighted', zero_division=0)
    acc_baseline = mfq / eval.shape[0]
    precision_b, recall_b, f1_b, _ = precision_recall_fscore_support(eval['label'], [mfq] * eval.shape[0], average='weighted', zero_division=0)
    print("Accuracy:", acc, "Baseline:", acc_baseline)
    print("F1:", f1, "Baseline:", f1_b)    
    
    # Load metrics from the training history
    metrics_path = data_path + 'models/self-belief-classifier-{}/'.format(data_type)
    # Get all files in the metrics_path named trainer_state.json
    files = [f for f in os.listdir(metrics_path) if f.endswith('trainer_state.json')]
    
    file_paths = []
    for root, dirs, files in os.walk(metrics_path):
        for file in files:
            if file.endswith('trainer_state.json'):
                file_paths.append(os.path.join(root, file))
    print("Files:",file_paths)
    # Get the latest file
    latest_file = max([os.path.join(metrics_path, f) for f in file_paths], key=os.path.getctime)
    print("Latest File:", latest_file)
    json = json.load(open(latest_file))
    log_history = json['log_history']
    #print("Log History:", log_history)
    print("Metrics:")
    metrics = pd.DataFrame(log_history)
    print(metrics)
    plot_metrics_over_epochs(metrics)
    
else:
    print("Invalid mode. Please use --train or --inference")


