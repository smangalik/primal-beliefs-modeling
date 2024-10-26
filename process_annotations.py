import pandas as pd
import sys  # noqa: F401
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, cohen_kappa_score, confusion_matrix 
from openai import OpenAI
import transformers
import torch
from tqdm import tqdm # noqa: F401
import time
transformers.logging.set_verbosity_error()

llama_model_path = "/chronos_data/pretrained_models/llama3.1-8b-Instruct-hf/"
chatgpt_model="gpt-4o"

llm_prompt = """Does the following post on social media contain an explicit/implicit self-belief?

Explicit Self-Belief: Statements that clearly and explicitly reflect the writer's beliefs about themselves. These self-belief statements should be direct and unambiguous, conveying the writer's personal assessment of their own usual abilities, characteristics, or worth. 
Implicit Self-Belief: Statements that indirectly express the writer’s beliefs about themselves. This includes vague references or statements about the types/categories of persons that the author believes they are, not directly tied to their identity. The purpose of this classification is to capture statements that can be used to infer explicit self-beliefs. 
No Self-Belief: Statements for which there is neither an explicit nor an implicit self-belief present

Some examples of explicit self-beliefs are: “I am the coolest person I know”, “I hate the sound of my voice”, “I am a hard worker”, “I think I am the worst” 
Some examples of implicit self-beliefs are: “I am told that I am funny”, “I love being a morning person”, “I work hard everyday” 
Some examples of no self-beliefs are: “I am a doctor”, “I love you”, “I like pizza”, “I miss you a lot”

Here is the post:
“{}”

Please ONLY respond in the format below, do not include other extraneous text in your response: 
<1 if explicit self-belief, 2 if implicit self-belief, 0 if no self-belief> 
<The probability of your classification being correct between 0 and 100>"""

# Read in the message data
excel_file = 'data/llm_annotations.xlsx'
annotations = pd.read_excel(excel_file)
# Code 3 (I am the only...) = implicit self-belief
annotations['annotator_1'] = annotations['annotator_1'].replace(3,2)
annotations['annotator_2'] = annotations['annotator_2'].replace(3,2)
annotations['annotator_consensus'] = annotations['annotator_3'].replace(3,2)
print("Annotations:",excel_file)
print(annotations.head())

# Split on annotation type
twitter_mask = annotations['message_id'].str.contains(':')
twitter_annotations = annotations[twitter_mask]
reddit_annotations = annotations[~twitter_mask]
chatgpt_annotations = annotations['annotator_chatgpt'].astype(int)
chatgpt_probas = annotations['annotator_chatgpt_proba'].astype(float)
llama_annotations = annotations['annotator_llama'].astype(int)
llama_probas = annotations['annotator_llama_proba'].astype(float)
annotator_1 = annotations['annotator_1'].astype(int)
annotator_2 = annotations['annotator_2'].astype(int)
annotator_consensus = annotations['annotator_consensus'].astype(int)

print(f'\nTwitter annotations: {twitter_annotations.shape[0]}')
print(twitter_annotations['annotator_2'].value_counts())
print(f'Reddit annotations: {reddit_annotations.shape[0]}')
print(reddit_annotations['annotator_2'].value_counts())

# Calculate agreement between annotators
twitter_agreement = (twitter_annotations['annotator_1'] == twitter_annotations['annotator_2']).astype(int)
reddit_agreement = (reddit_annotations['annotator_1'] == reddit_annotations['annotator_2']).astype(int)
print(f'\nTwitter annotator percent agreement: {round(twitter_agreement.mean(),4)}')
print(f'Twitter Cohen\'s Kappa: {cohen_kappa_score(twitter_annotations["annotator_1"], twitter_annotations["annotator_2"])}')
print(f'Reddit annotator perccent agreement: {round(reddit_agreement.mean(),4)}')
print(f'Reddit Cohen\'s Kappa: {cohen_kappa_score(reddit_annotations["annotator_1"], reddit_annotations["annotator_2"])}')

# Convert probas to multi-class representation
def proba_to_multiclass(annotation, proba):
    default_val = (1.0 - proba) / 2
    proba_classwise = [default_val,default_val,default_val]
    proba_classwise[annotation] = proba
    return proba_classwise
try:
    chatgpt_probas = pd.DataFrame([proba_to_multiclass(annotation, proba) for annotation, proba in zip(chatgpt_annotations, chatgpt_probas)], columns=['0','1','2'])
    #print('ChatGPT probas:',chatgpt_probas.head())
except Exception as e:
    print('ChatGPT annotations not available',e)
try:
    llama_probas = pd.DataFrame([proba_to_multiclass(annotation, proba) for annotation, proba in zip(llama_annotations, llama_probas)], columns=['0','1','2'])
    #print('Llama probas:',llama_probas.head())
except Exception as e:
    print('Llama annotations not available',e)

# Calculate agreement with ChatGPT and Llama
labels = [1,2,0]
try:
    for annotator in [annotator_consensus]:
        print(f'\nChatGPT Accuracy w/ {annotator.name} = {accuracy_score(annotator, chatgpt_annotations)}')
        print(f'ChatGPT Weighted F1 w/ {annotator.name} = {f1_score(annotator, chatgpt_annotations,average="weighted")}')
        print(f'ChatGPT AUC ROC w/ {annotator.name} = {roc_auc_score(annotator, chatgpt_probas, multi_class="ovr")}')
        print(f'ChatGPT Confusion Matrix {labels} w/ {annotator.name} = \n{confusion_matrix(annotator, chatgpt_annotations,labels=labels)}')
except Exception as e:
    print('ChatGPT annotations not available',e)
try:
    for annotator in [annotator_consensus]:
        print(f'\nLlama Accuracy w/ {annotator.name} = {accuracy_score(annotator, llama_annotations)}')
        print(f'Llama Weighted F1 w/ {annotator.name} = {f1_score(annotator, llama_annotations,average="weighted")}')
        print(f'Llama AUC ROC w/ {annotator.name} = {roc_auc_score(annotator, llama_probas, multi_class="ovr")}')
        print(f'Llama Confusion Matrix {labels} w/ {annotator.name} = \n{confusion_matrix(annotator, llama_annotations,labels=labels)}')
except Exception as e:
    print('Llama annotations not available',e)

# Check agreement between llama
annotate_llama = False
if llama_annotations.isnull().sum() > 0:
    print('-> Llama has not annotated all messages')
    annotate_llama = True
else:
    # Ask user if they want to re-annotate
    print('\n-> Llama has annotated all messages, do you want to re-annotate?')
    print('')
    response = input('y/[n]: ')
    if response == 'y':
        annotate_llama = True
        
def process_annotation(annotation_str:str):
    # Parse output
    self_belief_response = annotation_str.strip()
    self_belief_response = self_belief_response.replace("<","").replace(">","").split("\n")
    contains_self_belief = self_belief_response[0].strip()
    proba = 0.0
    if len(self_belief_response) > 1:
        proba = self_belief_response[1].strip()
        proba = float(proba) / 100.0
    contains_self_belief = int(''.join(filter(str.isdigit, contains_self_belief)))
    if contains_self_belief not in [0,1,2]:
        contains_self_belief_str = str(contains_self_belief)
        contains_self_belief = int(contains_self_belief_str[0])
        proba = float(contains_self_belief_str[1:]) / 100.0
    return contains_self_belief, proba
        
def get_llama_annotation(message:str):
       
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": llm_prompt.format(message)},
    ]

    outputs = llama_pipeline(
        messages,
        max_new_tokens=32,
        eos_token_id=llama_terminators,
        do_sample=True,
        temperature=0.1,
        top_p=1.0,
    )
    annotation_str = outputs[0]["generated_text"][-1]['content']
    
    try:
        annotation, proba = process_annotation(annotation_str)
        #print('Annotation:',annotation,'w/ proba:',proba)
        return annotation, proba
    except Exception as e:
        print(e)
        print('Error on:',outputs)
        return None, None

# Use Llama to get annotations for all messages
if annotate_llama:
    # set cuda device to gpu 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('\n\ncuda devices',torch.cuda.device_count())
    print('cuda available',torch.cuda.is_available())
    print('cuda current device',torch.cuda.current_device())
    print('cuda device name',torch.cuda.get_device_name())
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('-> Annotating with Llama on',device)
    llama_pipeline = transformers.pipeline(
        "text-generation",
        model=llama_model_path,
        #model_kwargs={"torch_dtype": torch.bfloat16},
        device = device,
    )
    print("Llama Pipeline:", llama_pipeline)
    print("Using device:", llama_pipeline.model.device)
    llama_pipeline.model.config.pad_token_id = llama_pipeline.model.config.eos_token_id
    llama_terminators = [
        llama_pipeline.tokenizer.eos_token_id,
        llama_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    annotations[['annotator_llama','annotator_llama_proba']] = annotations.apply(lambda x: get_llama_annotation(x['message']), axis=1, result_type='expand')
    annotations['annotator_llama'] = annotations['annotator_llama'].astype(int,errors='ignore')
    
    print("Llama Annotations:")
    print(annotations.head())
    
    annotations.to_excel(excel_file.replace(".xlsx","_llama.xlsx"), index=False)

# Use ChatGPT to get annotations for all messages
annotate_chatgpt = False
if chatgpt_annotations.isnull().sum() > 0:
    print('-> ChatGPT has not annotated all messages')
    annotate_chatgpt = True
else:
    # Ask user if they want to re-annotate
    print('-> ChatGPT has annotated all messages\n-> Do you want to re-annotate?')
    print('')
    response = input('y/[n]: ')
    if response == 'y':
        annotate_chatgpt = True

rate_limit = 500.0
sleep_time = (60.0 / rate_limit) + 0.001
def get_chatgpt_annotation(message:str, sleep_time:float=sleep_time):
    chatgpt_client = OpenAI()
    completion = chatgpt_client.chat.completions.create(
        model=chatgpt_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": llm_prompt.format(message)}
        ]   
    )
    # sleep to avoid rate limiting
    time.sleep(sleep_time)
    
    annotation_str = str(completion.choices[0].message.content)
    try:
        annotation, proba = process_annotation(annotation_str)
        #print('Annotation:',annotation,'w/ proba:',proba)
        return annotation, proba
    except Exception as e:
        print(e)
        print('Error on:',completion.choices[0])
        return None, None
    
# annotations = annotations.head(2)
if annotate_chatgpt:
    print('-> Annotating with ChatGPT')
    print('Provide your ChatGPT API key:')
    api_key = input()
    os.environ["OPENAI_API_KEY"] = api_key
    
    annotations[['annotator_chatgpt','annotator_chatgpt_proba']] = annotations.apply(lambda x: get_chatgpt_annotation(x['message']), axis=1, result_type='expand')
    annotations['annotator_chatgpt'] = annotations['annotator_chatgpt'].astype(int,errors='ignore')
    
    print("ChatGPT Annotations:")
    print(annotations.head())
    
    annotations.to_excel(excel_file.replace(".xlsx","_chatgpt.xlsx"), index=False)