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

output_file = "data/self_belief_candidates_annotated_10k.csv"

twitter_annotations_1 = pd.read_csv("data/twitter_self_belief_candidates_5k.csv", encoding='utf-8').drop(columns=['yearweek_userid'])
twitter_annotations_2 = pd.read_csv("data/twitter_self_belief_candidates_5k_2.csv", encoding='utf-8').drop(columns=['yearweek_userid','original_message'])
twitter_annotations = pd.concat([twitter_annotations_1,twitter_annotations_2],ignore_index=True)

reddit_annotations = pd.read_csv("data/askreddit_self_belief_candidates_10k.csv",encoding='utf-8').drop(columns=['index','message'])
reddit_annotations.rename(columns={'self_belief_candidate':'message'}, inplace=True)

annotations = pd.concat([twitter_annotations,reddit_annotations],ignore_index=True)

print(f'\nTwitter annotations: {twitter_annotations.shape}')
print(twitter_annotations.head())
print(f'\nReddit annotations: {reddit_annotations.shape}')
print(reddit_annotations.head())
print(f'\nAll annotations: {annotations.shape}')
print(annotations.sample(20))
        
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
        

# Use ChatGPT to get annotations for all messages
print('\n-> Do you want to re-annotate with ChatGPT?')
response = input('y/[n]: ')
if response == 'y':
    annotate_chatgpt = True
else:
    annotate_chatgpt = False
    print('\n-> Skipping ChatGPT annotation')
    
    annotations = pd.read_excel(output_file.replace('.csv','.xlsx'))
    
    # Twitter message_ids have letters
    twitter_mask = annotations['message_id'].str.contains('[a-zA-Z]', regex=True, na=False)
    twitter_annotations = annotations[twitter_mask]
    reddit_annotations = annotations[~twitter_mask]
    
    print("\nChatGPT Annotations:",annotations.shape)
    print(annotations)
    
    print("\nUnique ChatGPT Annotations:")
    print(annotations['annotator_chatgpt'].value_counts())
    
    print("\nUnique ChatGPT Annotations on Twitter:")
    print(twitter_annotations['annotator_chatgpt'].value_counts())
    
    print("\nUnique ChatGPT Annotations on Reddit:")
    print(reddit_annotations['annotator_chatgpt'].value_counts())
    
    print("\nChatGPT Probabilities Assigned:")
    print(annotations['annotator_chatgpt_proba'].describe())
    
    sys.exit()

rate_limit = 500.0
rate_refresh = 60.0
sleep_time = (rate_refresh / rate_limit)
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
        return e, annotation_str

if annotate_chatgpt:
    os.environ["OPENAI_API_KEY"] = "INSERT_YOUR_API_KEY_HERE"
    print('-> Annotating with ChatGPT')
    print('Provide your ChatGPT API key:')
    api_key = input()
    os.environ["OPENAI_API_KEY"] = api_key
    
    annots = []
    probas = []
    for message in tqdm(annotations['message'].tolist()):
        annotation, proba = get_chatgpt_annotation(message)
        annots.append(annotation)
        probas.append(proba)
    annotations['annotator_chatgpt'] = annots
    annotations['annotator_chatgpt_proba'] = probas
    annotations['annotator_chatgpt'] = annotations['annotator_chatgpt'].astype(int,errors='ignore')
    annotations['annotator_chatgpt'] = annotations['annotator_chatgpt'].astype(float,errors='ignore')
    
    print("\nChatGPT Annotations:")
    print(annotations)
    
    annotations.to_csv(output_file, encoding='utf-8', index=False)