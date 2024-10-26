import pandas as pd 
import spacy

# Load the data - message_id, user_id, message, subreddit
reddit_posts = pd.read_csv('data/askreddit_posts_15_19.csv')
reddit_comments = pd.read_csv('data/askreddit_comments_15_19.csv')
df = pd.concat([reddit_posts, reddit_comments], axis=0).reset_index(drop=True)

print("\nData loaded")
print(df)

# Clean up the messages
def clean_split_sentence(sentence):
  sentence = sentence.strip()
  sentence = sentence.replace('"','').replace("*","").replace("’","'").replace("‘","").replace('”','').replace('“','')\
    .replace('-','').replace("\n"," ").replace("\\n"," ").replace("\\","").replace(":","").replace("(","").replace(")","")
  sentence = sentence[0].upper() + sentence[1:]
  sentence_words = sentence.split()
  sentence = " ".join([sw for sw in sentence_words if not sw.startswith('@')])
  sentence = sentence.replace("I'm","I am").replace("I've","I have").replace("I'll","I will")
  sentence = sentence.replace("we're","we are").replace("we've","we have")
  
  # Split message on new lines and then punctuation
  sentence = sentence.replace("?", ".").replace("!", ".")
  sentences = sentence.split("\n")
  sentences = [sent.split(".") for sent in sentences]
  sentences = [sent for sublist in sentences for sent in sublist]
  # Drop any empty or whitespace-only sentences
  sentences = [sent for sent in sentences if sent != ""]
  # Filter the sentences that contain "I " or "i "
  sentences = [sent for sent in sentences if "I " in sent or "i " in sent]

  if len(sentences) == 0:
      return []
  
  return sentences

df['sentence'] = df['message'].apply(clean_split_sentence)
print("\nData cleaned")
print(df)
# Extract self-belief candidates
df = df.explode('sentence').reset_index(drop=True)
# remove empty strings
df = df[df['sentence'].str.strip() != '']
df = df.dropna()
print("\nData exploded")
print(df)

# Split messages into rows by sentences
# Use a more robust method for self-belief candidate extraction

def self_belief_candidate(doc) -> str:
    for i, token in enumerate(doc):
        if token.dep_ == "nsubj" and token.text in ["I","i"]:
            if len(doc[i:]) > 2 and any([verb.tag_ == "VBP" for verb in doc[i+1:]]):
                return doc[i:].text
    return ""

# Extract self-belief candidate from sentence
nlp = spacy.load("en_core_web_sm", disable=["lemmatizer","ner","textcat","entity_linker","entity_ruler","textcat_multilabel","textcat_multilabel"])
texts = df['sentence'].tolist()
docs = list(nlp.pipe(texts,n_process=4))
print("\nData tokenized.")
candidates = [self_belief_candidate(doc) for doc in docs]
print("\nSelf-belief candidates extracted.")

# Add the self belief candidates to the dataframe
df['self_belief_candidate'] = candidates
df = df[df['self_belief_candidate'] != ""]
print("\nExtracted self belief candidates")
print(df)

df[['message_id','message','self_belief_candidate']].to_csv('data/askreddit_self_belief_candidates_15_19.csv')