import pandas as pd
import argparse
import random 

# Parameters - EDIT ME
LENGTH_threshold = 10 # sentence word length to trigger smoothing 
COUNT_threshold_rate = 0.001 # ratio of occurences to trigger smoothing
KEEP_rate = 0.2 # ratio of occurences to keep after smoothing   

DEF_INPUTFILE    = "the_world_is_tweets.csv"
DEF_OUTPUTFILE   = "preprocessed_tweets.csv"

# command lines arguments for --input_file and --output_file
parser = argparse.ArgumentParser(description="Preprocess and clean-up tweets")
parser.add_argument('--input', '--input_file', dest='input_file', default=DEF_INPUTFILE,
            help='Input file. Default: %s' % (DEF_INPUTFILE))
parser.add_argument('--output', '--output_file', dest='output_file', default=DEF_OUTPUTFILE,
            help='Output file. Default: %s' % (DEF_OUTPUTFILE))
args = parser.parse_args()

# read data and create dataframe
df = pd.read_csv(args.input_file, header = 0, index_col = None)

# csv must have 2 columns
if len(df.columns) != 2:
    print("Input CSV must have 2 columns: id, sentence")
    exit()

# rename columns and clean sentences
df.columns = ['id', 'sentence']
df['sentence'] = df['sentence'].str.lower().str.strip()

# Display input data
print("Input messages")
print(df.head())
print("Number of unprocessed messages: " + str(len(df)))

# Smooth repetitions of long and frequent texts
raw_texts = df["sentence"]
duplicated_texts = raw_texts[raw_texts.isin(raw_texts[raw_texts.duplicated()])].drop_duplicates()
unduplicated_texts = raw_texts[~(raw_texts.isin(raw_texts[raw_texts.duplicated()]))]
smoothed_texts = []
print("\nNumber of duplicated texts:", len(duplicated_texts))

COUNT_threshold = round(len(raw_texts)*COUNT_threshold_rate)
print("\nSmoothing overly frequent texts with over {} occurences and {} words".format(COUNT_threshold, LENGTH_threshold))
for text in list(duplicated_texts):
    text_length = len(text.split()) 
    count_text = list(raw_texts).count(text)
    
    # only reduce duplicates when passing COUNT_threshold and LENGTH_threshold
    if (text_length >= LENGTH_threshold) and (count_text >= COUNT_threshold):
        count_kept = round(count_text*KEEP_rate)
        print("Reducing counts ({} -> {}) of '{}'".format(count_text, count_kept, text))
    else:    
        count_kept = count_text
    
    # add to smoothed_texts
    smoothed_texts.extend([text] * count_kept)

# Add unduplicated_texts to smoothed_texts        
smoothed_texts = smoothed_texts + list(unduplicated_texts)   

# shuffle text order
random.shuffle(smoothed_texts)

# filtering out irrelevant text for primals
def clean_text(text):   
    text = text.replace("<newline>", " ")\
        .replace("(feat", " ")\
        .replace("&amp;", "and")\
        .replace("&lt;", "<")\
        .replace("&gt;", ">")\
        .replace("<USER>", "")\
        .replace("<URL>", "")\
        .strip()
    return text  

# Apply cleaning
smoothed_texts = [clean_text(text) for text in smoothed_texts]

# Add id column and rename columns
texts = pd.DataFrame(smoothed_texts)
texts.insert(0, "message_id", ["mess_" + str(i) for i in texts.index])
texts.columns = ['message_id', 'message']

print("\nProcessed messages")
print(texts.head())
print("Number of processed messages:", len(texts))

texts_counts = texts['message'].value_counts()
print("\nExample duplicated sentences:")
print(texts_counts.head())
print("Number of unique messages:", len(texts_counts),'\n')


# Save to CSV file
texts.to_csv(args.output_file, header=True, index=False)













