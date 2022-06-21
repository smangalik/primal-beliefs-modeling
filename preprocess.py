import pandas as pd

# read data
df = pd.read_csv("the_world_is_tweets.csv", header = 0, index_col = None)


# get the first sentence of tweets
print(df['sentence'].head())
df['perfect_text'] = df['sentence'].str.lower()


# reduce duplicates for long and highly repeatitive texts
perfect_text = df["perfect_text"]
duplicated_text = perfect_text[perfect_text.isin(perfect_text[perfect_text.duplicated()])]
duplicated_text = duplicated_text.drop_duplicates()
unduplicated_text = perfect_text[~(perfect_text.isin(perfect_text[perfect_text.duplicated()]))]
reduced_duplicated_text = []
COUNT_threshold = 15    # MAY CHANGE
LENGTH_threshold = 10   # MAY CHANGE
KEEP_rate = 0.2        # MAY CHANGE
for text in list(duplicated_text):
    text_length = len(text.split()) 
    count_text = list(perfect_text).count(text)
    
    # only reduce duplicates when passing COUNT_threshold and LENGTH_threshold
    if (text_length>=LENGTH_threshold) and (count_text>= COUNT_threshold):
        print("deteceted text: {} - count: {}".format(text, str(count_text)))
        count_kept = round(count_text*KEEP_rate)
    else:    
        count_kept = count_text
    
    # add to reduced_duplicated_text
    reduced_duplicated_text.extend([text] * count_kept)
# add unduplicated_text to reduced_duplicated_text        
reduced_duplicated_text = reduced_duplicated_text + list(unduplicated_text)   
# shuffle
import random 
random.shuffle(reduced_duplicated_text)
print("len reduced_duplicated_text: " + str(len(reduced_duplicated_text)))


# process symbols
for i, text in enumerate(reduced_duplicated_text):
    ### filtering irrelevant primals
    if "<newline>" in text:
        text = text.replace("<newline>", " ")
    if "(feat" in text:
        text = text.replace("(feat", " ")    
    if "&amp" in text:
        text = text.replace("&amp;", "and")            
    if "&lt" in text:
        text = text.replace("&lt;", "<")  
    if "&gt" in text:
        text = text.replace("&gt;", ">")  
    reduced_duplicated_text[i] = text


# create new ids  
texts = pd.DataFrame(reduced_duplicated_text)
# add id and column names
texts.insert(0, "message_id", ["mess_" + str(i) for i in texts.index])   # use the automatic index
texts.columns = ['message_id', 'message']
# stats 
print("Number of messages: " + str(len(texts)))
texts_counts = texts['message'].value_counts()
print("Number of unique messages: " + str(len(texts_counts)))
print(texts_counts[:10])


# save to csv file
output_file = "preprocessed_tweets.csv"
texts.to_csv(output_file, header = True, index = False)













