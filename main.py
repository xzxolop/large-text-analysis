import kagglehub
import pandas as pd
from pathlib import Path

import core

path = Path(kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset"))
path_to_comments = path.joinpath("the-reddit-dataset-dataset-comments.csv")

comments_df = pd.read_csv(path_to_comments)
comments_body = comments_df["body"]

df = pd.DataFrame()
df['document'] = comments_body
print(df)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess(text):
    if not isinstance(text, str):
        return ''

    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

df['processed'] = df['document'].apply(preprocess)
print(df)

m = core.make_map_most_popular("data", df['processed'])

sorted_items = sorted(m.items(), key=lambda x: x[1], reverse=True)    

core.print_searched_words(sorted_items, 10)

def create_list(words, size: int):
    lst = []
    for i in range(0, size+1):
        print(words[i][1])
        lst.append(words[i][0])
        
    return lst            

top_ten_words = create_list(sorted_items, 10)
print(top_ten_words)

for word in top_ten_words:
    print('')
    print(word)

    m = core.make_map_most_popular(word, df['processed'])
    sorted_items = sorted(m.items(), key=lambda x: x[1], reverse=True)    
    core.print_searched_words(sorted_items, 10)
