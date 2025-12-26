import nltk
import kagglehub
from pathlib import Path
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

def load_data():
    path = kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset")
    path_to_comments = Path(path) / "the-reddit-dataset-dataset-comments.csv"
    print(f"Датасет загружен в: {path_to_comments}")

    comments_df = pd.read_csv(path_to_comments)
    
    print("comments_df:")
    print(comments_df)
#           type       id subreddit.id subreddit.name  ...                                          permalink                                               body sentiment score
# 0      comment  hyyz6g8        2r97t       datasets  ...  https://old.reddit.com/r/datasets/comments/t45...  Spatial problem: Suitability of new locations ...    0.0772     1
# 1      comment  hyyid7v        2r97t       datasets  ...  https://old.reddit.com/r/datasets/comments/sg9...  Have you tried toying around with GDELT or Ali...    0.0000     2
# 2      comment  hyxp1qp        2r97t       datasets  ...  https://old.reddit.com/r/datasets/comments/t44...  Damn random internet person of whom I know not...   -0.3851     3
# 3      comment  hyxgnyu        2r97t       datasets  ...  https://old.reddit.com/r/datasets/comments/t44...  Ah nice one. Best of luck with the baby. If yo...    0.9136     3
# 4      comment  hyxfjw6        2r97t       datasets  ...  https://old.reddit.com/r/datasets/comments/t49...  I was about to write and say this shouldn't be...    0.0762     2

    print("body:")
    print(comments_df["body"])
#0        Spatial problem: Suitability of new locations ...
#1        Have you tried toying around with GDELT or Ali...
#2        Damn random internet person of whom I know not...
#3        Ah nice one. Best of luck with the baby. If yo...
#4        I was about to write and say this shouldn't be...
# ...
# 54847
    nltk.download('stopwords', quiet=True)
    
    # TODO: также в качестве стоп-слов добавить ссылки (через рег. выражения)
    stop_words = set(stopwords.words('english'))
    print(stop_words)

    for i in range(10):
        preprocess(comments_df["body"][i])

    # Комментарий -> предложения -> удалить ненужные слова.
def preprocess(text: str):
    
    # Возвращет список предложений
    setances = nltk.sent_tokenize(text)
    print(setances)

    
