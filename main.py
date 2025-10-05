import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("pavellexyr/the-reddit-dataset-dataset")
print("Path to dataset files:", path)

posts_df = pd.read_csv(path + "\\the-reddit-dataset-dataset-posts.csv")
comments_df = pd.read_csv(path + "\\the-reddit-dataset-dataset-comments.csv")

print(posts_df)
print(comments_df)


