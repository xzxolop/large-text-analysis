import pandas as pd

d = {'word': ['aaa', 'www'], 'count': [3, 4]}
word_frequency_df = pd.DataFrame(data = d)

print(word_frequency_df)
print(word_frequency_df['word'])

word_frequency_df.add('sss', 3)

print(word_frequency_df)
