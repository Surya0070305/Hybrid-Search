from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding

def dataset_retriver(path):
    try:
        print('Retriving the training data from Hugging Face')
        input_dataset = load_dataset(path, split='train')
        print('The training data has been retrived')
    except Exception as e:
        print(f'The training data was not retrived from Hugging Face due to this error: {e}')

    return input_dataset


path='PubMedQA/pqa_artificial/'

med_dataset=dataset_retriver(path)
sub_sized_dataset=med_dataset.shuffle(seed=42).select(range(100))

print('Printing the sub dataset columns')
print(sub_sized_dataset)

print('Converting the dataframe to pandas dataframe for more data cleansing')
med_df=sub_sized_dataset.to_pandas()
print(med_df)

print('printing the lenght of the dataframe before duplicating')
print(len(med_df))
print('Dropping duplicates on the pubid column')
med_df=med_df.drop_duplicates(subset=['pubid'])
print('printing the lenght of the dataframe after duplicating')
print(len(med_df))


print('Drooping the null values in the columns question and long answer')
med_df=med_df.dropna(subset=['question','long_answer'])
print('printing the lenght of the dataframe after duplicating')
print(len(med_df))


print('The distribution of the column question')
print(med_df['final_decision'].value_counts(normalize=True)*100)
print(med_df['final_decision'].describe())
print(med_df['final_decision'].hist())
#plt.show()

'''
#med_df.to_csv('final_data.csv')

#context_cols=med_df.json_normalize(med_df['context'])
context_cols=med_df['context'].apply(pd.Series)
#conbined_text=context_cols['contexts'] + med_df['long_answer']
print(context_cols['contexts']+ '\n' + med_df['long_answer'])
'''

# Combines elements row-by-row safely converting elements to text
# 1. Join the list items in 'contexts' into a single string separated by space
context_cols=med_df['context'].apply(pd.Series)
med_df['context_text'] = context_cols['contexts'].str.join(' ')


new_df= med_df['context_text'] + '\n' + med_df['long_answer']
med_df['combined_text'] = new_df


med_df.to_csv('final_data_1.csv')






dense_model_name='BAAI/bge-large-en-v1.5'
sparse_model_name='prithvida/Splade_PP_en_v1'

sparse_model=SparseTextEmbedding(model_name=sparse_model_name, batch_size=32)

dense_model=TextEmbedding(model_name=dense_model_name, batch_size=32)

def make_sparse_embedding(text:list[str]):
    return list(sparse_model.embed(text, batch_size=32))

def make_dense_embedding(text:list[str]):
    return list(dense_model.embed(text, batch_size=32))