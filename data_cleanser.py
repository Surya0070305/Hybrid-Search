from data_extract import dataset_retriver
import pandas as pd







def data_cleanser(path):
    input_dataset = dataset_retriver(path)
    print('Converting the dataframe to pandas dataframe for more data cleansing')
    med_df=input_dataset.to_pandas()

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

    context_cols = med_df['context'].apply(pd.Series)
    med_df['context_text'] = context_cols['contexts'].str.join(' ')

    new_df = med_df['context_text'] + '\n' + med_df['long_answer']
    med_df['combined_text'] = new_df
    print('Now the data has been cleaned')
    med_df.to_csv('landing/cleaned/Med_Artifical.csv')


    return print('The data has been written successfully')

