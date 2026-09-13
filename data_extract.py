from datasets import load_dataset



def dataset_retriver(path):
    try:
        print('Retriving the training data from Hugging Face')
        input_dataset = load_dataset(path, split='train')
        print('The training data has been retrived')
    except Exception as e:
        print(f'The training data was not retrived from Hugging Face due to this error: {e}')



    #med_dataset = dataset_retriver(path)
    sub_sized_dataset = input_dataset.shuffle(seed=42).select(range(100))

    return sub_sized_dataset