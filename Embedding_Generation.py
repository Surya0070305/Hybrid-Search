from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding
import pandas as pd
import time as time

sparse_text_embedding_model='prithvida/Splade_PP_en_v1'
dense_text_embedding_model='BAAI/bge-large-en-v1.5'


def make_sparse_embedding(text:list[str]):
    sparse_model = SparseTextEmbedding(model_name=sparse_text_embedding_model, batch_size=32)
    return list(sparse_model.embed(text, batch_size=32))

def make_dense_embedding(text:list[str]):
    dense_model = TextEmbedding(model_name=dense_text_embedding_model, batch_size=32)
    return list(dense_model.embed(text, batch_size=32))


def embed_generation(path):

    df=pd.read_csv(path)
    combined_info=df['combined_text'].tolist()

    start = time.perf_counter()
    print("⏳ Generating sparse embeddings (SPLADE)...")
    df["sparse_embedding"] =make_sparse_embedding(combined_info)
    print("✅ Done!")
    end = time.perf_counter()
    print(f"Elapsed: {end - start:.4f} seconds")


    start = time.perf_counter()
    print("⏳ Generating dense embeddings (BGE-Large)...")
    df["dense_embedding"] =make_dense_embedding(combined_info)
    print("✅ Done!")
    end = time.perf_counter()
    print(f"Elapsed: {end - start:.4f} seconds")

    df.to_pickle('landing/Embedding/Medical_Artifact.pkl')

    return print('The dataset has been SPARSE and DENSE embedded successfully!')




