from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    SparseVector,
    PointStruct,
    QueryRequest,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    ScoredPoint,
)
import pandas as pd


def quadrant_collection_creation(collection_name='pqa_artifical'):
    client=QdrantClient(":memory:")
    client.create_collection(
        collection_name=collection_name,
        #Dense Vector Config
        vectors_config={
            'text-dense': VectorParams(
                size=1024,
                distance=Distance.COSINE
            )
        },
        #Sparse Vector Config

        sparse_vectors_config={
            'text-sparse': SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=False,
                )
            )
        }
    )

    print('The collection has been created')
    return client


def make_points(path='Medical_Artifact.pkl'):
    df = pd.read_pickle(path)
    df = df.reset_index(drop=True)

    sparse_vectors=df['sparse_embedding'].tolist()
    dense_vectors=df['dense_embedding'].tolist()
    med_info=df['combined_text'].tolist()
    rows=df.to_dict(orient='records')

    points=[]


    for idx, (text, sparse_vec, dense_vector) in enumerate(zip(med_info, sparse_vectors, dense_vectors)):

        sparse_vector= SparseVector(
            indices=sparse_vec.indices.tolist(),
            values=sparse_vec.values.tolist()
        )

        point=PointStruct(
            id=idx,
            payload={
                'text': text,
                'question_id':rows[idx]['pubid'],
            },
            vector={
                'text-sparse': sparse_vector,
                'text-dense': dense_vector.tolist(),
            }
        )

        points.append(point)

    print('The points has been loaded into Qdrant DB')
    return points