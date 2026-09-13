from qdrant_client.conversions.common_types import QueryRequest, SparseVector

from Embedding_Generation import make_sparse_embedding, make_dense_embedding



def search(client,query_text, collection_name, top_k=5):

    input_sparse_vectors=make_sparse_embedding([query_text])
    input_dense_vectors=make_dense_embedding([query_text])


    search_results=client.query_batch_points(
        collection_name=collection_name,
        requests=[
            QueryRequest(
                query=input_dense_vectors[0].tolist(),
                using='text-dense',
                limit=top_k,
                with_payload=True
            ),

            QueryRequest(
                query=SparseVector(
                    indices=input_sparse_vectors[0].indices.tolist(),
                    values=input_sparse_vectors[0].values.tolist(),
                ),
                using='text-sparse',
                limit=top_k,
                with_payload=True
            )
        ]
    )

    query_result= [search_results[0].points, search_results[1].points]


    dense_results  = query_result[0]
    sparse_results = query_result[1]

    print(f"Query: '{query_text}'")
    print(f"\n{'=' * 60}")
    print(f"DENSE (Semantic) Results — Top 5:")
    print(f"{'=' * 60}")

    for i, point in enumerate(dense_results[:top_k]):
        text= point.payload['text'].split('\n')[0][:100]
        print(f' {i+1}. [Score: {point.score:.4f}] {text}')

    print(f"\n{'=' * 60}")
    print(f"DENSE (Semantic) Results — Top 5:")
    print(f"{'=' * 60}")

    for i, point in enumerate(sparse_results[:top_k]):
        text= point.payload['text'].split('\n')[0][:100]
        print(f' {i+1}. [Score: {point.score:.4f}] {text}')

    return [dense_results, sparse_results]





