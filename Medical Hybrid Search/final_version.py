from data_cleanser import data_cleanser
from Embedding_Generation import embed_generation
from Quadrant_DB import quadrant_collection_creation, make_points
from Search import search
from RRF import rrf

path='PubMedQA/pqa_artificial'


def rank_list(search_results):
    return [(point.id, rank+1) for rank, point in enumerate(search_results)]

try:
    print('The data extract and cleaning is in progress')
    data_cleanser(path)

except Exception as e:
    print(f'The following data extract and clensing is failed due to this error-->{e}')

cleaned_data_file='landing/cleaned/Med_Artifical.csv'

try:
    print('The embedding process is in progress')
    embed_generation(cleaned_data_file)

except Exception as e:
    print(f'The following embedding process is failed due to this error-->{e}')


pkl_file_path='landing/Embedding/Medical_Artifact.pkl'

collection_name='Medical_Artifact'
client= quadrant_collection_creation(collection_name)
points=make_points(pkl_file_path)

client.upsert(collection_name,points)
print(f"✅ {len(points)} points indexed in Qdrant!")


sample_query='Tell me about Keratin'
search_results=search(client=client,query_text=sample_query, collection_name=collection_name, top_k=5)


dense_rank_list=rank_list(search_results[0])
sparse_rank_list=rank_list(search_results[1])


rrf_rank_list=rrf([dense_rank_list,sparse_rank_list])



print(f"\n🏆 HYBRID SEARCH RESULTS for: '{sample_query}'")
print(f"{'='*70}")

fused_records= client.retrieve(
    collection_name=collection_name,
    ids=[item[0] for item in rrf_rank_list]
)


record_by_id ={r.id: r for r in fused_records}

for rank, (item_id, score) in enumerate(rrf_rank_list, 1):
    record = record_by_id[item_id]
    title = record.payload['text'].split('\n')[0][:75]

    # Check if this item was in dense, sparse, or both
    in_dense = any(id == item_id for id, _ in dense_rank_list)
    in_sparse = any(id == item_id for id, _ in sparse_rank_list)
    source = "BOTH" if (in_dense and in_sparse) else ("Dense only" if in_dense else "Sparse only")

    print(f"  {rank:>2}. [{source:<12}] {title}")