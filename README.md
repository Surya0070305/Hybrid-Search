# Hybrid-Search
This Repo is used to understand the capability of Hybrid Search(Dense+Sparse) 

The data is pulled from Hugging Face Medical Dataset
https://huggingface.co/datasets/qiaojin/PubMedQA

The data is cleaned and pushed into Qdrant DB. 
Indexed using the id points.

The Sparse and Dense Models that are used are mentioned below

sparse_text_embedding_model='prithvida/Splade_PP_en_v1'
dense_text_embedding_model='BAAI/bge-large-en-v1.5'

This will give the separate results as well as combined results.

Query: 'Tell me about Keratin'


DENSE (Semantic) Results — Top 5:

 1. [Score: 0.6458] Keratin 8 is a major component of intermediate filaments in single-layered epithelia of the gastroin
 2. [Score: 0.5542] To investigate the presence or absence of Toll-like receptor (TLR)-2 and TLR-4 in synovial tissues c
 3. [Score: 0.5512] To determine bone mass using quantitative phalangeal bone ultrasound in young coeliac patients after
 4. [Score: 0.5482] To investigate the effect of liver X receptor agonist T0901317 on transforming growth factor-β1 (TGF
 5. [Score: 0.5470] We previously reported significant variations in oxidation status and molecular length among sources

DENSE (Semantic) Results — Top 5:

 1. [Score: 16.8209] Keratin 8 is a major component of intermediate filaments in single-layered epithelia of the gastroin
 2. [Score: 11.9558] To investigate the effect of liver X receptor agonist T0901317 on transforming growth factor-β1 (TGF
 3. [Score: 5.0298] Melatonin, an indolamine produced and secreted predominately by the pineal gland, exhibits a variety
 4. [Score: 5.0082] Osteopontin, an important immune modulator and oncogenic promoter, is upregulated in H. pylori-infec
 5. [Score: 4.1247] Telmisartan is an angiotensin II receptor blocker and selective modulator of peroxisome proliferator


After applying the RRF between DENSE result and SPARSE result. The Hybrid Search result are as shown below for the sample query.

🏆 HYBRID SEARCH RESULTS for: 'Tell me about Keratin'
======================================================================
   1. [BOTH        ] Keratin 8 is a major component of intermediate filaments in single-layered 
   2. [BOTH        ] To investigate the effect of liver X receptor agonist T0901317 on transform
   3. [Dense only  ] To investigate the presence or absence of Toll-like receptor (TLR)-2 and TL
   4. [Dense only  ] To determine bone mass using quantitative phalangeal bone ultrasound in you
   5. [Sparse only ] Melatonin, an indolamine produced and secreted predominately by the pineal 
   6. [Sparse only ] Osteopontin, an important immune modulator and oncogenic promoter, is upreg
   7. [Dense only  ] We previously reported significant variations in oxidation status and molec
   8. [Sparse only ] Telmisartan is an angiotensin II receptor blocker and selective modulator o


