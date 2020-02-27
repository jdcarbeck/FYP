from ModelGen.Corpus import Corpus
from ModelGen.Model import Model
from ModelGen.Query import Query
from ModelGen.SentenceRanker import SentenceRanker

import pprint

corpus = Corpus('./WikiCorpus/WaterGateText/AA/wiki_00', regen=True)
concepts = corpus.get_concepts()
# print("\nExample of concepts found: {}\n".format(concepts[:3]))

model = Model(corpus.get_concepts())
query = Query(corpus, model)

concept1="watergate"
concept2="burglars"

# print("Cohernece value {}\n".format(model.coherence_val()))
found_docs, query_topic_dist = query.retrieve_docs([concept1, concept2])
print("Found {} sentences for query concepts: {}, {}\nShowing first 4:".format(len(found_docs), concept1, concept2))
print(found_docs)

sentence_rank = SentenceRanker(model, corpus, query_topic_dist)

sentence_rank.score_sentences(found_docs)



# model.show_model()

