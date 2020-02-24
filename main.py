from ModelGen.Corpus import Corpus
from ModelGen.Model import Model
from ModelGen.Query import Query

corpus = Corpus('./WikiCorpus/WaterGateText/AA/wiki_00', regen=False)
model = Model(corpus.get_concepts())
query = Query(corpus, model)

query.get_concept_chain(["article", "impeachment"])
print()
query.get_concept_chain(["headquarters", "watergate burglaries"])
# model.show_model()
# print(corpus.sents_with_con("united states"))
# concepts = corp.get_concepts()
# model = Model(concepts)
# # print(len(concepts))
# limit = 11
# step = 1
# print(model.compute_coherence_values(limit, step=step, start=10))
# model.show_model()

