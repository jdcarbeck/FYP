from ModelGen.Corpus import Corpus
from ModelGen.Model import Model

corp = Corpus('./WikiCorpus/WaterGateText/AA/wiki_00')
concepts = corp.get_concepts()
model = Model(concepts)
# print(len(concepts))
limit = 21
step = 4
print(model.compute_coherence_values(limit, step=step, start=20))



