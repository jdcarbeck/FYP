from ModelGen.Corpus import Corpus
from ModelGen.Model import Model

corp = Corpus('./WikiCorpus/WaterGateText/AA/wiki_00', regen=False)
concepts = corp.get_concepts()
model = Model(concepts)
# print(len(concepts))
limit = 23
step = 1
print(model.compute_coherence_values(limit, step=step, start=22))
model.show_model()