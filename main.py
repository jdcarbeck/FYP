from ModelGen.Corpus import Corpus

corp = Corpus('./WikiCorpus/WaterGateText/AA/wiki_00')
print(len(corp.get_concepts()))


