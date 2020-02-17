from ModelGen.ConceptExtract import Concepts
from ModelGen.Corpus import Corpus

"""Takes wiki dump and creates a LDA model from dump "Concepts" """
def model_from_wiki():
    pass

corp = Corpus('./WikiCorpus/WaterGateText/AA/wiki_00')
print(len(corp.get_concepts()))


