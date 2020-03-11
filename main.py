from ModelGen.Corpus import Corpus
from ModelGen.Model import Model
from ModelGen.Query import Query
from ModelGen.SentenceRanker import SentenceRanker, ScoredSentence
from operator import itemgetter

import pprint

corpus = Corpus('./WikiCorpus/WaterGateText/AA/wiki_00', regen=False)
concepts = corpus.get_concepts()
# print("\nExample of concepts found: {}\n".format(concepts[:3]))

model = Model(corpus.get_concepts(),topics=40)
query = Query(corpus, model)

concept2="impeachment"
concept3="stone"
concept4="speech"
concept5="president"

print("Cohernece value {}\n".format(model.coherence_val()))
found_docs, query_topic_dist = query.retrieve_docs([concept2, concept3, concept4, concept5])
# print("Found {} sentences for query concepts: {}, {}\nShowing first 4:".format(len(found_docs), concept3, concept2))
# print(found_docs[:3])

sentence_rank = SentenceRanker(model, corpus, query_topic_dist)

scores = sentence_rank.score_sentences(found_docs)

scores = sentence_rank.sort_sentences(scores)

for score in scores[:4]:
    score.print()

# sorted_scores = sorted(scores, key=itemgetter(1), reverse=True)
# pprint.pprint(sorted_scores[:10])

# print(model.compute_coherence_values(30,step=1))


# model.show_model()

