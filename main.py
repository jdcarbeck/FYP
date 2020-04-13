from ModelGen.Corpus import Corpus
from ModelGen.Model import Model
from ModelGen.Query import Query
from ModelGen.SentenceRanker import SentenceRanker, ScoredSentence
from ModelGen.Summary import Summary
from operator import itemgetter
from ModelGen.ConceptExtract import Concepts
import numpy

import pprint as pp

corpus = Corpus([], filename='./WikiCorpus/WaterGateText/AA/wiki_00', regen=False)
concepts = corpus.get_concepts()
# print("\nExample of concepts found: {}\n".format(concepts[:3]))

model = Model(corpus.get_concepts(),topics=10)
query = Query(corpus, model)

text = "In the context of the Watergate scandal, Operation Gemstone was a proposed series of clandestine or illegal acts, first outlined by G. Gordon Liddy in two separate meetings with three other individuals: then-Attorney General of the United States, John N. Mitchell, then-White House Counsel John Dean, and Jeb Magruder, an ally and former aide to H.R. Haldeman, as well as the temporary head of the Committee to Re-elect the President, pending Mitchell's resignation as Attorney General."
pp.pprint(text)

concepts = Concepts(text).get()
print(concepts)

# concepts = ["operation gemstone"]

print("Cohernece value {}\n".format(model.coherence_val()))
found_docs, query_topic_dist = query.retrieve_docs(concepts, similarity=0.90)


# print("Found {} sentences for query concepts: {}, {}\nShowing first 4:".format(len(found_docs), concept3, concept2))
# print(found_docs[:3])

summary = Summary(found_docs, corpus)
summary_list = summary.doc_summary()
print(summary_list)

# sentence_rank = SentenceRanker(model, corpus, query_topic_dist)

# scores = sentence_rank.score_sentences(found_docs)

# scores = sentence_rank.sort_sentences(scores)

# for score in scores[:4]:
#     score.print()

# sorted_scores = sorted(scores, key=itemgetter(1), reverse=True)
# pprint.pprint(sorted_scores[:10])

# print(model.compute_coherence_values(30,step=1))


# model.show_model()

