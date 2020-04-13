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
print("Cohernece value {}\n".format(model.coherence_val()))
query = Query(corpus, model)

text = "In the context of the Watergate scandal, Operation Gemstone was a proposed series of clandestine or illegal acts, first outlined by G. Gordon Liddy in two separate meetings with three other individuals: then-Attorney General of the United States, John N. Mitchell, then-White House Counsel John Dean, and Jeb Magruder, an ally and former aide to H.R. Haldeman, as well as the temporary head of the Committee to Re-elect the President, pending Mitchell's resignation as Attorney General.\n"
print("\033[33mDocument being read: \033[0m", text, "\n")

top_concepts = query.top_concepts(text)


users_knowledge = [["operation sandwedge", "political enemies", "caulfield"],["senate watergate committee","impeachment","testimony"],["october", "saturday night massacre","tapes"]]

for i, user in enumerate(users_knowledge):
    print("----------------------------------------------------------------------------------")
    print("\033[1mMODELING USER: {}\n\033[0m".format(i))
    print("\033[33mTop concepts from document:\033[0m", top_concepts)
    print("\033[33mUser knowledge sugesstion:\033[0m", user, "\n")
    query_concepts = top_concepts + user
    found_docs, query_topic_dist = query.retrieve_docs(query_concepts, similarity=0.90)
    summary = Summary(found_docs, corpus)
    summary_list = summary.doc_summary()
    links = []
    for sent in summary_list:
        title, link = corpus.get_links(sent)
        if (title, link) not in links:
            links.append((title, link))

    print("\033[32mSUMMARY:\033[0m",summary_list)
    for title, url in links:
        print(title, ": ", url)

# print(model.compute_coherence_values(30,step=1))
# model.show_model()

