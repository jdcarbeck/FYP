from .Corpus import Corpus
from .Model import Model
from .ConceptExtract import Concepts
from gensim.matutils import cossim

class Query:
    def __init__(self, corpus: Corpus, model: Model):
        self.corpus = corpus
        self.model = model

    """
        Get the documents associated with a list of concepts
        gen a set of keywords to be associated with the concept chain
            • get documents (sentences) that relate to a given concept
            • combine into a document to use on the model
            • produce keywords specified that relate to given concepts
    """
    def get_concept_chain(self, concepts: [str], keywords=10):
        # find sents relating to given concepts
        unseen_sent = []
        for con in concepts:
            sentences = self.corpus.con2sen[con]
            for sent in sentences:
                if sent not in unseen_sent:
                    unseen_sent.append(sent)
        unseen_doc = " ".join(unseen_sent)
        # concept extract the unseen_doc
        unseen_concepts = Concepts(unseen_doc).get()

        dist = self.model.topic_dist(unseen_concepts)
        # 0th element has the topic break down of document
        # 1st elemtn has the word to topic relation
        # 2nd element has the word to topic breakdown
        return dist
        # print(dist[2])

    def retrieve_docs(self, concepts: [str], similarity = 0.70):
        # topic_dist = self.model.topic_dist(concepts)
        topic_dist = self.get_concept_chain(concepts)
        similar_docs = []
        count = 0
        for doc in self.corpus.docs:
            for sent in doc.sen2con.keys():
                sent_concepts = doc.sen2con[sent]
                doc_dist = self.model.topic_dist(sent_concepts)
                sim = cossim(doc_dist[0], topic_dist[0])
                if sim > similarity:
                    count+=1
                    similar_docs.append(sent)
        print("{}%% of documents found similar".format(count/len(self.corpus.sen2con.keys())))
        return similar_docs, topic_dist

