from gensim.models.ldamodel import LdaModel
from .Corpus import Corpus
from .ConceptExtract import Concepts

class SentenceRanker:
    def __init__(self, model: LdaModel, docs: Corpus, topic_dist):
        self.model = model
        self.docs = docs
        self.topic_dist = topic_dist
        self.max_topic = max(topic_dist[0], key=lambda x:x[1])

    def get_best(self, amount, method):
        # Compute Max Score
        # Sum Score
        # Max TF score
        # Sum Tf score
        pass

    """A sentence is made of a list of concepts"""
    def score_sentences(self, sentences: [str]):
        concepts = []
        for sent in sentences:
            if sent not in self.docs.sen2con.keys():
                pass
            else:
                concepts.append(self.docs.sen2con[sent])
        print(len(sentences))
        print(len(concepts))

        # Now have a representation of each sentece as
        

        # Find Topic Score
        # Find TF Score
        pass

    """Concept scoring, using model to find word relation topic"""
    def concept_score(self, concept: str):
        # Convert concept to ID value
        con_id = self.model.dct.token2id[concept]
        # Find word topic Score
        topic_con_vec = self.model.lda_model.get_term_topics(con_id)
        # set max topic concept score
        max_topic_con_score = self.get_topic_score(self.max_topic[0], topic_con_vec)
        # determine sum score
        con_sum_score = self.get_sum_score(topic_con_vec)

        return max_topic_con_score, con_sum_score
    
    def get_topic_score(self, topic_id, topic_dist):
        topic_score = 0
        for topic in topic_dist:
            if topic[0] == topic_id:
                topic_score = topic[1]
        return topic_score

    """Summation of each of the product of a topic2doc and con2topic"""
    def get_sum_score(self, con_topic_dist):
        sum_score = 0
        for topic in self.topic_dist[0]:
            con_topic_score = self.get_topic_score(topic[0], con_topic_dist)
            sum_score += (topic[1] * con_topic_score)
        return sum_score