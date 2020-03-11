import pulp as p
from .Corpus import Corpus

class Summary:
    def __init__(self, doc: [String], corpus: Corpus):
        self.doc = doc
        self.term_freq = {} 
        """
        need to have count of sentences
        term sentence frequency
        """

    def doc_summary(self, document, sen_len=250, alpha=0.5):
        """
            sentences needs to be dictionary with terms in them and document needs to have con
            documents = {
                sentences: {
                    'sentences text': [concepts]
                }
                concepts: [concepts]
            }

        """
        sentences = document['sentences']

        lp_problem = p.LpProgram('problem', p.LpMaximize)
        x_i = p.LpVariable.dicts('xi', sentences, cat='Binary')
        x_j = p.LpVariable.dicts('xj', sentences, cat='Binary')
        constriant_len = p.LpAffineExpression((len(i) + len(j)) * (x_i[i]*x_j[j]) <= sen_len for i in sentences for j in sentences)
        objective = p.LpAffineExpression( \
            (alpha * ((self.cos_sim(document.con, i.con) + self.cos_sim(document.con, j.con) - self.cos_sim(i.con, j.con))*(x_i[i] * x_j[j])) for i in sentences for j in sentences) \
                + \
            ((1 - alpha) * ((self.ngd_sim(document.con, i.con) + self.ngd_sim(document.con, j.con) - self.ngd_sim(i.con, j.con))*(x_i[i] * x_j[j])) for i in sentences for j in sentences)
        )
        lp_problem += objective
        lp_problem += constriant_len

        lp_problem.writeLP("CheckLpProgram.lp")
        lp_problem.solve()

    def cos_sim(self, item1_con, item2_con):
        numerator = 0
        item1_con_freq = [self.con_freq[con] for con in item1_con]
        item2_con_freq = [self.con_freq[con] for con in item2_con]
        pass

    def ngd_sim(self):
        pass