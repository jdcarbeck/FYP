import codecs
from bs4 import BeautifulSoup
from .ConceptExtract import Concepts

class Corpus:
    docs = []
    concepts = []

    def __init__(self, filename):
        self.docs = self.generate_docs(filename)

    def get_concepts(self):
        return self.concepts

    def generate_docs(self, filename):
        with codecs.open(filename, encoding='utf-8') as f:
            data = f.read()
        soup = BeautifulSoup(data, features="html.parser")
        documents = soup.find_all('doc')
        size = len(documents)
        print(("Processing: {} wiki articles").format(size))
        for doc in documents:
            title = doc.get('title')
            url = doc.get('url')
            uid = doc.get('id')
            text = doc.get_text().replace("\n", "")
            d = Document(title, url, uid, text)
            self.docs.append(d)
            self.concepts = self.concepts + d.concepts
        print("Finished!")

class Document:
    def __init__(self, title, url, uid, text):
        self.title = title
        self.url = url
        self.uid = uid
        self.text = text
        self.concepts = Concepts(text).get()