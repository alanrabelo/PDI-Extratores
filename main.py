from data_manager import DataManager
from extractors import Extractors
from classifiers import Classifiers

data_manager = DataManager()
extractor = Extractors()
classifiers = Classifiers()

X, y, encoder = data_manager.loadData()

encoded_x = extractor.glcm(X)
classifiers.classify(encoded_x, y, title='GLCM - ')
encoded_x = extractor.lbp(X)
classifiers.classify(encoded_x, y, title='LBP - ')
encoded_x = extractor.huMoments(X)
classifiers.classify(encoded_x, y, title='HU - ')
