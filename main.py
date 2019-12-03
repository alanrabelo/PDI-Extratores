from data_manager import DataManager
from extractors import Extractors
from classifiers import Classifiers

data_manager = DataManager()
extractor = Extractors()
classifiers = Classifiers()

X, y, encoder = data_manager.loadData()

results = []
encoded_x1 = extractor.glcm(X)
classifiers.classify(results, encoder, y, title='GLCM - ')

encoded_x2 = extractor.lbp(X)
classifiers.classify(results, encoder, y, title='LBP - ')

encoded_x3 = extractor.huMoments(X)
classifiers.classify(results, encoder, y, title='HUMomments - ')


for index, result in enumerate(encoded_x1):
    new_result = result

    for x2 in encoded_x2[index]:
        new_result.append(x2)
    for x3 in encoded_x3[index]:
        new_result.append(x3)

    results.append(new_result)

classifiers.classify(results, encoder, y, title='Mix - ')
