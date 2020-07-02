import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

class Result:
    CLICKBAIT = "clickbait"
    NOT_CLICKBAIT = "no-clickbait"

#each article is broken into 9 different categories gathered from the dataset from the clickbait-classifer.org
class Article:

    def __init__(self, id, postTimestamp, postText, postMedia, targetTitle, targetDescription, targetKeywords, targetParagraphs, targetCaptions):
        self.id = id
        self.postTimestamp = postTimestamp
        self.postText = postText
        self.postMedia = postMedia
        self.targetTitle = targetTitle
        self.targetDescription = targetDescription
        self.targetKeywords = targetKeywords
        self.targetParagraphs = targetParagraphs
        self.targetCaptions = targetCaptions

    def getID(self):
        return self.id

    def getPostTimestamp(self):
        return self.postTimestamp

    def getPostText(self):
        return self.postText

    def getPostMedia(self):
        return self.postMedia

    def getTargetTitle(self):
        return self.targetTitle

    def getTargetDescription(self):
        return self.targetDescription

    def getTargetKeywords(self):
        return self.targetKeywords

    def getTargetParagraphs(self):
        return self.targetParagraphs

    def getTargetCaptions(self):
        return self.targetCaptions

class Truth:

    def __init__(self, id, truthJudgments, truthMean, truthMedian, truthMode, truthClass):
        self.id = id
        self.truthJudgments = truthJudgments
        self.truthMean = truthMean
        self.truthMedian = truthMedian
        self.truthMode = truthMode
        self.truthClass = truthClass

    def getID(self):
        return self.id

    def getTruthJudgements(self):
        return self.truthJudgments

    def getTruthMean(self):
        return self.truthMean

    def getTruthMedian(self):
        return self.truthMedian

    def getTruthMode(self):
        return self.truthMode

    def getTruthClass(self):
        return self.truthClass

    def getTRUTH(self):
        if self.truthClass == 'clickbait':
            return Result.CLICKBAIT

        elif self.truthClass == 'no-clickbait':
            return Result.NOT_CLICKBAIT

        else:
            return -1

class Article_Truth:

    def __init__(self, postText, targetParagraphs, Truth):
        self.postText = postText
        self.targetParagraphs = targetParagraphs
        self.Truth = Truth

    def getPostText(self):
        return self.postText

    def getTargetParagraphs(self):
        return self.targetParagraphs

    def getTruth(self):
        return self.Truth

class ReviewContainer:
    def __init__(self, text):
        self.text = text

    def getPostText(self):
        return [x.postText for x in self.text]

    def getTargetParagraphs(self):
        return [x.targetParagraphs for x in self.text]

    def getResult(self):
        return [x.Truth for x in self.text]

file_name = "instances.jsonl"
second_file = "truth.jsonl"
clickbait_article = []
truth = []
clickbait_article_truth = []

with open(second_file, encoding='utf-8') as file:
    for line in file:
        someTruth = json.loads(line)
        truth.append(Truth(someTruth['id'], someTruth['truthJudgments'], someTruth['truthMean'], someTruth['truthMedian'], someTruth['truthMode'], someTruth['truthClass']))

with open(file_name, encoding='utf-8') as file:
    for line in file:
        article = json.loads(line)
        clickbait_article.append(Article(article['id'], article['postTimestamp'], article['postText'], article['postMedia'], article['targetTitle'].lower(), article['targetDescription'], article['targetKeywords'], article['targetParagraphs'], article['targetCaptions']))


for i in range(0, len(clickbait_article)):
    clickbait_article_truth.append(Article_Truth(clickbait_article[i].getPostText(), clickbait_article[i].getTargetParagraphs(), truth[i].getTRUTH()))

training, test = train_test_split(clickbait_article_truth, test_size=.3, random_state=42)
train_container = ReviewContainer(training)
test_container = ReviewContainer(test)

train_x = train_container.getTargetParagraphs()
train_y = train_container.getResult()

print("Truth contains, clickbait: %d\tnot-clickbait: %d\n" % (train_y.count(1), train_y.count(0)))

test_x = test_container.getTargetParagraphs()
test_y = test_container.getResult()

vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

#svm classifier
clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)
clf_svm.predict(test_x_vectors[0])
print("SVM Linear: ", end="")
print(clf_svm.score(test_x_vectors, test_y))
print("F1 score: ", end="")
print(f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels=[Result.CLICKBAIT, Result.NOT_CLICKBAIT]))

# # #decision tree classifier
clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)
clf_dec.predict(test_x_vectors[0])
print("Decision Tree: ", end="")
print(clf_dec.score(test_x_vectors, test_y))
print("F1 score: ", end="")
print(f1_score(test_y, clf_dec.predict(test_x_vectors), average=None))

#
#logistic regression classifier
clf_log = LogisticRegression()
clf_log.fit(train_x_vectors, train_y)
clf_log.predict(test_x_vectors[0])
print("Logistic Regression: ", end="")
print(clf_log.score(test_x_vectors, test_y))
print("F1 score: ", end="")
print(f1_score(test_y, clf_log.predict(test_x_vectors), average=None))



test_set = ['check this article out', 'us soccer should start answering']
new_test = vectorizer.transform(test_set)
print(clf_svm.predict(new_test))


#Gaussian naive bayes classifier
# clf_gnb = GaussianNB()
# clf_gnb.fit(train_x_vectors, train_y)
# clf_gnb.predict(test_x_vectors[0].toarray())
# print(clf_gnb.score(test_x_vectors, test_y))
