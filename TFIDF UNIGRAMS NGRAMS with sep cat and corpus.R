library(tm)
library(SnowballC)
library(wordcloud)
library(ggplot2)
library(RTextTools)
library(caret)
library(caTools)
library(stringr)
library(class)
library(gmodels)
library(kernlab)
library(graph)
library(Rgraphviz)

setwd("C:/Users/Ashish/Desktop/20NEWSGROUP")

#define path for training files
Train_dir<-"C:/Users/Ashish/Desktop/20NEWSGROUP/20news-bydate-train"
Test_dir<-"C:/Users/Ashish/Desktop/20NEWSGROUP/20news-bydate-test"

Train_CorpusDir<-DirSource(Train_dir, encoding = "UTF-8", recursive=TRUE)
Test_CorpusDir<-DirSource(Test_dir, encoding = "UTF-8", recursive=TRUE)
news_train <- Corpus(Train_CorpusDir, readerControl=list(reader=readPlain,language="en"))
news_test<-Corpus(Test_CorpusDir, readerControl=list(reader=readPlain,language="en"))

str(news_train)
print(news_train[[1]]$content)
print(news_test[[1]]$content)

#construct an empty vector, to be used for holding class labels of the documents
Train_category_vec <- vector()

#loop all the files: for each document, make its parent folder name as the class label of this document, and put the class label value in classvec vector
#Train_CorpusDir$fileList  #display all the files

for (filedir in Train_CorpusDir$filelist) {
  
  classlabelTrain=basename(dirname(filedir))
  Train_category_vec=c(Train_category_vec,classlabelTrain)
  
}
#factor class vec to let R know it's a categorical variable
Train_category_vec<- factor(Train_category_vec)

#for test set
Test_category_vec<-vector()
for (filedir in Test_CorpusDir$filelist) {
  
  classlabeltest=basename(dirname(filedir))
  Test_category_vec=c(Test_category_vec,classlabeltest)
  
}
Test_category_vec<-factor(Test_category_vec)

summary(Test_category_vec)
write.csv(table(Test_category_vec),"summarydatasetTest.csv")

#Exploring the dataset
#No. of documents per category
Train_docs_summary<- as.data.frame(table(Train_category_vec))
Test_docs_summary<- as.data.frame(table(Test_category_vec))
write.csv(table(Train_category_vec),"summarydataset.csv")
summary(Train_docs_summary)
str(Train_docs_summary)

#no of documents per category in test and train set
png("traindoc1.png")
ggplot(Train_docs_summary,aes(x=Train_category_vec,y=Freq))+geom_bar(stat="identity",fill="blue")+theme(axis.text.x = element_text(angle = 45, hjust = 1))+labs(title="No. of documents per Category in Training dataset") +labs(y="No of documents", x="Category Name")
dev.off()
png("testdoc.png")
ggplot(Test_docs_summary,aes(x=Test_category_vec,y=Freq))+geom_bar(stat="identity",fill="blue")+theme(axis.text.x = element_text(angle = 45, hjust = 1))+labs(title="No. of documents per Category in Test dataset") +labs(y="No of documents", x="Category Name")
dev.off()


#no of lines per documents in test and train set
LinesPerDoc_Train<-sapply(news_train, FUN = function(x) sum(str_count(string = x, pattern = "\\w") + 1))
LinesPerDoc_Train_df<-as.data.frame(LinesPerDoc_Train)
str(LinesPerDoc_Train_df)
summary(LinesPerDoc_Train_df)
ggplot(LinesPerDoc_Train_df,aes(Train_category_vec,LinesPerDoc_Train))+geom_histogram(col="red",breaks=seq(0,300,10) ,aes(fill=..count..))+ labs(title="Histogram for lines per doc in Train Dataset") +labs(x="No of Lines per doc in Train Dataset", y="Number of Documents")

LinesPerDoc_Test<-sapply(news_test, FUN = function(x) sum(str_count(string = x, pattern = "\\n") + 1))
LinesPerDoc_Test_df<-as.data.frame(LinesPerDoc_Test)
str(LinesPerDoc_Test_df)
summary(LinesPerDoc_Test_df)
ggplot(LinesPerDoc_Test_df,aes(LinesPerDoc_Test))+geom_histogram(col="red", breaks=seq(0,300,10) ,aes(fill=..count..))+ labs(title="Histogram for lines per doc in Test dataset") +labs(x="No of Lines per doc", y="Number of Documents")


ggplot(LinesPerDoc_Train_df,aes(Train_category_vec,LinesPerDoc_Train))+geom_histogram(col="red",breaks=seq(0,300,10) ,aes(fill=..count..))+labs(title="Histogram for lines per doc in Train Dataset") +labs(x="No of Lines per doc in Train Dataset", y="Number of Documents")

#Preprocessing the corpus
#clean test and train corpus
CleanCorpus<-function(corpus_train){
  removespecial<-function(x)gsub("[^[:alnum:]]"," ",x)# remove all except alpha numeric, urls
  corpus_train<-tm_map(corpus_train,content_transformer(removespecial))
  corpus_train<-tm_map(corpus_train,tolower)
  corpus_train<-tm_map(corpus_train, removePunctuation)
  corpus_train<-tm_map(corpus_train, removeNumbers)
  mystopwords<-c(stopwords("english"),"com","edu","nntp","etc","lines","from","reply","subject","lines","re","keywords","organisation")
  corpus_train<-tm_map(corpus_train,removeWords,mystopwords)
  corpus_train<-tm_map(corpus_train,PlainTextDocument)
  corpus_train<-tm_map(corpus_train,stemDocument,language="english")
  corpus_train<-tm_map(corpus_train,stripWhitespace)
  corpus_train<-tm_map(corpus_train,PlainTextDocument)
}

news_train_cl<-CleanCorpus(news_train)
news_test_cl<-CleanCorpus(news_test)


#save the corpus
writeLines(as.character(news_train_cl), con="news_train_cl.txt")
writeLines(as.character(news_test_cl), con="mycorpus.txt")

#1)UNIGRAMS
#A)create dtm train matrix for test and train set with 0.99 sparsity with TFIDF- TO BE RUN FOR CREATING UNIGRAM TRAIN DTM FOR ALL ALGORITHMS 
dtm_Train_TFIDF_sp<-DocumentTermMatrix(news_train_cl,control=list(wordLengths=c(3,18),bounds = list(global = c(2,Inf)),weighting=function(x)weightTfIdf(x,normalize=TRUE)))
dtm_Train_TFIDF<-removeSparseTerms(dtm_Train_TFIDF_sp,0.99)
dtm_Train_TFIDF$dimnames$Docs<-1:nrow(dtm_Train_TFIDF)
write.csv(as.data.frame(data.matrix(dtm_Train_TFIDF)),"dtm_Train_TFIDF.csv")
#B) (i) create dtm test matrix for test with TFIDF- Not to be run for unigram naive bayes test DTM 
#make dictionary and use it to make test and train set vocabulary the same.
mydictionary<-dtm_Train_TFIDF$dimnames[["Terms"]]
dtm_Test_TFIDF<-DocumentTermMatrix(news_test_cl,control=list(weighting=function(x)weightTfIdf(x,normalize=TRUE),dictionary=mydictionary))
dtm_Test_TFIDF$dimnames$Docs<-1:nrow(dtm_Test_TFIDF)
##OR####
#B) (ii) for unigram naive bayes test dtm, we do not make dictionary equal as we apply laplace smoothing, so we process below commands instead of above
dtm_Test_TFIDF_nb<-DocumentTermMatrix(news_test_cl,control=list(wordLengths=c(3,18),bounds = list(global = c(2,Inf)),weighting=function(x)weightTfIdf(x,normalize=TRUE)))
dtm_Test_TFIDF_nb<-removeSparseTerms(dtm_Test_TFIDF_nb,0.97)
dtm_Test_TFIDF_nb$dimnames$Docs<-1:nrow(dtm_Test_TFIDF_nb)

#2)N-GRMAS
#for ngrams train TFIDF dtm matrix below commands are required
ngramTokenizer <-function(x){
  unlist(lapply(ngrams(words(x), 3), paste, collapse = " "), use.names = FALSE)
}
dtm_Train_TFIDF_sp<-DocumentTermMatrix(news_train_cl,control=list(tokenize=ngramTokenizer,wordLengths=c(3,18),bounds = list(global = c(2,Inf)),weighting=function(x)weightTfIdf(x,normalize=TRUE)))
dtm_Train_TFIDF<-removeSparseTerms(dtm_Train_TFIDF,0.999)
dtm_Train_TFIDF$dimnames$Docs<-1:nrow(dtm_Train_TFIDF)
write.csv(as.data.frame(data.matrix(dtm_Train_TFIDF)),"dtm_Train_TFIDF.csv")

#B) (i) create dtm test matrix for test with TFIDF- Not to be run for ngram naive bayes test DTM 
#make dictionary and use it to make test and train set vocabulary the same.
mydictionary<-dtm_Train_TFIDF$dimnames[["Terms"]]
dtm_Test_TFIDF<-DocumentTermMatrix(news_test_cl,control=list(weighting=function(x)weightTfIdf(x,normalize=TRUE),dictionary=mydictionary))
dtm_Test_TFIDF$dimnames$Docs<-1:nrow(dtm_Test_TFIDF)
dtm_Test_TFIDF<-removeSparseTerms(dtm_Test_TFIDF,0.999)
dtm_Train_TFIDF$dimnames$Docs<-1:nrow(dtm_Test_TFIDF)

#B) (ii) for n-gram naive bayes test dtm, we do not make dictionary equal as we apply laplace smoothing, so we process below commands instead of above
dtm_Test_TFIDF_sp_nb<-DocumentTermMatrix(news_test_cl,control=list(tokenize=ngramTokenizer,wordLengths=c(3,18),bounds = list(global = c(2,Inf)),weighting=function(x)weightTfIdf(x,normalize=TRUE)))
dtm_Test_TFIDF_nb$dimnames$Docs<-1:nrow(dtm_Test_TFIDF_nb)
dtm_Test_TFIDF_nb<-removeSparseTerms(dtm_Test_TFIDF_nb,0.999)
dtm_Train_TFIDF_nb$dimnames$Docs<-1:nrow(dtm_Test_TFIDF_nb)

dim(dtm_Train_TFIDF)
dim(dtm_Test_TFIDF)

#find freq words and association generating word cloud and associations
#visualising word clouds for topics based on religion-Alt.Atheism,talk.religion.misc & soc.religion.christian
wordFreqAlt.Atheism<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[1:480,]))),decreasing = TRUE)
wordCloudAlt.Atheism<-wordcloud(words=names(wordFreqAlt.Atheism),freq=wordFreqAlt.Atheism,max.words=100,rot.per=0.2,random.order = FALSE,colors = brewer.pal(6,"Dark2"))
#ggplot(aes(names(as.data.frame(wordFreqAlt.Atheism)),as.data.frame(wordFreqAlt.Atheism)))+ geom_bar()+theme(axis.text.x=element_text(angle=45,hjust=1))

wordFreqTalk.religion.misc<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[10938:11314,]))),decreasing=TRUE)
wordCloudTalk.religion.misc<-wordcloud(words=names(wordFreqTalk.religion.misc),freq=wordFreqTalk.religion.misc,rot.per=0.2,max.words=100,random.order = FALSE,colors = brewer.pal(6,"Dark2"))

wordFreqSoc.religion.christian<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[8764:9362,]))),decreasing=TRUE)
wordCloudSoc.religion.christian<-wordcloud(words=names(wordFreqSoc.religion.christian),freq=wordFreqSoc.religion.christian,rot.per=0.2,max.words=100,random.order = FALSE,colors = brewer.pal(6,"Dark2"))


#topics:"comp.graphics","comp.os.ms-windows.misc","comp.sys.ibm.pc.hardware","comp.sys.mac.hardware","comp.windows.x",
wordFreqComp.graphics<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[481:1064,]))),decreasing=TRUE)
wordCloudComp.graphics<-wordcloud(words=names(wordFreqComp.graphics),freq=wordFreqComp.graphics,rot.per=0.2,max.words=100,random.order = FALSE,colors = brewer.pal(6,"Dark2"))#nemo instead of 6 try
write.csv(wordFreqComp.graphics,"compgraphics0.65sparsity.csv")

wordFreqComp.os.ms.windows.misc<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[1065:1655,]))),decreasing=TRUE)
wordCloudComp.os.ms.windows.misc<-wordcloud(words=names(wordFreqComp.os.ms.windows.misc),freq=wordFreqComp.os.ms.windows.misc,rot.per=0.2,max.words=100,random.order = FALSE,colors = brewer.pal(6,"Dark2"))

wordFreqComp.sys.ibm.pc.hardware<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[1656:2245,]))),decreasing=TRUE)
wordCloudComp.sys.ibm.pc.hardware<-wordcloud(words=names(wordFreqComp.sys.ibm.pc.hardware),freq=wordFreqComp.sys.ibm.pc.hardware,rot.per=0.2,max.words=100,random.order = FALSE,colors = brewer.pal(6,"Dark2"))

wordFreqComp.sys.mac.hardware<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[2246:2823,]))),decreasing=TRUE)
wordCloudComp.sys.mac.hardware<-wordcloud(words=names(wordFreqComp.sys.mac.hardware),freq=wordFreqComp.sys.mac.hardware,rot.per=0.2,max.words=100,random.order = FALSE,colors = brewer.pal(6,"Dark2"))
write.csv(wordFreqComp.sys.mac.hardware,"Comp.sys.mac.hardware0.65sparsity.csv")


wordFreqComp.windows.x<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[2824:3416,]))),decreasing=TRUE)
wordCloudComp.windows.x<-wordcloud(words=names(wordFreqComp.windows.x),freq=wordFreqComp.windows.x,rot.per=0.2,max.words=100,random.order = FALSE,colors = brewer.pal(6,"Dark2"))

#topic:misc.forsale"
wordFreqMisc.forsale<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[3417:4001,]))),decreasing=TRUE)
wordCloudMisc.forsale<-wordcloud(words=names(wordFreqMisc.forsale),freq=wordFreqMisc.forsale,rot.per=0.2,max.words=100,random.order = FALSE,colors = brewer.pal(6,"Dark2"))

#topic:"rec.autos","rec.motorcycles",
wordFreqRec.autos<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[4002:4595,]))),decreasing=TRUE)
wordCloudRec.autos<-wordcloud(words=names(wordFreqRec.autos),freq=wordFreqRec.autos,rot.per=0.2,max.words=100,random.order = FALSE,colors = brewer.pal(6,"Dark2"))

wordFreqRec.motorcycles<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[4596:5193,]))),decreasing=TRUE)
wordCloudRec.motorcycles<-wordcloud(words=names(wordFreqRec.motorcycles),freq=wordFreqRec.motorcycles,rot.per=0.2,max.words=100,random.order = FALSE,colors = brewer.pal(6,"Dark2"))

#topic:"rec.sport.baseball","rec.sport.hockey"
wordFreqRec.sport.baseball<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[5194:5790,]))),decreasing=TRUE)
wordCloudRec.sport.baseball<-wordcloud(words=names(wordFreqRec.sport.baseball),freq=wordFreqRec.sport.baseball,rot.per=0.2,max.words=100,random.order = FALSE,colors = brewer.pal(6,"Dark2"))

wordFreqRec.sport.hockey<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[5791:6390,]))),decreasing=TRUE)
wordCloudRec.sport.hockey<-wordcloud(words=names(wordFreqRec.sport.hockey),freq=wordFreqRec.sport.hockey,rot.per=0.2,max.words=100,random.order = FALSE,colors = brewer.pal(6,"Dark2"))

#topics:"sci.crypt","sci.electronics","sci.med","sci.space",
wordFreqSci.crypt<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[6391:6985,]))),decreasing=TRUE)
wordCloudSci.crypt<-wordcloud(words=names(wordFreqSci.crypt),freq=wordFreqSci.crypt,rot.per=0.2,max.words=100,random.order = FALSE,colors = brewer.pal(6,"Dark2"))

wordFreqSci.electronics<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[6986:7576,]))),decreasing=TRUE)
wordCloudSci.electronics<-wordcloud(words=names(wordFreqSci.electronics),freq=wordFreqSci.electronics,rot.per=0.2,max.words=100,random.order = FALSE,colors = brewer.pal(6,"Dark2"))

wordFreqSci.med<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[7577:8170,]))),decreasing=TRUE)
wordCloudSci.med<-wordcloud(words=names(wordFreqSci.med),freq=wordFreqSci.med,rot.per=0.2,max.words=100,random.order = FALSE,colors = brewer.pal(6,"Dark2"))

wordFreqSci.space<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[8171:8763,]))),decreasing=TRUE)
wordCloudSci.space<-wordcloud(words=names(wordFreqSci.space),freq=wordFreqSci.space,rot.per=0.2,max.words=100,random.order = FALSE,colors = brewer.pal(6,"Dark2"))

#topics:"talk.politics.guns","talk.politics.mideast","talk.politics.misc",
wordFreqTalk.politics.guns<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[9363:9908,]))),decreasing=TRUE)
wordCloudTalk.politics.guns<-wordcloud(words=names(wordFreqTalk.politics.guns),freq=wordFreqTalk.politics.guns,rot.per=0.2,max.words=100,random.order = FALSE,colors = brewer.pal(6,"Dark2"))

wordFreqTalk.politics.mideast<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[9909:10472,]))),decreasing=TRUE)
wordCloudTalk.politics.mideast<-wordcloud(words=names(wordFreqTalk.politics.mideast),freq=wordFreqTalk.politics.mideast,rot.per=0.2,max.words=100,random.order = FALSE,colors = brewer.pal(6,"Dark2"))

wordFreqTalk.politics.misc<-sort(colSums(as.matrix(inspect(dtm_Train_TFIDF[10473:10937,]))),decreasing=TRUE)
wordCloudTalk.politics.misc<-wordcloud(words=names(wordFreqTalk.politics.misc),freq=wordFreqTalk.politics.misc,rot.per=0.2,max.words=100,random.order = FALSE,colors = brewer.pal(6,"Dark2"))


#find freq terms occuring more than 20 times in the training set
freqTermsNG<-findFreqTerms(dtm_Train_TFIDF,lowfreq=50)
plot(dtm_Train_TFIDF,term=freqTermsNG,corThreshold=0.1,weighting=T)

findAssocs(dtm_Train_TFIDF,"god",0.10)
findAssocs(dtm_Train_TFIDF,"christian",0.10)
findAssocs(dtm_Train_TFIDF,"file",0.10)
findAssocs(dtm_Train_TFIDF,"drive",0.10)
findAssocs(dtm_Train_TFIDF,"game",0.10)
findAssocs(dtm_Train_TFIDF,"jesus",0.10)
findAssocs(dtm_Train_TFIDF,"player",0.10)
findAssocs(dtm_Train_TFIDF,"program",0.10)
findAssocs(dtm_Train_TFIDF,"team",0.10)

#plotting the associations
myterm<-c("jesus","christ","god","christian","father","holi","belong","etern","attitud","spirit","heaven","sin","teach","therefor","whether","bibl","faith","name","lord","die","interpret","roman","say","son")
plot(dtm_Train_TFIDF,term=myterm,corThreshold=0.1,weighting=T)

#########################
#Analysis
#########################

####different versions of the dtm for different algorithms
#change to data frame for naive bayes,decision trees, random forest,knn
dtm_Train_TFIDF_DF<-as.data.frame(inspect(dtm_Train_TFIDF),stringsAsFactors = FALSE)
dtm_Train_TFIDF_DF$targetcategory<-Train_category_vec
dtm_Test_TFIDF_DF<-as.data.frame(inspect(dtm_Test_TFIDF),stringsAsFactors = FALSE)
dtm_Test_TFIDF_DF$targetcategory<-Test_category_vec

#split into test and validation set for tuning of paramters of some algorithms
splitrat<-sample.split(dtm_Train_TFIDF_DF$targetcategory,SplitRatio = 0.8)
NGTrain<-subset(dtm_Train_TFIDF_DF,splitrat==TRUE)
NGVal<-subset(dtm_Train_TFIDF_DF,splitrat==FALSE)

###############
#Algorithms
##############
#R Text tools Pkg

#create container
dtm_TrainTFIDF_container<-create_container(as.matrix(dtm_Train_TFIDF),trainSize=1:nrow(dtm_Train_TFIDF),sort(Train_category_vec,decreasing = FALSE),virgin=FALSE)
dtm_TestTFIDF_container<-create_container(as.matrix(dtm_Test_TFIDF),labels=rep(0,length(Test_category_vec)), testSize=1:length(Test_category_vec), virgin=FALSE)

#change the category vectors to numeric
levels(Train_category_vec) <- 1:length(levels(Train_category_vec))
Train_category_vec <- as.numeric(Train_category_vec)
class(Train_category_vec)
levels(Test_category_vec) <- 1:length(levels(Test_category_vec))
Test_category_vec <- as.numeric(Test_category_vec)

#1)SUPPORT VECTOR MACHINES

#tuning the parameters for radial kernel: cost and radial
control_SVM <- trainControl(method="cv", number=3)
# design the parameter tuning grid
grid_SVM_radial <- expand.grid(.C=10^(-1:3),.sigma=seq(0.1,0.3,0.1))
#training the model
NGSVM_RadModel <- train(NGTrain[,-which(names(NGTrain)=="targetcategory")],NGTrain$targetcategory, method="svmRadial", trControl=control_SVM,tuneGrid=grid_SVM_radial)
trainpred.SVM<-predict(NGSVM_RadModel,NGTrain,type="raw")
Valpred.SVM<-predict(NGSVM_RadModel,NGVal,type="raw")

#tuning with r texttools pkg
news_model_SVM_rad <- train_model(dtm_TrainTFIDF_container,3,algorithm=c ("SVM"), method="C-classification",cross=1,kernel="radial", cost=1)
tuned <- cross_validate(dtm_TrainTFIDF_container,gamma = 10^(-6:-1), cost = 10^(-1:3))
#am not getting output for both of above tunings, so tried different combinations of gamma and cost for radial svm

# train a SVM linear/radial model for different cost from 0.1 to 100
news_model_SVM <- train_model(dtm_TrainTFIDF_container, "SVM", kernel="linear", cost=1)
summary(news_model_SVM)

#make predictions
news_results_SVM <- classify_model(dtm_TestTFIDF_container, news_model_SVM)
#pred_test<-predict(svm_model,dtm_test)#to generate putput if parameters tuned with caret pacakge(method 1 of tuning)

#inspect results
Conf_SVM<-table(sort(Test_category_vec,decreasing=FALSE), as.character(news_results_SVM[,"SVM_LABEL"]))
write.csv(Conf_SVM,"miss13sepradialcost01lin.csv")

##################
#2nd model:MAXENT
news_model_Maxent <- train_model(dtm_TrainTFIDF_container, "MAXENT")
summary(news_model_Maxent)

#make predictions
news_results_MAXENT <- classify_model(dtm_TestTFIDF_container, news_model_Maxent)
head(news_results_MAXENT)

#inspect results
Conf_MAXENT<-table(as.character(as.numeric(Test_category_vec)), as.character(news_results_MAXENT[,"MAXENTROPY_LABEL"]))
write.csv(Conf_MAXENT,"miss11sepMAXENT.csv")

###########
#3rd :neural network
news_model_NNET <- train_model(dtm_TrainTFIDF_container,size = 5,rang=0.23,maxitnnet = 200, "NNET")
summary(news_model_NNET)

#make predictions
news_results_NNET <- classify_model(dtm_TestTFIDF_container, news_model_NNET)
head(news_results_MAXENT)

#inspect results
Conf_NNET<-table(as.character(Test_category_vec), as.character(news_results_NNET[,1]))
write.csv(Conf_NNET,"miss15sepNNET5rang0.23NODECAY.csv")


##############DECISION TREES############################ 

library(rpart)
library(rpart.plot)
NGTrain$targetcategory<-factor(NGTrain$targetcategory)
NGVal$targetcategory<-factor(NGVal$targetcategory)

#tuning the parameter cp 
control <- trainControl(method="cv", number=3)
# design the parameter tuning grid
grid <- expand.grid(.cp=seq(0.001,0.5,.01))
# train the model
NGTreeModel <- train(NGTrain[,-which(names(NGTrain)=="targetcategory")],NGTrain$targetcategory, method="rpart", trControl=control,tuneGrid=grid)
NGTreeModel[["bestTune"]] #we get 0.001 as tuned cp

trainpred.rpart<-predict(NGTreeModel,NGTrain,type="raw")
Valpred.rpart<-predict(NGTreeModel,NGVal,type="raw")

## Also tried tuning the minsplit, minbucket and cp parameters manually with different combinations of same
control<-rpart.control(cp=0.003,minbucket=60,minsplit=10)

##Training the model
NGmodelCART<-rpart(targetcategory~.,data=NGTrain,control=control)

##Prints the rpart object
rpart.plot(NGmodelCART)

## Prediction on Test and Train data set using Trained model
## Factors - Predicted
trainpred.rpart.man<-predict(NGmodelCART,newdata=NGTrain,type="class")
Valpred.rpart.man<-predict(NGmodelCART,newdata=NGVal,type="class")


##Misclassification Matrix
MisClassTrainCART<-table(Predict=trainpred.rpart.man,Actual=NGTrain$targetcategory)   ## Train Data Prediction
MisClassTrainCART
write.csv(MisClassTrainCART,"MisClassTrainCART0.003_60_10.csv")


MisClassValCART<-table(Predict=Valpred.rpart.man,Actual=NGVal$targetcategory)  ## Test Data Prediction
MisClassValCART
write.csv(MisClassValCART,"MisClassValCART_0.003_60_10.csv")


##AccuracY
accuracyCARTTtrain<-sum(diag(MisClassTrainCART))/sum(MisClassTrainCART)
accuracyCARTTtrain
accuracyCARTTVal<-sum(diag(MisClassValCART))/sum(MisClassValCART)
accuracyCARTTVal

#testing cart model on actual test data

Actualtestpred.rpart<-predict(NGmodelCART,newdata=dtm_Test_TFIDF_DF,type="class")

MisClassActualTestCART<-table(Predict=Actualtestpred.rpart,Actual=Test_category_vec)  ## Test Data Prediction
write.csv(MisClassActualTestCART,"MisClassActualTestCART_0.003_60_10.csv")
accuracyCARTTtest<-sum(diag(MisClassActualTestCART))/sum(MisClassActualTestCART)
accuracyCARTTtest

######################## Random Forest Ensembles   ################################
library(randomForest)
##Tuning with caret package
#control <- trainControl(method="cv", number=3)
# design the parameter tuning grid
#Forest_grid <- expand.grid(mtry=seq(100,300,50))
##tuning with random forest package
#tuneRF(NGTrain[,-which(names(NGTrain)=="targetcategory")],NGTrain$targetcategory,mtryStart = 100,ntreeTry = 500,stepFactor = 50,doBest = TRUE,improve = 0.05)
#above options are not working, so manually tuned the paraeters for different combinations

# train the model
NGForestModel <- randomForest(NGTrain[,-which(names(NGTrain)=="targetcategory")],NGTrain$targetcategory,nodesize = 15,ntree=500 )

##Predict on Test Data and Train Data
Trainpred.forest<-predict(NGForestModel,newdata=NGTrain,type="response")
Valpred.forest<-predict(NGForestModel,newdata=NGVal,type="response")

##Misclassification Matrix
Conf_RF_Train<-table("Predict"=Trainpred.forest,"Actual"=NGTrain$targetcategory)
Conf_RF_Val<-table("Predict"=Valpred.forest,"Actual"=NGVal$targetcategory)  

write.csv(Conf_RF_Train,"missclassForestTrainnode15tree500.csv")
write.csv(Conf_RF_Val,"missclassForestValnode15tree500.csv")

##Accuracy 
accuracyRFtrain<-sum(diag(Conf_RF_Train))/sum(Conf_RF_Train)
accuracyRFtrain
accuracyRFVal<-sum(diag(Conf_RF_Val))/sum(Conf_RF_Val)
accuracyRFVal


#TESTING ON ACTUAL DATA###
TestPred.RandomForestEnsemble<-predict(NGForestModel,dtm_Test_TFIDF)
RF_MisClassTest<-table("Predict"=TestPred.RandomForestEnsemble,"Actual"=Test_category_vec) 
write.csv(RF_MisClassTest,"RFClassTestsept17node15tree500.csv")
sum(diag(RF_MisClassTest))/sum(RF_MisClassTest)

#############

# Naivebayes
library(e1071)
#changing to dataframe for both test and training sets and apending category vector to training set
dtm_Train_TFIDF_DF<-as.data.frame(inspect(dtm_Train_TFIDF),stringsAsFactors=FALSE)
dtm_Train_TFIDF_DF$targetcategory<-Train_category_vec
dtm_Test_TFIDF_DF<-as.data.frame(inspect(dtm_Test_TFIDF_nb),stringsAsFactors=FALSE)

NGmodelNB<-naiveBayes(targetcategory~.,data=dtm_Train_TFIDF_DF,laplace=1)
summary(NGmodelNB)
TrainPrediction_bayesNG <- predict(NGmodelNB,newdata=dtm_Train_TFIDF_DF,type="class")
head(TrainPrediction_bayesNG)

#confusion matrix
TrainResultNB_NG <- table(Actual=dtm_Train_TFIDF_DF$targetcategory,Prediction=TrainPrediction_bayesNG)
write.csv(TrainResultNB_NG,"NBTrain20sep_0.97.csv")
TrainResultNB_NG

accuracyNB_NGTrain <- sum(diag(TrainResultNB_NG)) / sum(TrainResultNB_NG)
accuracyNB_NGTrain

#######test on the actual test set
ActualTestPrediction_bayesNG <- predict(NGmodelNB,newdata=dtm_Test_TFIDF_DF,type="class")

#confusion matrix
ActualTestNB_NGConfMat <- table(Actual=Test_category_vec,Prediction=ActualTestPrediction_bayesNG)
write.csv(ActualTestNB_NGConfMat,"NBTest0.97sparse.csv")
accuracyNB_NGActualTest <- sum(diag(ActualTestNB_NGConfMat)) / sum(ActualTestNB_NGConfMat)
accuracyNB_NGActualTest

##########
#Knearest Neighbours: could not run due to long computing time and had to stop

str(dtm_Train_TFIDF_DF)

split_Vec<-sample.split(dtm_Train_TFIDF_DF$targetcategory,SplitRatio = 0.8)
NGTrain<-subset(dtm_Train_TFIDF_DF,split_Vec==TRUE)
NGVal<-subset(dtm_Train_TFIDF_DF,split_Vec==FALSE)
str(NGTrain)
str(NGVal)
NG_KNN_PredTrain<-knn(as.matrix(NGTrain[,-which(colnames(NGTrain)=="targetcategory")]),as.matrix(NGTrain[,-which(colnames(NGTrain)=="targetcategory")]),cl=NGTrain$targetcategory)
NG_KNN_PredVal<-knn(NGTrain[,-which(colnames(NGTrain)=="targetcategory")],NGVal[,-which(colnames(NGVal)=="targetcategory")],cl=NGTrain$targetcategory)

NGKNNconfMatrixVal<-table(Actual=NGVal$targetcategory,Prediction=NG_KNN_PredVal)
NGKNNconfMatrixTrain<-table(Actual=NGTrain$targetcategory,Prediction=NG_KNN_PredTrain)

NGKNNconfMatrix
write.csv(NGKNNconfMatrixTrain,"NGKNNconfMatrixTrain.csv")
write.csv(NGKNNconfMatrixVal,"NGKNNconfMatrixVal.csv")
NG_KNN_accuracyTrain<-sum(diag(NGKNNconfMatrixTrain))/sum(NGKNNconfMatrixTrain)
NG_KNN_accuracyVal<-sum(diag(NGKNNconfMatrixVal))/sum(NGKNNconfMatrixVal)

##Accuracy based on Acceptance criteria
accuracyknn<-(100-mean(c((nrow(NGVal)-sum(diag(NGKNNconfMatrixVal)))/nrow(NGVal)),(nrow(NGTrain)-sum(diag(NGKNNconfMatrixTest)))/nrow(NGTrain)))
accuracyknn

###on acutal test data
NG_KNN_PredActualTest<-knn(as.matrix(NGTrain[,-which(colnames(NGTrain)=="targetcategory")]),as.matrix(dtm_Test_TFIDF_DF[,-which(colnames(tdm_stack_Test)=="targetcategory")]),cl=NGTrain$targetcategory)


##############################CLUSTERING#########################


#hierarchical clustering

library(cluster)

dtm_Train_TFIDF_clust<-removeSparseTerms(dtm_Train_TFIDF_sp,0.94)
dtm_Train_TFIDF_clust$dimnames$Docs<-1:nrow(dtm_Train_TFIDF_clust)
dim(dtm_Train_TFIDF_clust)
#we get 253 terms over 11314 documents

#clustering docs
dist_mat_docs<-dist(as.matrix(dtm_Train_TFIDF_clust),method="euclidean")
NewsFit_hier<-hclust(d=dist_mat_docs,method="ward.D")

#plot dendogram
plot(NewsFit_hier,hang=-1,cex=0.9,main="Ward.D Cluster Dendogram for 20NewsGroup at 0.95 sparsity")
#6 or 9 or 10 clusters as per dendogram

rect.hclust(NewsFit_hier,k=6,border="red")

#divide into clusters
News_cluster_hier<-cutree(NewsFit_hier,10)
tb<-table(dtm_Train_TFIDF_clust$dimnames$Docs,News_cluster_hier)
tb
write.csv(News_cluster_hier,"6hierarchichal doc clustering6_0.97noscaled.csv")
write.csv(tb,"6 hierarchichal doc clustering6_0.97SCALED.csv")

plot(silhouette(News_cluster_hier,dist_mat_docs),main="Silhouette Plot of 10 Cluster Hierarchical Clustering")

#K means clustering
set.seed(1044)
#scree plot for no. of clusters
tot_withins<-numeric(30)
for (i in 1:30){
  tot_withins[i]<-kmeans(dtm_Train_TFIDF_clust,i,nstart=25)$tot.withinss
}
plot(1:30,tot_withins , type="b", xlab="Number of Clusters",ylab="Within cluster sum of squares")
# there is no clear elbow but 9/10 or 11 clusters look feasible.I have done clustering for 10 groups.
NG_Cluster<-kmeans(as.matrix(dtm_Train_TFIDF_clust),10,nstart=25)
NG_Cluster
NG_ClusterPlusNG<-cbind(as.matrix(dtm_Train_TFIDF_clust),NG_Cluster$cluster)
write.csv(NG_ClusterPlusNG,"20kmeansNG_ClusterPlusNG.csv")
table(NG_Cluster$cluster)

### show clusters using the first 2 principal components
plot(prcomp(as.matrix(dtm_Train_TFIDF_clust))$x, col=NG_Cluster$cluster)

library(proxy)
library(fpc)
#plot
clusplot(as.matrix(dtm_Train_TFIDF_clust),NG_Cluster$cluster,color=FALSE,shade=FALSE,labels=2,line=0)


#skmeans
library(skmeans)
NewsFit_skmeans_docs<-skmeans((as.matrix(dtm_Train_TFIDF_clust)),10)
summary(silhouette(NewsFit_skmeans_docs))
clust_sk <- skmeans((as.matrix(dtm_Train_TFIDF_clust)), 20, method='pclust')
summary(silhouette(clust_sk))
plot(silhouette(NewsFit_skmeans_docs))
write.csv(clust_sk$cluster,"skmeasPVclust20.csv")
write.csv(NewsFit_skmeans_docs$cluster,"skmeas9.csv")


##############################
#for TF unigram DTM for NAIVE BAYES 

#1)UNIGRAMS
#A)create dtm train matrix for test and train set with 0.99 sparsity with TFIDF- TO BE RUN FOR CREATING UNIGRAM TRAIN DTM FOR ALL ALGORITHMS 
dtm_Train_TFIDF_sp<-DocumentTermMatrix(news_train_cl,control=list(wordLengths=c(3,18),bounds = list(global = c(2,Inf)),weighting=function(x)weightTf(x)))
dtm_Train_TFIDF<-removeSparseTerms(dtm_Train_TFIDF_sp,0.99)
dtm_Train_TFIDF$dimnames$Docs<-1:nrow(dtm_Train_TFIDF)
write.csv(as.data.frame(data.matrix(dtm_Train_TFIDF)),"dtm_Train_TFIDF.csv")
#B) for unigram naive bayes test dtm, we do not make dictionary equal as we apply laplace smoothing, so we process below commands instead of above
dtm_Test_TFIDF_nb<-DocumentTermMatrix(news_test_cl,control=list(wordLengths=c(3,18),bounds = list(global = c(2,Inf)),weighting=function(x)weightTf(x)))
dtm_Test_TFIDF_nb<-removeSparseTerms(dtm_Test_TFIDF_nb,0.97)
dtm_Test_TFIDF_nb$dimnames$Docs<-1:nrow(dtm_Test_TFIDF_nb)

#2)N-GRMAS
#for ngrams train TF dtm matrix below commands are required
ngramTokenizer <-function(x){
  unlist(lapply(ngrams(words(x), 3), paste, collapse = " "), use.names = FALSE)
}
dtm_Train_TFIDF_sp<-DocumentTermMatrix(news_train_cl,control=list(tokenize=ngramTokenizer,wordLengths=c(3,18),bounds = list(global = c(2,Inf)),weighting=function(x)weightTf(x)))
dtm_Train_TFIDF<-removeSparseTerms(dtm_Train_TFIDF,0.999)
dtm_Train_TFIDF$dimnames$Docs<-1:nrow(dtm_Train_TFIDF)
write.csv(as.data.frame(data.matrix(dtm_Train_TFIDF)),"dtm_Train_TFIDF.csv")

#B)for n-gram naive bayes TF test dtm, we do not make dictionary equal as we apply laplace smoothing, so we process below commands instead of above
dtm_Test_TFIDF_sp_nb<-DocumentTermMatrix(news_test_cl,control=list(tokenize=ngramTokenizer,wordLengths=c(3,18),bounds = list(global = c(2,Inf)),weighting=function(x)weightTf(x)))
dtm_Test_TFIDF_nb$dimnames$Docs<-1:nrow(dtm_Test_TFIDF_nb)
dtm_Test_TFIDF_nb<-removeSparseTerms(dtm_Test_TFIDF_nb,0.999)
dtm_Train_TFIDF_nb$dimnames$Docs<-1:nrow(dtm_Test_TFIDF_nb)
