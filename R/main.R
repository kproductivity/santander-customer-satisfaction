library(h2o)

source("~/Projects/santander-customer-satisfaction/R/functions.R")

# Init H2o cluster
localH2O <- h2o.init(ip = 'localhost', port = 54321, max_mem_size = '1g')

# Load data
trainPath <- "~/Projects/santander-customer-satisfaction/data/train.csv"
train.hex <- h2o.uploadFile(path = trainPath, header = TRUE)

dim(train.hex)
colnames(train.hex)

# Remove constant columns
mySD <- function(x){
  sd(x, na.rm = FALSE)
}
sdCol <- apply(train.hex, 2, mySD)
constantCol <- as.data.frame(h2o.which(sdCol==0))
train.hex <- train.hex[, -constantCol[,1]] 

dim(train.hex)

################
# Naive Bayes  #
################

train.hex$TARGET <- as.factor(train.hex$TARGET)

## To filter out correlation between the features
model.pca <- h2o.prcomp(training_frame = train.hex[,-337], k = 336, transform = "STANDARDIZE")
new.hex <- h2o.predict(model.pca, train.hex)
new.hex <- h2o.cbind(new.hex, train.hex$TARGET)

# All PCs model
model.nb <- h2o.naiveBayes(x = 1:50, y = 51, training_frame = new.hex, laplace = 3)
perf <- h2o.performance(model = model.nb, newdata = new.hex)
h2o.confusionMatrix(perf)
h2o.confusionMatrix(perf)$Error[3]

# Grid
grid <- h2o.grid("naivebayes", x = c(1:50), y = 51,
                 training_frame = new.hex,
                 hyper_params = list(laplace = c(2,3)))
summary(grid)

# Feature selection
# Assess variable importance using rf
model.rf <- h2o.randomForest(x = 1:336, y = 337, training_frame = new.hex)
impvars <- h2o.varimp(model.rf)

model.nb <- h2o.naiveBayes(x = impvars[1:5, 1], y = 337, training_frame = new.hex, laplace = 3)

#################
# Random forest #
#################

genSubmission(model.rf)
