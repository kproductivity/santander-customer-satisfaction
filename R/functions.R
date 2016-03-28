# Assess performance
assessPerf <- function(myModel){
  # myModel - H2o model
  perf <- h2o.performance(model = model.nb, newdata = new.hex)
  h2o.confusionMatrix(perf)
  plot(perf, type = "roc", col = "blue", typ = "b")
  perf@metrics$AUC
}

# Generate submission file
genSubmission <- function(myModel){
  # myModel - H2o model
  testPath <- "~/Projects/santander-customer-satisfaction/data/test.csv"
  test.hex <- h2o.uploadFile(path = testPath, header = TRUE)
  test.pca <- h2o.predict(model.pca, test.hex)
  
  submit <- as.data.frame(test.hex$ID)
  submit$TARGET <- as.data.frame(h2o.predict(myModel, newdata=test.pca))$predict
  write.csv(submit, "submission.csv", row.names = FALSE, quote = FALSE)
}
