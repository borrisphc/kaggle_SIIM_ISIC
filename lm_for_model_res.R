#######################################################
# load training data  
#######################################################
val <- read_csv('/home/rstudio/CNN_model_result/my_train.csv')
M1  <- read_csv('/home/rstudio/CNN_model_result/my_train_EffNB0_stu_freeze_V3.csv') %>% rename(V1 = target)
M2  <- read_csv('/home/rstudio/CNN_model_result/my_train_EffNB0_stu_V2.csv') %>% rename(V2 = target)
M3  <- read_csv("/home/rstudio/CNN_model_result/my_train_EffNB3_stu_freeze_V5.csv") %>% rename(V3 = target)


lm_data <- 
  left_join(M1, M2, by = c("image_name")) %>% 
  left_join(.,M3, by = c('image_name')) %>% 
  left_join(.,val %>% select(image_name, target))
# set.seed(8989)
# # split in train (train_val)
# train_id <- sample(1:nrow(lm_data),nrow(lm_data)*0.99999)
# lm_data_train <- lm_data[train_id,]
# lm_data_val <- lm_data[-train_id,]

# train model
res <- glm(target ~ V1+V2+V3+V1*V2+V1*V3+V2*V3, data = lm_data %>% select(-image_name), family = "binomial")
summary(res)
res2 <- glm(target ~ ., data = lm_data %>% select(-image_name), family = "binomial")


# AUC(y_pred = lm_data_val$V1, y_true = as.vector(lm_data_val$target))
# AUC(y_pred = lm_data_val$V2, y_true = as.vector(lm_data_val$target))
# AUC(y_pred = lm_data_val$V3, y_true = as.vector(lm_data_val$target))
# 
# AUC(y_pred = lm_data_val %>% select(-image_name, -target) %>% apply(.,1,mean), y_true = as.vector(lm_data_val$target))
# pre <- predict(res, newdata =  lm_data_val %>% select(-image_name, -target), type = "response") 
# AUC(y_pred = pre, y_true = as.vector(lm_data_val$target))
# pre <- predict(res2, newdata =  lm_data_val %>% select(-image_name, -target), type = "response") 
# AUC(y_pred = pre, y_true = as.vector(lm_data_val$target))


#######################################################
# load validation data  
#######################################################


val <- read_csv('/home/rstudio/CNN_model_result/my_val.csv')
M1  <- read_csv('/home/rstudio/CNN_model_result/my_val_TTA_EffNB0_stu_freeze_V3.csv') %>% rename(V1 = target)
M2  <- read_csv('/home/rstudio/CNN_model_result/my_val_TTA_EffNB0_stu_V2.csv') %>% rename(V2 = target)
M3  <- read_csv("/home/rstudio/CNN_model_result/my_val_TTA_EffNB3_stu_freeze_V5.csv") %>% rename(V3 = target)


lm_data_val <- 
  left_join(M1, M2, by = c("image_name")) %>% 
  left_join(.,M3, by = c('image_name')) %>% 
  left_join(.,val %>% select(image_name, target))

# performance
AUC(y_pred = lm_data$V1, y_true = as.vector(lm_data$target))
AUC(y_pred = lm_data$V2, y_true = as.vector(lm_data$target))
AUC(y_pred = lm_data$V3, y_true = as.vector(lm_data$target))

AUC(y_pred = lm_data %>% select(-image_name, -target) %>% apply(.,1,mean), y_true = as.vector(lm_data$target))
pre <- predict(res, newdata =  lm_data %>% select(-image_name, -target), type = "response") 
AUC(y_pred = pre, y_true = as.vector(lm_data$target))
pre <- predict(res2, newdata =  lm_data %>% select(-image_name, -target), type = "response") 
AUC(y_pred = pre, y_true = as.vector(lm_data$target))


#######################################################
# load testing data  
#######################################################

val <- read_csv('/home/rstudio/CNN_model_result/my_test.csv')
M1  <- read_csv('/home/rstudio/CNN_model_result/my_test_TTA_EffNB0_stu_freeze_V3.csv') %>% rename(V1 = target)
M2  <- read_csv('/home/rstudio/CNN_model_result/my_test_TTA_EffNB0_stu_V2.csv') %>% rename(V2 = target)
M3  <- read_csv("/home/rstudio/CNN_model_result/my_test_TTA_EffNB3_stu_freeze_V5.csv") %>% rename(V3 = target)

lm_data <- 
  left_join(M1, M2, by = c("image_name")) %>% 
  left_join(.,M3, by = c('image_name')) %>% 
  left_join(.,val %>% select(image_name))

pre <- predict(res, newdata =  lm_data %>% select(-image_name), type = "response") 

sub_res <- lm_data %>% select(image_name) %>% mutate(target = pre)
getwd()
write.csv(sub_res, "lm_res.csv", row.names = F)
res <- lm_data %>% mutate( target = (V1+V2+V3)/3) %>% select(image_name, target)
write.csv(res, "mean.csv", row.names = F)





