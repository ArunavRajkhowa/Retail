#---------IMPORTING LIBRARIES-------------------clcle
library(tidymodels)
library(visdat)
library(tidyr)
library(car)
library(pROC)
library(ggplot2)
library(vip)
library(rpart.plot)
library(DALEXtra)
#-----------SETTING UP DIRECTORY and reading files------------------------
setwd("D:\\IITK Data Analytics\\R\\Retail\\")
df_train=read.csv("store_train.csv")
df_test=read.csv("store_test.csv")
vis_dat(df_train) #no missing values

#combing the files for data prep
df_test$store=NA
df_train$data='Train'
df_test$data='Test'

df=rbind(df_train,df_test)
vis_dat(df)
#-----PART 1 -------------------


#----------DATA PREPARATION----------------------------------------------
df$store=as.factor(df$store) #treating response  column separately

dp_pipe=recipe(store~.,data=df) %>%
  
  update_role(Id,countyname,Areaname,storecode,countytownname,new_role = "drop_vars") %>%
  update_role(state_alpha,store_Type,new_role="to_dummies") %>% 
  
  step_rm(has_role("drop_vars")) %>%
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_other(has_role("to_dummies"),threshold =0.02,other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>%
  step_impute_median(all_numeric(),-all_outcomes())
dp_pipe=prep(dp_pipe)

prep_df=bake(dp_pipe,new_data=NULL)
#test=bake(dp_pipe,new_data=test)

vis_dat(prep_df) #all looks good!


#dropping NA response rows in training set
prep_df=prep_df[!((is.na(prep_df$store)) & prep_df$data=='Train'), ]
#what is the meaning of ! here

#separating train and test dataset---------------------------
train=prep_df %>% filter(data=='Train') %>% select(-data,-store_Type_X__other__)
test=prep_df %>% filter(data=='Test') %>% select(-data,-store,-store_Type_X__other__)
vis_dat(train)
vis_dat(test)
# DATA PREP ENDS

# RF--------------

rf_model = rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

folds = vfold_cv(train, v = 10)

rf_grid = grid_regular(mtry(c(5,25)), trees(c(100,500)),
                       min_n(c(2,10)),levels = 5)


my_res=tune_grid(
  rf_model,
  store~.,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = TRUE)
)

autoplot(my_res)+theme_light()

fold_metrics=collect_metrics(my_res)

my_res %>% show_best()

final_rf_fit=rf_model %>% 
  set_engine("ranger",importance='permutation') %>% 
  finalize_model(select_best(my_res,"roc_auc")) %>% 
  fit(store~.,data=train)

# variable importance 

final_rf_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

# predicitons

train_pred=predict(final_rf_fit,new_data = train,type="prob") %>% select(.pred_1)
test_pred=predict(final_rf_fit,new_data = test,type="prob") %>% select(.pred_1)
colnames(test_pred)='store'
write.csv(test_pred,'Arunav_Rajkhowa_P2_part2.csv',row.names = F)
### finding cutoff for hard classes

train.score=train_pred$.pred_1

real=train$store

# KS plot

rocit = ROCit::rocit(score = train.score, 
                     class = real) 

kplot=ROCit::ksplot(rocit)  #explain the plot

# cutoff on the basis of KS

my_cutoff=kplot$`KS Cutoff`

## test hard classes 

test_hard_class=as.numeric(test_pred>my_cutoff)

## partial dependence plots
model_explainer =explain_tidymodels(
  final_rf_fit,
  data = dplyr::select(train, -store),
  y = as.integer(train$store),
  verbose = FALSE
)

pdp = model_profile(
  model_explainer,
  variables = "store",
  N = 2000
  #groups='children'
)

plot(pdp)