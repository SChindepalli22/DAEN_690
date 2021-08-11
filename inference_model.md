---
title: "regularized_regression"
author: "Arjun Paudel"
date: "7/26/2021"
output:
  html_document: 
    keep_md: yes
editor_options:
  chunk_output_type: console
---




```r
library(tidyverse)
library(tidytext)
library(textrecipes)
library(widyr)
library(tidymodels)
library(here)
library(janitor)
library(hardhat)
library(themis)
tidymodels_prefer()

# read data
dt <- read_csv(here("cleaned_data_210620_inference.csv"))

#feature engineering
dt <- dt %>% 
  mutate(landing = if_else(str_detect(str_to_lower(phase_of_flight), "landing"),1,0)) %>% 
  mutate(takeoff = if_else(str_detect(str_to_lower(phase_of_flight), "takeoff"),1,0)) %>% 
  mutate(taxi = if_else(str_detect(str_to_lower(phase_of_flight), "taxi"),1,0)) %>% 
  mutate(approach = if_else(str_detect(str_to_lower(phase_of_flight), "approach"),1,0))  


dt <- dt %>% 
  select (landing, takeoff, taxi, approach,type_incdt,cat_rank,two_aircraft,
          narrative,arpt_emp_veh,prvt_citz_veh,law_enforcement,a_c_maint_Taxi,
          faa_af_emp,constrn_persl,tugs,snow_rmvl_veh,student_pilot,
          frng_a_c_or_pilot,rwy_twy_constrn,ctlr_trng,hold_short_instrs_issued,
          hold_short_rdbk,crsd_hold_short_line_only,entered_rwy,luaw_tiph,
          luaw_tiph_dptd_w_o_clrnc,Landing_or_deptd_w_o_clrc_comm, 
          Landing_deptd_twy_or_clsd_rwy_twy
          )
dt %>% janitor::tabyl(cat_rank)
```

```
##  cat_rank     n     percent
##         A   142 0.008089785
##         B   125 0.007121290
##         C  7007 0.399191021
##         D 10216 0.582008773
##         E    34 0.001936991
##         P    29 0.001652139
```

```r
dt <- dt %>% drop_na() %>% 
  filter(!cat_rank %in% c("E", "P")) %>% 
  mutate(cat_rank = case_when(
    cat_rank %in% c("A", "B") ~ "AB",
    cat_rank %in% c ("C", "D") ~ "CD"
  ) )

#train test split

initialsplit <- initial_split(dt, strata = cat_rank)
dt_train <- training(initialsplit)
dt_test <- testing(initialsplit)
folds <- vfold_cv(dt_train, strata = cat_rank)



sparse_bp <- default_recipe_blueprint(composition = "dgCMatrix")

stop_words2 <- tidytext::stop_words %>% 
  filter(word != "no")

# pre-processing
glmnet_recipe <-
  recipe(cat_rank ~ ., data = dt_train) %>%
  step_tokenize(narrative) %>%
  step_stopwords(narrative, custom_stopword_source = stop_words2$word) %>%
  step_ngram(narrative,num_tokens = 2L, min_num_tokens = 2L) %>% 
  step_tokenfilter(narrative, max_times = 10000, min_times = 100, max_tokens = 5000) %>% 
  step_tfidf(narrative) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE ) %>% 
  step_zv(all_predictors()) %>% 
  step_smote(cat_rank)

#model spec
glmnet_spec <-
  logistic_reg(penalty = tune(), mixture = tune()) %>%
  set_mode("classification") %>%
  set_engine("glmnet")

#pipeline
glmnet_workflow <-
  workflow() %>%
  add_recipe(glmnet_recipe, blueprint = sparse_bp) %>%
  # add_recipe(glmnet_recipe) %>%
  add_model(glmnet_spec)

#hyperparameter grid
glmnet_grid <- tidyr::crossing(
  penalty = 10^seq(-3, -2, length.out = 20),
  mixture = c(0.05, 0.2, 0.4, 0.6, 0.8, 1)
)
```


```r
glmnet_tune <-
  tune_grid(glmnet_workflow,
            resamples = folds, grid = glmnet_grid,
            metrics = metric_set(accuracy, recall, precision, roc_auc),
            control = control_grid(save_pred = TRUE)
  )
```

#doParallel::stopImplicitCluster()

```r
autoplot(glmnet_tune)
```

![](inference_model_files/figure-html/unnamed-chunk-3-1.png)<!-- -->

```r
glmnet_tune %>% show_best("accuracy") %>% knitr::kable()
```



|   penalty| mixture|.metric  |.estimator |      mean|  n|   std_err|.config                |
|---------:|-------:|:--------|:----------|---------:|--:|---------:|:----------------------|
| 0.0010000|    0.20|accuracy |binary     | 0.9582266| 10| 0.0015429|Preprocessor1_Model021 |
| 0.0010000|    0.05|accuracy |binary     | 0.9580508| 10| 0.0016229|Preprocessor1_Model001 |
| 0.0011288|    0.05|accuracy |binary     | 0.9575232| 10| 0.0015189|Preprocessor1_Model002 |
| 0.0011288|    0.20|accuracy |binary     | 0.9573473| 10| 0.0013963|Preprocessor1_Model022 |
| 0.0012743|    0.05|accuracy |binary     | 0.9571714| 10| 0.0015359|Preprocessor1_Model003 |

```r
glmnet_tune %>% show_best("roc_auc") %>% knitr::kable()
```



|   penalty| mixture|.metric |.estimator |      mean|  n|   std_err|.config                |
|---------:|-------:|:-------|:----------|---------:|--:|---------:|:----------------------|
| 0.0100000|     1.0|roc_auc |binary     | 0.8926490| 10| 0.0072375|Preprocessor1_Model120 |
| 0.0100000|     0.8|roc_auc |binary     | 0.8919964| 10| 0.0077479|Preprocessor1_Model100 |
| 0.0088587|     1.0|roc_auc |binary     | 0.8918450| 10| 0.0075726|Preprocessor1_Model119 |
| 0.0078476|     1.0|roc_auc |binary     | 0.8914435| 10| 0.0077622|Preprocessor1_Model118 |
| 0.0088587|     0.8|roc_auc |binary     | 0.8911377| 10| 0.0080168|Preprocessor1_Model099 |

```r
best_param <- glmnet_tune %>% select_best("roc_auc")

glmnet_tune %>% collect_predictions(parameters = best_param, summarize = FALSE) %>% 
  group_by(id) %>% 
  roc_curve(cat_rank, .pred_AB) %>% 
  ggplot(aes(1 - specificity, sensitivity, color = id)) +
  geom_abline(lty = 2, color = "gray80", size = 1.5) +
  geom_path( alpha = 0.6, size = 1.2) +
  coord_equal()
```

![](inference_model_files/figure-html/unnamed-chunk-3-2.png)<!-- -->

```r
glmnet_workflow_final <- glmnet_workflow %>%
  finalize_workflow(parameters = best_param)
```


```r
final_fit <- glmnet_workflow_final %>%
  last_fit(split = initialsplit)
```


```r
final_fit %>% collect_metrics()%>% knitr::kable()
```



|.metric  |.estimator | .estimate|.config              |
|:--------|:----------|---------:|:--------------------|
|accuracy |binary     |  0.921129|Preprocessor1_Model1 |
|roc_auc  |binary     |  0.912990|Preprocessor1_Model1 |

```r
final_fit %>%
  collect_predictions() %>% 
  conf_mat(truth = cat_rank, estimate = .pred_class)
```

```
##           Truth
## Prediction   AB   CD
##         AB   60  280
##         CD   19 3432
```

```r
final_fit %>% 
  collect_predictions() %>% 
  roc_curve(cat_rank, .pred_AB) %>% 
  autoplot()
```

![](inference_model_files/figure-html/unnamed-chunk-5-1.png)<!-- -->


```r
fit_train <- glmnet_workflow_final %>% 
  fit(dt_train) 


fit_train %>% pull_workflow_fit() %>%
  tidy(exponentiate = TRUE,conf.int=TRUE) %>% 
  arrange(desc(abs(estimate))) %>% 
  head(20) %>%
  mutate(term = reorder(term, estimate)) %>% 
  ggplot(aes(estimate, term, fill = estimate > 0))+
  geom_col(show.legend = FALSE)+
  labs(x = "Coefficient values",
       y = "Phrases")
```

![](inference_model_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

```r
fit_train %>% pull_workflow_fit() %>%
  tidy(exponentiate = TRUE,conf.int=TRUE) %>% 
  arrange(-estimate) %>% head()%>% knitr::kable()
```



|term                           |  estimate| penalty|
|:------------------------------|---------:|-------:|
|tfidf_narrative_no_conflicts   | 14.374368|    0.01|
|tfidf_narrative_local_lc       |  6.862553|    0.01|
|tfidf_narrative_no_loss        |  6.806017|    0.01|
|tfidf_narrative_final_takeoff  |  6.224974|    0.01|
|tfidf_narrative_read_correct   |  6.167752|    0.01|
|tfidf_narrative_cleared_runway |  5.067652|    0.01|
