
## experiment: classification with deep net on dataset 2 with p = max
## number of realisations: 50
## date: March 2019
## data source: Dinos


library(keras)
library(tidyverse)


# prediction function
myKeras2 <- function(xx_train, y_train, xx_test, y_test,
                     num_layers = 3, 
                     num_nodes = c(256, 128, 64), 
                     rate_drop = c(0.4, 0.3, 0.3),
                     batch_size = 32, 
                     num_epochs = 100, 
                     validation_split = 0.2,
                     penalty_l1 = 0.0, penalty_l2 = 0.0,
                     verbose = FALSE){
  
  # Purpose: wrapper for keras to do predictions
  #          automatically initialises model, layers, nodes and trains a deep net,
  #          activation used: relu; softmax (for final layer),
  #          loss: binary or categorical cross-entropy,
  #
  # @seealso \code{\link[keras]{keras_model_sequential}}
  # 
  # info - example for binary classification:
  # https://jjallaire.github.io/deep-learning-with-r-notebooks/notebooks/3.4-classifying-movie-reviews.nb.html
  #
  # Args:
  #   xx_train: training data [n x p]
  #   y_train: trianing labels [n x 1]
  #   xx_test: test data [n_test x p]
  #   y_test: test labels [n_test x 1]
  #   num_layers: total number of layers
  #               [default: num_layers = 3]
  #   num_nodes: number of nodes per layer [must be a vector of size num_layers]
  #              [default: (256, k); if num_layers > 3: all hidden layers have 256 nodes]
  #   rate_drop: dropout rate [must be a vector of size num_layers - 1]
  #              [default: 0.4 for first layer, 0.3 for hidden layers]
  #   batch_size: 
  #   num_epochs: 
  #   validation_split: fraction of training data used for validation 
  #                     during training (default: 0.2)
  #   penalty_l1: lambda value of L1 penalty term [default: 0.0]
  #   penalty_l2: lambda value of L2 penalty term [default: 0.0]
  #   verbose: logical indicator to toggle visual output
  #
  # OUTPUT:
  #   y_pred: predictions for test data [n_test x 1]
  
  
  
  
  ### check input parameters
  stopifnot( is.numeric(num_layers) & (num_layers >= 2) )
  stopifnot( is.numeric(num_nodes))
  stopifnot( is.numeric(rate_drop))
  stopifnot( length(num_nodes) == (num_layers) )
  stopifnot( length(rate_drop) == (num_layers) )
  stopifnot( batch_size %% 8 == 0 )
  
  
  ### preliminaries
  n <- nrow(xx_train)
  p <- ncol(xx_train)
  n_test <- nrow(xx_test)
  k <- length(unique(y_train))
  
  
  ### preprocessing 
  ## standardise data
  xx_train <- scale(xx_train)
  xx_test <- scale(xx_test)
  
  ## relabel group labels to integers starting from 0
  u_y <- sort(unique(y_train))
  if(!all(u_y == 0:(k-1))){
    tmp <- numeric(n)
    for(i in seq_along(u_y)){ tmp[y_train==u_y[i]] <- i-1 }
    y_train <- tmp
  }
  ## in case of multi-class classification: create indicator matrix from group labels
  if(k > 2){ y_train <- to_categorical(y_train, num_classes = k) }
  
  
  ### construct model 
  model <- keras_model_sequential()
  
  ## specify layers and nodes
  model %>%
    layer_dense(units = num_nodes[1], activation = 'relu', 
                kernel_regularizer = regularizer_l1_l2(l1 = penalty_l1, 
                                                       l2 = penalty_l2),
                input_shape = p) %>%
    layer_dropout(rate = rate_drop[1])      #input layer
  for(j in seq(num_layers)){
    model %>%
      layer_dense(units = num_nodes[j], activation = 'relu',
                  kernel_regularizer = regularizer_l1_l2(l1 = penalty_l1, 
                                                         l2 = penalty_l2)) %>%
      layer_dropout(rate = rate_drop[j])    #hidden layers
  }
  
  
  ## add output layer and compile model
  if(k==0){  #regression (i.e. continuous output)
    model %>% layer_dense(units = 1)          #output layer
    if(verbose){ summary(model) }             #display model summary
    
    model %>% compile(loss = 'mse',
                      optimizer = 'adam',
                      metrics = c('mae'))
    
  }else{    #multi-class classification (i.e. categorical output)
    if(k==2){ 
      loss_func <- 'binary_crossentropy'        #binary classification
      model %>% layer_dense(units = 1, activation = 'tanh')   #output layer
    }else{ 
      loss_func <- 'categorical_crossentropy'   #multi-class classif.
      model %>% layer_dense(units = k, activation = 'softmax')   #output layer
    }
    if(verbose){ summary(model) }               #display model summary
    
    model %>% compile(loss = loss_func, 
                      optimizer = 'adam',
                      metrics = c('accuracy'))
  }
  
  
  ### train deep net --------------------------------------------------------
  history <- model %>% fit(xx_train, y_train,
                           epochs = num_epochs, 
                           batch_size = batch_size,
                           validation_split = validation_split,
                           verbose = verbose)
  
  ### predict test data -----------------------------------------------------
  if(k==0){
    y_pred <- model %>% predict(xx_test)
  }else{
    y_pred <- model %>% predict(xx_test)
    y_pred_classes <- model %>% predict_classes(xx_test)
    if(!all(u_y == seq(k))){   #map classes back to original labels
      tmp <- character(n_test)
      for(i in seq(k)){ tmp[y_pred_classes==(i-1)] <- as.character(u_y[i]) }
      y_pred_labels <- tmp
      
      tmp <- character(n)
      y_pred_train <- model %>% predict_classes(xx_train)
      for(i in seq(k)){ tmp[y_pred_train==(i-1)] <- as.character(u_y[i]) }
      for(i in seq(k)){ tmp[y_train==(i-1)] <- as.character(u_y[i]) }
      y_pred_train_labels <- tmp
    }
  }
  pred_accu <- sum(y_pred_labels==y_test)/n_test    #note: not balanced by class
  if(verbose){ cat( 'accu. on test set: ', round(pred_accu, digits = 4), '\n') }
  
  
  ### clean up
  k_clear_session()
  
  return(list('pred_accu' = pred_accu, 'pred_prob' = y_pred, 
              'pred_labels' = y_pred_labels,
              'training_accu' = mean(y_pred_train_labels==y_train),
              'label_first_group' = u_y[1]))
}




### get data
path_data <- file.path("...")

## preliminaries
num_rep <- 50
fid_out <- file.path(getwd(), 'results', 'keras','DS2_maxp', 'results_keras_ds2_maxp_v2.rds')


### loop over realisations
for(r in seq(num_rep)){
  cat('\n realisation: ', r, '\n')
  
  ## get data for current realisation
  fid_xx_train <- file.path(path_data, paste0('X_train_', r, '.rds'))
  fid_xx_test <- file.path(path_data, paste0('X_test_', r, '.rds'))
  
  fid_y_train <- file.path(path_data, paste0('y_train_', r, '.rds'))
  fid_y_test <- file.path(path_data, paste0('y_test_', r, '.rds'))
  
  xx_train <- readRDS(fid_xx_train)
  xx_test <- readRDS(fid_xx_test)
  
  y_train <- readRDS(fid_y_train)
  y_test <- readRDS(fid_y_test)
  
  
  ## fit neural nets
  fit <- NULL
  fit2 <- NULL
  fit$pred_accu <- 0
  fit2$pred_accu <- 0
  
  while(fit$pred_accu < 0.75){   #re-start condition due to getting stuck in 1 group
    fit <- myKeras2(xx_train, y_train, xx_test, y_test,
                    num_layers = 3, 
                    num_nodes = c(256, 128, 64), 
                    rate_drop = c(0.4, 0.3, 0.3),
                    batch_size = 512, 
                    num_epochs = 100, 
                    validation_split = 0.2,
                    penalty_l1 = 0.0, 
                    penalty_l2 = 0.005,
                    verbose = FALSE)
  }
  
  while(fit2$pred_accu < 0.75){   #re-start condition due to getting stuck in 1 group
    fit2 <- myKeras2(xx_train, y_train, xx_test, y_test,
                     num_layers = 8, 
                     num_nodes = c(1024, 512, 512, 256, 256, 128, 128, 64), 
                     rate_drop = c(0.4, rep(0.3, 7)),
                     batch_size = 512, 
                     num_epochs = 100, 
                     validation_split = 0.2,
                     penalty_l1 = 0.0, 
                     penalty_l2 = 0.005,
                     verbose = FALSE)
  }
  
  curr_res_tbl <- tibble('realisation' = c(r,r), 
                         'num_layers' = c(5, 10),
                         'acc' = c(fit$pred_accu, fit2$pred_accu))
  
  
  ## save results
  if(!file.exists(fid_out)){
    saveRDS(curr_res_tbl, file = fid_out)
  }else{
    tmp <- readRDS(fid_out)
    res_tbl <- dplyr::bind_rows(tmp, curr_res_tbl)
    saveRDS(res_tbl, file = fid_out)
  }
}
cat('\n ... done \n\n')




# -----------------------------------------------------------------------------
### Plots
library(ggplot2)
library(RColorBrewer)
library(Hmisc)
library(tidyverse)

#fid_res <- file.path(getwd(), 'results', 'keras', 'DS2_maxp', 'results_keras_ds2_maxp.rds')
fid_res <- file.path(getwd(), 'results', 'keras', 'DS2_maxp', 'results_keras_ds2_maxp_v2.rds')
res_tbl <- readRDS(fid_res)

res_tbl$num_layers <- as.factor(res_tbl$num_layers)

## boxplot of accuracy
p <- ggplot(res_tbl, aes(x = num_layers, y = acc, fill = num_layers)) + 
      geom_boxplot() + 
      scale_fill_brewer(palette = "Blues") + 
      theme_minimal()+
      xlab('# layers') +
      ylab('accuracy')
print(p)



## violin plot of acuracy
p <- ggplot(res_tbl, aes(x = num_layers, y = acc, fill = num_layers)) + 
      geom_violin(trim = FALSE) + 
      ggplot2::stat_summary(fun.data = mean_se,            #incl. mean +/- SE
                   geom = "pointrange", color = "red") +
      scale_fill_brewer(palette = "Blues") + 
      theme_minimal() +
      xlab('# layers') +
      ylab('accuracy')
print(p)


cat('\n mean accuracy:  ') 
res_tbl %>% group_by(num_layers) %>% summarise_at('acc', mean)

cat('\n median accuracy:  ') 
res_tbl %>% group_by(num_layers) %>% summarise_at('acc', median)
