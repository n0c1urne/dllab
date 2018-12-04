# deep learning lab - Exercise 3


## Overview over the files

Run one of the tests for fun:

    python3 test_model1.py

Note that running a test will overwrite the corresponding ```results/model1.json``` file.

Since I changed the given structure a lot, here's a quick overview:

### movies

Contains recordings of the two test runs submitted to the contest (model 2 and model 4), as well as a recording of the training/validation data.

### model*.py

The model implementations:

1. Simple model with 2 conv layers
2. Better model with 3 conv layers, this was submitted to contest
3. Simple history model, based on model 2 with access to last 5 images
4. An LSTM model based on the CNN model 2, submitted to contest

### model_base.py

Base class for all models, contains code for tensorflow training. Original training code was moved here and heavily adapted.

### test_model*.py

Small start files to run a test on the specific model. Uses ```test_utils.py```. Run these to test a model.

### preprocess_data.py

Preprocesses the data once and creates file ```dataset_fast.cache``` (only used once on new data, output already produced). All training runs use this file.

### utils.py

Contains helper functions (one-hot encoding, RGB to gray scale, sampling batches from training/validation data)

### test_utils.py

Contains the code for running a 15 episode test on an agent.



### folder models

Contains the models used for the report.

### folder results

Contains the result json files for the 4 tests from the report and the results from the manual driving during data aquisition.

### folder tensorboard

Contains the tensorboard recordings from the training runs reported in the paper.