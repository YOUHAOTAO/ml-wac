
# WAC: ML Web-based Attack Classifier
[![PyPI version](https://badge.fury.io/py/ml-wac.svg)](https://badge.fury.io/py/ml-wac)

A Machine Learning Web-based Attack Classifier for the purpose of detecting and identifying LFI, RFI, SQLI, and XSS attacks based on request paths. This initiative is part of a research project at the University of Amsterdam conducted by [Jord](https://github.com/jbeek00) and [Isaac](https://github.com/izak0s) under the supervision of [Evgeniia](https://github.com/afeena).

## Getting started
### Installation using PyPI

    pip install ml-wac

### Performance
Performance evaluations have been conducted on all models to determine their accuracy and inference time on the 3579-item test set.

| Model Name             | Accuracy | Inference Time (ms) |
|------------------------|----------|------------------------------|
| XGBoost                | 98.80%   | 82.71                        |
| Decision Tree          | 98.21%   | 1.95                         |
| Logistic Regression    | 98.30%   | 5.58                         |
| Support Vector Machine | 99.25%   | 19225.70                     |


### Examples
#### Predict a single path

    from ml_wac.wac import WebAttackClassifier
    
    # Create new instance
    wac = WebAttackClassifier()
    
    # Predict a single path. Optionally, a certainty threshold can be provided
    prediction = wac.predict_single("/test?id=<script>alert(1)</script>", threshold=0.7)
    
    print(prediction)

#### Predict multiple paths

	from ml_wac.wac import WebAttackClassifier
		
    # Create new instance
    wac = WebAttackClassifier()
    
    # Predict a list of paths, returns a list of predicted attack types
    predictions = wac.predict([
	    "/status?message=<script>/*+Bad+stuff+here+*/</script>",
	    "/?download=../include/connection.php",
	    "/?file=../../uploads/evil.php",
	    "/products?category=Gifts'+OR+1=1--"
    ])
	
	print(predictions)

#### Use other trained models
Use [one of the other pre-trained models](https://github.com/izak0s/ml-wac/blob/main/ml_wac/types/model_type.py) for inference. By default the logistic regression model is used.

    from ml_wac.types.model_type import ModelType
	from ml_wac.wac import WebAttackClassifier

	# Load the XG_BOOST model
	wac = WebAttackClassifier(model_type=ModelType.XG_BOOST)

	# Predict a single path. Optionally, a certainty threshold can be provided
	prediction = wac.predict_single("/test?id=<script>alert(1)</script>", threshold=0.7)
	
	print(prediction)
