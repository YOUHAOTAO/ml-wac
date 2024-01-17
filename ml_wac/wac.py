import pickle
from typing import Union
import os
import numpy as np

from ml_wac.types.attack_type import AttackType
from ml_wac.types.model_type import ModelType


class WebAttackClassifier:
    def __init__(self, model_type: ModelType = ModelType.LOGISTIC_REGRESSION):
        current = os.path.dirname(os.path.abspath(__file__))

        with open(os.path.join(current, '..', 'models', model_type.value), 'rb') as f:
            self.saved_model = pickle.load(f)

        with open(os.path.join(current, '..', 'vectorizers', "vectorizer.sklearn"), 'rb') as f:
            self.vectorizer = pickle.load(f)

    def predict_single(self, path: str, threshold: float = 0.7) -> AttackType:
        """
        Predict the web attack type of a single path

        :param path: a path of the URI (e.g. /hello?param=<script>alert('Hi')</script>
        :param threshold: value of the threshold, replacing predictions below with type UNKNOWN
        :return: the predicted attack type
        """
        return self.predict([path], threshold)[0]

    def predict(self, paths: Union[list, np.ndarray], threshold: float = 0.7) -> np.ndarray[AttackType]:
        """
        Predict a batch of web attack types

        :param paths: a list of paths containing a URI (e.g. /hello?param=<script>alert('Hi')</script>
        :param threshold: value of the threshold, replacing predictions below with type UNKNOWN
        :return: a numpy array stating the attack type per batch item
        """
        return self.predict_proba(paths, threshold)[1]

    def predict_proba(self, paths: Union[list, np.ndarray], threshold: float = 0.7) -> Union[
        np.ndarray[np.float64], np.ndarray[AttackType]]:
        """
        Predict a batch of web attack types and include the raw probabilities result

                :param paths: a list of paths containing a URI (e.g. /hello?param=<script>alert('Hi')</script>
        :param threshold: value of the threshold, replacing predictions below with type UNKNOWN
        :return: the probabilities per attack type and a numpy array stating the attack type per batch item
        """
        # Vectorize the paths
        paths_vectorized = self.vectorizer.transform(paths)

        # Predict the probabilities of the classification
        predictions = self.saved_model.predict_proba(paths_vectorized)

        # Calculate the max
        predictions_argmax = np.argmax(predictions, axis=1)
        predictions_max = np.max(predictions, axis=1)

        # Apply threshold, replace with UNKNOWN
        predictions_argmax[predictions_max < threshold] = AttackType.UNKNOWN.value

        # Convert back to attack type enum
        return predictions, np.vectorize(AttackType)(predictions_argmax)
