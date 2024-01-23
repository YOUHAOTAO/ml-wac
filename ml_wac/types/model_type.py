from enum import Enum


class ModelType(Enum):
    LOGISTIC_REGRESSION = "logistic_regression.model"
    XG_BOOST = "xgboost.model"
    DECISION_TREE = "decision_tree.model"
    SUPPORT_VECTOR_MACHINE = "support_vector_machine.model"
