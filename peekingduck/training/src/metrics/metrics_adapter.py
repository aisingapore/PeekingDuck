
from torchmetrics import AUROC, Accuracy, MetricCollection, Precision, Recall
from torchmetrics.classification import MulticlassCalibrationError

class MetricsAdapter:


    def __init__(self, task: str = 'multiclass', num_classes: int = 2) -> None:
        self.metrics = {}
        self.num_classes = num_classes
        self.task = task
    def setup(self):
        pass

    def accuracy(self):
        self.metrics['accuracy'] = Accuracy(task=self.task, num_classes=self.num_classes)

    def precision(self):
        self.metrics['precision'] = Precision(task=self.task, num_classes=self.num_classes, average="macro")

    def recall(self):
        self.metrics['recall'] = Recall(task=self.task, num_classes=self.num_classes, average="macro")
    
    def f1_score(self):
        pass

    def auroc(self):
        self.metrics['auroc'] = AUROC(task=self.task, num_classes=self.num_classes, average="macro")

    def multiclass_calibration_error(self):
        self.metrics['multiclass_calibration_error'] = MulticlassCalibrationError(num_classes=self.num_classes)

    def make_collection(self):
        self.metrics_collection = MetricCollection(list(self.metrics.values()))
        return self.metrics_collection