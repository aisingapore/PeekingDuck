
from torchmetrics import AUROC, Accuracy, MetricCollection, Precision, Recall

class MetricsAdapter:


    def __init__(self, num_classes: int = 2) -> None:
        self.metrics = []
        self.num_classes = num_classes

    def setup(self):
        pass

    def accuracy(self):
        self.metrics['accuracy'] = Accuracy(num_classes=self.num_classes)

    def precision(self):
        self.metrics['precision'] = Precision(num_classes=self.num_classes, average="macro")

    def recall(self):
        self.metrics['recall'] = Recall(num_classes=self.num_classes, average="macro")
    
    def f1_score(self):
        pass

    def auroc(self):
        self.metrics['auroc'] = AUROC(num_classes=self.num_classes, average="macro")