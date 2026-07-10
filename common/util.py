import math
import torch
import torch.nn.functional as F

DATASET_PATH = "./datasets"

class AverageMeter:
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.val2 = 0.0
        self.avg2 = 0.0
        self.sum2 = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.count += n
        self.val = val
        self.sum += val * n
        self.avg = self.sum / self.count
        self.val2 = val**2
        self.sum2 += val**2 * n
        self.avg2 = self.sum2 / self.count

    def sdev_biased(self):
        return math.sqrt(max(self.avg2 - self.avg**2, 0.0))

def get_meters(num):
    return [AverageMeter() for _ in range(num)]

def calculate_accuracy(prediction, target):
    # Note that prediction.shape == target.shape == [B, ]
    matching = (prediction == target).float()
    return matching.mean().item()

def test_loop(net, dataloader, device):
    accuracy_meter, loss_meter = get_meters(2)
    net.train(False)
    with torch.no_grad():
        for test_batch in dataloader:
            images, labels = test_batch
            images = images.to(device)
            labels = labels.to(device)
            logits = net.forward(images)
            loss = F.cross_entropy(logits, labels)
            loss_meter.update(loss)

            prediction = logits.argmax(dim=-1)
            accuracy_meter.update(calculate_accuracy(prediction, labels))

    return accuracy_meter.avg, loss_meter.avg
