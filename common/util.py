import math
import torch
import torch.nn.functional as F

DATASET_PATH = "./datasets"

class AverageMeter:
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.sum2 = 0.0
        self.count = 0

    def update(self, val):
        self.count += 1
        self.val = val
        self.sum += val
        self.sum2 += val**2

    def update_values(self, val, n):
        self.count += n
        self.val = val
        self.sum += val*n

    def update_sum(self, sum_delta, n=1):
        self.count += n
        self.val = 0.0
        self.sum += sum_delta

    def avg(self):
        return self.sum / self.count

    def avg2(self):
        return self.sum2 / self.count

    def sdev_biased(self):
        return math.sqrt(max(self.avg2() - self.avg()**2, 0.0))

def get_meters(num):
    return [AverageMeter() for _ in range(num)]

def calculate_accuracy(prediction, target):
    # Note that prediction.shape == target.shape == [B, ]
    matching = (prediction == target).float()
    return matching.sum().item(), len(target)

def test_loop(net, dataloader, device):
    accuracy_meter, loss_meter = get_meters(2)
    net.train(False)
    with torch.no_grad():
        for test_batch in dataloader:
            images, labels = test_batch
            images = images.to(device)
            labels = labels.to(device)
            logits = net.forward(images)

            prediction = logits.argmax(dim=-1)
            matches, all = calculate_accuracy(prediction, labels)
            accuracy_meter.update_sum(sum_delta=matches, n=all)

            loss = F.cross_entropy(logits, labels)
            loss_meter.update_values(val=loss, n=all)

    return accuracy_meter.avg(), loss_meter.avg()
