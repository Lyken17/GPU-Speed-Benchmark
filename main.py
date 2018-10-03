import torch
import torch.nn as nn

from torchvision.models import *
from utils import AverageMeter, memReport


nets = [resnet50(), resnet101(), resnet152()]
batches = [1, 2, 4, 8, 16, 32]

device = "cpu"
if torch.cuda.is_available():
    device = "gpu"

net = nets[0]

for batch in batches:
    fake_data = torch.randn(batch, 3, 224, 224)
    net = net.to(device)
    data = fake_data.to(device)

    print("Warm up GPU")
    for i in range(50):
        out = net(data)

    ips = AverageMeter()
    import time
    for i in range(200):
        start = time.time()
        out = net(data)
        duration = time.time() - start
        ips.update(1.0 / duration)
        print("Iterations per second %.2f" % ips.avg)

    print("Batch size: %d Speed %.2f" % (batch, ips.avg))

    





