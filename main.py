import torch

from utils import *
from dataset import *
from augmentations import *

from torchvision.transforms.functional import to_pil_image

lr = params['lr'] # learning rate
strategy = params['strategy'] # strategy to select distributions
value = params['value'] # value for strategy
action = params['action'] # action to perform
dl_type = params['dataloader'] # which dataloader to load
iterations = int(params['iterations']) # number of iterations to perform

dl = dataloader if dl_type == "augmented" else dataloader_plain
value = int(value) if strategy == "topk" else float(value)

print("#################")
print("# loading model #")
print("#################")

model, processor = load_model_and_processor()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print('#####################')
print('# performing action #')
print('#####################')

try:
  if action == "train":
    print("training the model")
    hyps, refs, data = training_loop(
      model=model, 
      processor=processor, 
      dataloader=dl, 
      classnames=classnames, 
      select_distributions=topk if strategy == "topk" else threshold, 
      criterion=value,
      lr=float(lr),
      iters=iterations)

  elif action == "eval":
    print("evaluating the model")
    hyps, refs = eval_loop(
      model=model,
      processor=processor,
      dataloader=dl,
      classnames=classnames,
      iterations=iterations
    )

  elif action == "get_stats":
    print("getting stats")
    mean, std = None, None
    mean, std = get_avg_std_entropy(
      model=model,
      processor=processor,
      dataloader=dl,
      classnames=classnames,
      iterations=iterations
    )
except KeyboardInterrupt:
  print("quitting training early")


if action == "train" or action == "eval":
  if len(refs) != 0:
    acc = accuracy(refs, hyps)
    acc_top5 = top_5_accuracy(refs, hyps)

    print(f"accuracy: {acc}, accuracy_top5 = {acc_top5}")

else:
  if mean is not None:
    print(f"mean = {mean}, std = {std}")