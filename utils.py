import torch
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image

device = "cuda" if torch.cuda.is_available else "cpu"

def top_5_accuracy(refs, hyps):
  """
  Calculates top 5 accuracy
  Args:
  - refs: list of references under the form of class IDs
  - hyps: list of hypothesis, stored as lists of (at least) 5 elements
  """
  assert( len(refs) == len(hyps) ) # ensure you have equal amount of hyps and refs
  assert( len(hyps[0]) >= 5 ) # ensure you can at least calculate top5 accuracy

  score = 0
  for i in range(len(refs)):
    for h in hyps[i]:
      if refs[i] == h:
        score += 1
  return score / len(refs)

def accuracy(refs, hyps):
  """
  Calculates top 5 accuracy
  Args:
  - refs: list of references under the form of class IDs
  - hyps: list of lists of hypothesis (for the purpose of accuracy only the first element is considered)
  """
  assert(len(refs) == len(hyps)) # ensure hyps and refs have equal size
  assert(type(hyps[0]) == list) # ensure that hyps are stored under the form of lists (otherwise hyps[i][0] gives error)

  score = 0
  for i in range(len(refs)):
    if refs[i] == hyps[i][0]:
      score += 1
  return score / len(refs)

def get_indices(attention_mask):
  """
  gets the indices of the last non-padding token in by analyzing the attention mask.

  example: [
    [1, 1, 1, 0],
    [1, 1, 0, 0]
  ]
  returns [2, 1]
  """
  indices = torch.arange(attention_mask.shape[1]).to(attention_mask.device)
  indices = indices * attention_mask
  return torch.max(indices, dim=-1).values

def get_cls_tokens(tokens, indices):
  """
  gets the tokens at position indices[i] for the i-th element in the batch

  Args:
    - tokens: element of shape (batch, seq_len, emb_dim)
    - indices: tensor of shape (seq_len) containing the indices of the last non-padding token in a sequence
  """
  assert(tokens.device == indices.device)

  batch_indices = torch.arange(tokens.shape[0]).to(tokens.device)
  return tokens[batch_indices, indices]

from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

def embed(inputs, model):
  """
  Embeds the inputs using the CLIP model
  Args:
    - inputs: pre-processed inputs (output of processor)
    - model: CLIP model
  """
  assert(inputs.device == model.device)

  out = model.text_model.embeddings(input_ids=inputs).to(inputs.device)
  return out

def inject_context(embeddings, context):
  """
  Injects context vectors in the embeddings
  Args:
    - embeddings: embeddings of shape (batch, max_seq_len, emb)
    - context: context vectors of shape (context, emb)
  """
  assert(context.dim() == 2)
  assert(embeddings.dim() == 3)
  assert(embeddings.device == context.device)

  # example: context has shape (4, 512)
  embeddings[:, 1:context.shape[0]+1, :] = context
  return embeddings

def encode(embeddings, attention_mask, model):
  """
  Encodes the embeddings using the CLIP model
  Args:
    - embeddings: tensor of shape (batch, seq_len, emb)
    - attention_mask: attention mask corresponding to the original inputs from which the embeddings have been calculated
    - model: CLIP model
  """
  assert(embeddings.device == attention_mask.device == model.device)


  input_shape = embeddings.shape[:2]

  causal_attention_mask = _create_4d_causal_attention_mask(
    input_shape, embeddings.dtype, device=embeddings.device
  )

  attention_mask = _prepare_4d_attention_mask(attention_mask, embeddings.dtype)

  encoder_output = model.text_model.encoder(
    inputs_embeds=embeddings,
    attention_mask=attention_mask,
    causal_attention_mask=causal_attention_mask
  ).last_hidden_state

  embeddings = model.text_model.final_layer_norm(encoder_output)

  # delete things to avoid taking up CUDA memory
  del input_shape
  del causal_attention_mask
  del attention_mask
  del encoder_output

  return embeddings

def text_model_with_context(inputs, context, model):
  """
  This function embeds inputs, injects the context vectors and then encodes the information
  Args:
    - inputs: inputs to CLIP (obtained with processor)
    - context: context vectors
    - model: CLIP model
  """

  hidden_states = embed(inputs['input_ids'], model)
  hidden_states = inject_context(hidden_states, context)
  return encode(hidden_states, inputs['attention_mask'], model)

def text_model(inputs, model):
  return model.text_model(**inputs)

def text_features(text_model_output, attention_mask, model):
  """
  returns the text representation from the output of text_model.
  - get cls tokens
  - use CLIP.text_projection to project cls tokens

  Args:
  - text_model_output: last hidden state of model.text_model (can be obtained by calling decomposed_text_model)
  - attention_mask: attention mask which tells what tokens are non-padding
  - model: CLIP model
  """
  cls = get_cls_tokens(text_model_output, get_indices(attention_mask))
  return model.text_projection(cls)

def get_output(text_repr, image_repr, logit_scale):
  """
  Takes the textual representation and the image representation and return the clip's output.
  - normalize representations
  - matmul
  - scale

  Args:
    text_repr: textual representation (can be obtained with CILP.get_text_features)
    image_repr: visual representation (can be obtained with CLIP.get_image_features)
    logit scale: scales the result of the multiplication (can be obtained with CLIP.logit_scale.exp())
  """
  text_repr = text_repr / text_repr.norm(dim=-1, keepdim=True)
  image_repr = image_repr / image_repr.norm(dim=-1, keepdim=True)
  out = torch.matmul(image_repr, text_repr.T)

  del text_repr
  del image_repr

  return out.T * logit_scale

from torch.optim import Adam

def entropy(distribution):
  """
  Calculates the entropy of a distribution
  Args:
  - distribution, tensor of shape (n)
  """
  # distribution is a tensor of shape (num_classes)
  distribution = distribution + 1e-10
  return - torch.sum(distribution * torch.log(distribution))

def update_training_data(training_data, indices):
  indices = indices.tolist()
  for i in indices:
    if i >= 1 and  i <= 5:
      training_data["augmix"] += 1
  
    if i >= 6 and i <= 10:
      training_data["gaussian"] += 1

    if i >= 11 and i <= 15:
      training_data["patch"] += 1

    if i >= 16 and i <= 18:
      training_data["spectral"] += 1

    if i >= 19 and i <= 21:
      training_data["canny"] += 1
    
  return training_data

def training_loop(model, processor, dataloader, classnames, select_distributions, criterion, lr, iters=None):
  """
  Trains the context vectors to minimize entropy in marginal distribution
  Args:
  - model: CLIP model
  - processor: proccesses data
  - classnames: different classes for zero shot classification
  - select_distributions: function that selects distributions
  - criterion: criterion used by select_distributions
  - lr: learning rate
  - iters (optional): iterations before training stops
  """
  model.train()
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # initialize context vectors and optimizer
  text = [f"a photo of a {c}" for c in classnames]
  inputs = processor(text=text, padding=True, return_tensors="pt"); inputs = inputs.to(device)

  embeddings = embed(inputs['input_ids'], model)
  context_vectors = embeddings[0, 1:5, :].clone().detach()
  context_vectors = context_vectors.requires_grad_(True)

  optimizer = Adam([context_vectors], lr=lr)

  hyps, refs = [], []

  training_data = {}
  training_data["augmix"] = 0
  training_data["gaussian"] = 0
  training_data["patch"] = 0
  training_data["canny"] = 0
  training_data["spectral"] = 0

  for batch, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
    # calculate CLIP's output
    image_inputs = processor(images=images, return_tensors="pt"); image_inputs = image_inputs.to(device)
    text_inputs = processor(text=text, padding=True, return_tensors="pt"); text_inputs = inputs.to(device)

    img_repr = model.get_image_features(**image_inputs)

    text_model = text_model_with_context(text_inputs, context_vectors, model)
    text_repr = text_features(text_model, text_inputs['attention_mask'], model)

    output = get_output(img_repr, text_repr, model.logit_scale.exp())

    # filter distributions, calculate marginal distribution, optimize entropy
    distributions = torch.softmax(output, dim=-1)
    distributions, indices = select_distributions(distributions, criterion)
    update_training_data(training_data, indices)

    marginal = torch.mean(distributions, dim=0)
    loss = entropy(marginal)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    # re-evaluate with optimized context vectors
    text_model = text_model_with_context(text_inputs, context_vectors, model)
    text_repr = text_features(text_model, text_inputs['attention_mask'], model)

    img_repr = img_repr[0].unsqueeze(0)

    output = get_output(img_repr, text_repr, model.logit_scale.exp())
    indices = torch.argsort(output, dim=-1, descending=True).squeeze()
    ref = torch.argmax(output, dim=-1)

    # register results
    refs.append(labels[0])
    hyps.append(indices[:5].cpu().tolist())

    if iters != None and batch == iters:
      return hyps, refs, training_data

  return hyps, refs, training_data

def eval_loop(model, processor, dataloader, classnames, iterations=None):
  """
  Evaluates clean (not augmented) images, do not use it with augmented dataloader.
  Args:
    - model: CLIP model
    - processor: CLIP processor
    - dataloader: dataloader
    - classnames: classes for zero-shot classification
    - iterations (optional): specify how many samples are seen for evaluation
  """

  dataloader_size = len(dataloader)

  if iterations is not None:
    assert iterations < dataloader_size, "iterations specified overshoot the length of the dataset"

  device = model.device 
  model.eval()

  with torch.no_grad():

    text = [f"a photo of a {c}" for c in classnames]
    refs, hyps = [], []

    for batch, (images, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):

      if iterations is not None and batch == iterations:
        if len(refs) != 0:
          return refs, hyps
        else:
          return None, None

      inputs = processor(text = text, images = images, return_tensors="pt", padding=True)
      inputs = inputs.to(device)

      output = model(**inputs).logits_per_image  
      distributions = torch.softmax(output, dim=-1)
      indices = torch.argsort(distributions, dim=-1, descending=True).squeeze()

      # register results
      for l in labels:
        refs.append(l)

      for i in range(indices.shape[0]):
        hyps.append(indices[i, :5].cpu().tolist())

      break

  return hyps, refs

def topk(distributions, k):
  # distributions is a tensor of shape (n, m)
  # n is the number of image augmentations, m is the number of classes
  n = distributions.shape[0]
  entropies = torch.zeros(distributions.shape[0])
  for i in range(n):
    entropies[i] = entropy(distributions[i])

  indices = torch.argsort(entropies, descending=False)
  indices = indices[:k]

  return distributions[indices, :], indices

def threshold(data, threshold):
  entropies = torch.tensor([entropy(distribution) for distribution in data])
  mask = entropies <= threshold

  if torch.count_nonzero(mask) == 0:
    mask[0] = True
  
  return data[mask], mask.nonzero(as_tuple = True)[0]

def get_avg_std_entropy(model, processor, dataloader, classnames, iterations=None):
  device = model.device
  model.train()
  with torch.no_grad():
    text = [f"a photo of a {c}" for c in classnames]
    entropies = []
    total = iterations if iterations is not None else len(dataloader)
    for i, (images, _) in tqdm(enumerate(dataloader), total=total):
      if i == iterations:
        entropies = torch.tensor(entropies)
        return torch.mean(entropies), torch.std(entropies)

      inputs = processor(text=text, images=images, return_tensors="pt", padding=True); inputs = inputs.to(device)

      outputs = model(**inputs).logits_per_image
      distribution = torch.softmax(outputs, dim=-1)

      for i in range(outputs.shape[0]):
        entropies.append(entropy(distribution[i]))

    entropies = torch.tensor(entropies)
    return torch.mean(entropies), torch.std(entropies)
