# /// script
# dependencies = [
#   "python-dotenv",
#   "numpy",
#   "wandb",
#   "torch",
#   "torchvision",
# ]
# [tool.uv]
# exclude-newer = "2024-02-20T00:00:00Z"
# ///

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from datetime import datetime
import time
import torch.profiler
import hashlib
import os
import math
import socket
from zoneinfo import ZoneInfo
import wandb

############################ CONFIG ########################

from dotenv import dotenv_values
config = { **dotenv_values(".env"), **os.environ }

LOG_NAME = config.get("LOG_NAME", "MNIST")
TIMEZONE = config.get("TIMEZONE", "UTC")
PAPERTRAIL_HOST = config.get("PAPERTRAIL_HOST")
PAPERTRAIL_PORT = config.get("PAPERTRAIL_PORT")
WANDB_KEY = config.get("WANDB_KEY")
WANDB_PROJECT = config.get("WANDB_PROJECT", "mnist_project")

BINARIZE_IMAGE_TRESHOLD = float(config.get("BINARIZE_IMAGE_TRESHOLD", 0.75))
IMG_WIDTH = int(config.get("IMG_WIDTH", 16))
INPUT_SIZE = IMG_WIDTH * IMG_WIDTH
DATA_SPLIT_SEED = int(config.get("DATA_SPLIT_SEED", 42))
TRAIN_FRACTION = float(config.get("TRAIN_FRACTION", 0.9))
NUMBER_OF_CATEGORIES = int(config.get("NUMBER_OF_CATEGORIES", 10))
ONLY_USE_DATA_SUBSET = config.get("ONLY_USE_DATA_SUBSET", "0").lower() in ("true", "1", "yes")

SEED = int(config.get("SEED", 97798))
if SEED < 0:
    SEED = random.randint(0, 1000_000)
NET_ARCHITECTURE = [int(l) for l in config.get("NET_ARCHITECTURE", "[1300,1300,1300]")[1:-1].split(',')]
BATCH_SIZE = int(config.get("BATCH_SIZE", 256))

EPOCHS = int(config.get("EPOCHS", 50))
EPOCH_STEPS = round(54_000 / BATCH_SIZE) # 54K train /6K val/10K test
TRAINING_STEPS = EPOCHS*EPOCH_STEPS
PRINTOUT_EVERY = int(config.get("PRINTOUT_EVERY", EPOCH_STEPS // 4))
VALIDATE_EVERY = int(config.get("VALIDATE_EVERY", EPOCH_STEPS))

LEARNING_RATE = float(config.get("LEARNING_RATE", 0.01))
DECAY_CONST_GATES = float(config.get("DECAY_CONST_GATES", 0.05))

PASSTHROUGH_REGULARIZATION = float(config.get("PASSTHROUGH_REGULARIZATION", 1.))
CONNECTION_REGULARIZATION = float(config.get("CONNECTION_REGULARIZATION", 5.))
GATE_WEIGHT_REGULARIZATION = float(config.get("GATE_WEIGHT_REGULARIZATION", 1.))
LOSS_CE_STRENGTH = float(config.get("LOSS_CE_STRENGTH", 0.9))

config_printout_keys = ["LOG_NAME", "TIMEZONE", "WANDB_PROJECT",
               "BINARIZE_IMAGE_TRESHOLD", "IMG_WIDTH", "INPUT_SIZE", "DATA_SPLIT_SEED", "TRAIN_FRACTION", "NUMBER_OF_CATEGORIES", "ONLY_USE_DATA_SUBSET",
               "SEED", "NET_ARCHITECTURE", "BATCH_SIZE",
               "EPOCHS", "EPOCH_STEPS", "TRAINING_STEPS", "PRINTOUT_EVERY", "VALIDATE_EVERY",
               "LEARNING_RATE", "PASSTHROUGH_REGULARIZATION", "DECAY_CONST_GATES",
               "CONNECTION_REGULARIZATION", "GATE_WEIGHT_REGULARIZATION", "LOSS_CE_STRENGTH"]
config_printout_dict = {key: globals()[key] for key in config_printout_keys}

# Making sure sensitive configs are not logged
assert "PAPERTRAIL_HOST" not in config_printout_dict.keys()
assert "PAPERTRAIL_PORT" not in config_printout_dict.keys()
assert "WANDB_KEY" not in config_printout_dict.keys()

WANDB_KEY and wandb.login(key=WANDB_KEY)
WANDB_KEY and (wandb_run := wandb.init(project=WANDB_PROJECT, name=f"{LOG_NAME}_{SEED}", config=config_printout_dict))

############################ LOG ########################

def create_papertrail_logger(log_name, timezone, papertrail_host, papertrail_port):
    def papertrail(message):
        timestamp = datetime.now(ZoneInfo(timezone))
        if (papertrail_host is not None) and (papertrail_port is not None):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    priority = 22
                    hostname = ""
                    syslog_message = (
                        f"<{priority}>{timestamp.strftime('%b %d %H:%M:%S')} "
                        f"{hostname} {log_name}: {message}"
                    )
                    sock.sendto(
                        syslog_message.encode("utf-8"),
                        (papertrail_host, int(papertrail_port)),
                    )
            except:
                pass
        print(f'{timestamp.strftime("%H:%M:%S")} {message}', flush=True)
    return papertrail

log = create_papertrail_logger(LOG_NAME, TIMEZONE, PAPERTRAIL_HOST, PAPERTRAIL_PORT)

if WANDB_KEY is None:
    log("-"*80)
    for k in config_printout_dict.keys():
        log(f"{k}={config_printout_dict[k]}")
    log("-"*80)

############################ DEVICE ########################

try:
    device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else 
                      "cpu")
except:
    device = "cpu"
log(f"device={device}")
WANDB_KEY and wandb.log({"device": str(device)})

############################ MODEL ########################

class LearnableGate16Array(nn.Module):
    def __init__(self, number_of_gates, number_of_inputs, name):
        super(LearnableGate16Array, self).__init__()
        self.number_of_gates = number_of_gates
        self.number_of_inputs = number_of_inputs
        self.name = name
        self.w = nn.Parameter(torch.zeros((16, number_of_gates), dtype=torch.float32)) # [16, W]
        self.zeros = torch.empty(0)
        self.ones = torch.empty(0)
        self.binarized = False
        self.frozen = False
        self.c = nn.Parameter(torch.zeros((number_of_inputs, number_of_gates, 2), dtype=torch.float32)) # connectome       
        # Only Gaussian inits supported for now
        nn.init.normal_(self.w, mean=0.0, std=1)
        nn.init.normal_(self.c, mean=0.0, std=1)


    def forward(self, x):
        batch_size = x.shape[0]
        connections = F.softmax(self.c, dim=0) if not self.binarized else self.c
        # [batch_size, number_of_inputs] x [number_of_inputs, number_of_gates*2] -> [batch_size, number_of_gates*2]
        x = torch.matmul(x, connections.view(self.number_of_inputs, self.number_of_gates*2))
        x = x.view(batch_size, self.number_of_gates, 2)

        A = x[:,:,0]
        A = A.transpose(0,1)
        B = x[:,:,1]
        B = B.transpose(0,1)

        if self.zeros.shape != A.shape:
            self.zeros = torch.zeros_like(A)
        if self.ones.shape != A.shape:
            self.ones = torch.ones_like(A)
            
        # Numbered according to https://arxiv.org/pdf/2210.08277 table
        AB = A*B

        g0  = self.zeros
        g1  = AB
        g2  = A - AB
        g3  = A
        g4  = B - AB
        g5  = B
        g7  = A + B - AB
        g6  = g7    - AB            # A + B - 2 * A * B
        g8  = self.ones - g7
        g9  = self.ones - g6
        g10 = self.ones - B
        g11 = self.ones - g4
        g12 = self.ones - A
        g13 = self.ones - g2
        g14 = self.ones - AB
        g15 = self.ones

        weights = F.softmax(self.w, dim=0) if not self.binarized else self.w
        gates = torch.stack([
            g0, g1, g2, g3, g4, g5, g6, g7,
            g8, g9, g10, g11, g12, g13, g14, g15
            ], dim=0)
        assert gates.dim() > 1
        if gates.dim() == 2:
            gates = gates.unsqueeze(dim=1)                    # broadcast [C,N] -> [C,1,N]; C=16, N=batch_size
        x = (gates * weights.unsqueeze(dim=-1)).sum(dim=0) # [C,W,N] .* (broadcast [C,W] -> [C,W,1]) =[sum-over-C]=> [W,N]
        return x.transpose(0,1)
    

class Model(nn.Module):
    def __init__(self, seed, net_architecture, number_of_categories, input_size):
        super(Model, self).__init__()
        self.net_architecture = net_architecture
        self.first_layer_gates = self.net_architecture[0]
        self.last_layer_gates = self.net_architecture[-1]
        self.number_of_categories = number_of_categories
        self.input_size = input_size
        self.seed = seed
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.outputs_per_category = self.last_layer_gates // self.number_of_categories
        assert self.last_layer_gates == self.number_of_categories * self.outputs_per_category

        layers_ = []
        for layer_idx, layer_gates in enumerate(net_architecture):
            if layer_idx==0:
                layers_.append(LearnableGate16Array(number_of_gates=layer_gates,number_of_inputs=input_size, name=layer_idx))
            else:
                layers_.append(LearnableGate16Array(number_of_gates=layer_gates,number_of_inputs=prev_gates, name=layer_idx))
            prev_gates = layer_gates
        self.layers = nn.ModuleList(layers_)

    def forward(self, X):
        for layer_idx in range(0, len(self.layers)):
            X = self.layers[layer_idx](X)

        X = X.view(X.size(0), self.number_of_categories, self.outputs_per_category).sum(dim=-1)
        X = F.softmax(X, dim=-1)
        return X

    def get_passthrough_fraction(self):
        pass_fraction_array = torch.zeros(len(self.layers), dtype=torch.float32, device=device)
        indices = torch.tensor([3, 5, 10, 12], dtype=torch.long)
        for layer_ix, layer in enumerate(self.layers):
            weights_after_softmax = F.softmax(layer.w, dim=0)
            pass_weight = (weights_after_softmax[indices, :]).sum()
            total_weight = weights_after_softmax.sum()
            pass_fraction_array[layer_ix] = pass_weight / total_weight
        return pass_fraction_array
    
    def state_dict(self, *args, **kwargs):
        state_dict = super(Model, self).state_dict(*args, **kwargs)
        state_dict['net_architecture'] = self.net_architecture
        state_dict['seed'] = self.seed
        if hasattr(self, 'dataset_input'):
            state_dict['dataset_input'] = self.dataset_input
        if hasattr(self, 'dataset_output'):
            state_dict['dataset_output'] = self.dataset_output
        connections = [ [], [] ]
        for layer in self.layers:
            ones_at = torch.argmax(layer.c, dim=0)
            connections[0].append(ones_at[:,0])
            connections[1].append(ones_at[:,1])
        state_dict['connections'] = connections
        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        if 'net_architecture' in state_dict:
            self.net_architecture = state_dict.pop('net_architecture')
        if 'seed' in state_dict:
            self.seed = state_dict.pop('seed')
        if 'connections' in state_dict:
            state_dict.pop('connections')
        super(Model, self).load_state_dict(state_dict, strict=strict)
    

############################ DATA ########################





### GENERATORS
def binarize_image_with_histogram(image, verbose=False):
    threshold = torch.quantile(image, BINARIZE_IMAGE_TRESHOLD)
    return (image > threshold).float()

transform = transforms.Compose([
    transforms.Resize((IMG_WIDTH, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)),
    transforms.Lambda(lambda x: binarize_image_with_histogram(x))
])

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)


train_size = int(TRAIN_FRACTION * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(DATA_SPLIT_SEED))

if ONLY_USE_DATA_SUBSET:
    train_dataset = torch.utils.data.Subset(train_dataset, range(1024))
    val_dataset = torch.utils.data.Subset(val_dataset, range(1024))

### MOVE TRAIN DATASET TO GPU ###

train_dataset_samples = len(train_dataset)
train_images = torch.empty((train_dataset_samples, INPUT_SIZE), dtype=torch.float32, device=device)
train_labels = torch.empty((train_dataset_samples, NUMBER_OF_CATEGORIES), dtype=torch.float32, device=device)

train_labels_ = torch.empty((train_dataset_samples), dtype=torch.long, device=device)
for i, (image, label) in enumerate(train_dataset):
    train_images[i] = image
    train_labels_[i] = label
train_labels = torch.nn.functional.one_hot(train_labels_, num_classes=NUMBER_OF_CATEGORIES)
train_labels = train_labels.type(torch.float32)

### MOVE VAL DATASET TO GPU ###

val_dataset_samples = len(val_dataset)

val_images = torch.empty((val_dataset_samples, INPUT_SIZE), dtype=torch.float32, device=device)
val_labels = torch.empty((val_dataset_samples, NUMBER_OF_CATEGORIES), dtype=torch.float32, device=device)

val_labels_ = torch.empty((val_dataset_samples), dtype=torch.long, device=device)
for i, (image, label) in enumerate(val_dataset):
    val_images[i] = image
    val_labels_[i] = label
val_labels = torch.nn.functional.one_hot(val_labels_, num_classes=NUMBER_OF_CATEGORIES)
val_labels = val_labels.type(torch.float32)


### MOVE TEST DATASET TO GPU ###

test_dataset_samples = len(test_dataset)

test_images = torch.empty((test_dataset_samples, INPUT_SIZE), dtype=torch.float32, device=device)
test_labels = torch.empty((test_dataset_samples, NUMBER_OF_CATEGORIES), dtype=torch.float32, device=device)

test_labels_ = torch.empty((test_dataset_samples), dtype=torch.long, device=device)
for i, (image, label) in enumerate(test_dataset):
    test_images[i] = image
    test_labels_[i] = label
test_labels = torch.nn.functional.one_hot(test_labels_, num_classes=NUMBER_OF_CATEGORIES)
test_labels = test_labels.type(torch.float32)


### INSTANTIATE THE MODEL AND MOVE TO GPU ###

model = Model(seed=SEED, net_architecture=NET_ARCHITECTURE, number_of_categories=NUMBER_OF_CATEGORIES, input_size=INPUT_SIZE).to(device)

### VALIDATE ###

def validate(dataset="val", model=model):
    if dataset == "val":
        number_of_samples = val_dataset_samples
        sample_images = val_images
        sample_labels = val_labels
    elif dataset == "test":
        number_of_samples = test_dataset_samples
        sample_images = test_images
        sample_labels = test_labels
    elif dataset == "train":
        number_of_samples = train_dataset_samples
        sample_images = train_images
        sample_labels = train_labels
    else:
        raise IOError(f"Unknown dataset {dataset}")
    val_loss = 0.0
    val_steps = 0
    correct = 0
    for start_idx in range(0, number_of_samples, BATCH_SIZE):
        end_idx = min(start_idx + BATCH_SIZE, number_of_samples)
        val_indices = torch.arange(start_idx, end_idx, device=device)    
        x_val = sample_images[val_indices]
        y_val = sample_labels[val_indices]
        with torch.no_grad():
            val_output = model(x_val)
            val_loss += F.cross_entropy(val_output, y_val, reduction="sum").item()
            correct += (val_output.argmax(dim=1) == y_val.argmax(dim=1)).sum().item()
        val_steps += len(x_val)
    val_loss /= val_steps
    val_accuracy = correct / val_steps
    return val_loss, val_accuracy

def binarize_model(model=model, bin_value=1):
    model_binarized = Model(seed=SEED, net_architecture=NET_ARCHITECTURE, number_of_categories=NUMBER_OF_CATEGORIES, input_size=INPUT_SIZE).to(device)
    model_binarized.load_state_dict(model.state_dict())

    for layer_idx in range(0, len(model_binarized.layers)):
        ones_at = torch.argmax(model_binarized.layers[layer_idx].w.data, dim=0)
        model_binarized.layers[layer_idx].w.data.zero_()
        model_binarized.layers[layer_idx].w.data.scatter_(dim=0, index=ones_at.unsqueeze(0), value=bin_value)

        ones_at = torch.argmax(model_binarized.layers[layer_idx].c.data, dim=0)
        model_binarized.layers[layer_idx].c.data.zero_()
        model_binarized.layers[layer_idx].c.data.scatter_(dim=0, index=ones_at.unsqueeze(0), value=bin_value)

        model_binarized.layers[layer_idx].binarized = True

    return model_binarized

def l1_maxOnly_regularization(weights_after_softmax):
    max_values, _ = torch.max(weights_after_softmax, dim=0, keepdim=True)
    non_max_sum = (1 - max_values).sum()
    largest_possible_sum = torch.prod(torch.tensor(weights_after_softmax.shape[1:])) # when all elements are uniform; cutting out 0 dim since it is maxxed over
    return non_max_sum / largest_possible_sum # uniform distribution gives 1 per layer

def passthrough_regularization(weights_after_softmax):
    indices = torch.tensor([3, 5, 10, 12], dtype=torch.long)
    pass_weight = (weights_after_softmax[indices, :]).sum()
    total_weight = weights_after_softmax.sum()
    return pass_weight / total_weight


### TRAIN ###

val_loss, val_accuracy = validate(dataset="val")
log(f"INIT VAL loss={val_loss:.3f} acc={val_accuracy*100:.2f}%")
WANDB_KEY and wandb.log({"init_val": val_accuracy*100})

### load ###
# model.load_state_dict(torch.load("20250225-140458_binTestAcc7911_seed982779_epochs100_3x300_b256_lr10.pth", map_location=torch.device(device), weights_only=False))
# val_loss, val_accuracy = validate(dataset="val")
# log(f"INIT VAL loss={val_loss:.3f} acc={val_accuracy*100:.2f}%")
if (CONNECTION_REGULARIZATION > 0) and (GATE_WEIGHT_REGULARIZATION  > 0):
    log("REGULARIZATING")
### end load ###

log(f"EPOCH_STEPS={EPOCH_STEPS}, will train for {EPOCHS} EPOCHS")
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0) #!!!
time_start = time.time()


for i in range(TRAINING_STEPS):
    indices = torch.randint(0, train_dataset_samples, (BATCH_SIZE,), device=device)
    x = train_images[indices]
    y = train_labels[indices]
    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
        model_output = model(x)
        loss_ce = F.cross_entropy(model_output, y) * LOSS_CE_STRENGTH

        connection_regularization_loss = 0
        gate_weight_regularization_loss = 0
        passthrough_regularization_loss = 0
        for layer in model.layers:
            passthrough_regularization_loss += passthrough_regularization(F.softmax(layer.w, dim=0))
            connection_regularization_loss += l1_maxOnly_regularization(F.softmax(layer.c, dim=0))
            gate_weight_regularization_loss += l1_maxOnly_regularization(F.softmax(layer.w, dim=0))
        
        passthrough_regularization_loss = passthrough_regularization_loss / len(model.layers)
        connection_regularization_loss = connection_regularization_loss / len(model.layers)
        gate_weight_regularization_loss = gate_weight_regularization_loss / len(model.layers)
        regularization_loss = PASSTHROUGH_REGULARIZATION * passthrough_regularization_loss + CONNECTION_REGULARIZATION * connection_regularization_loss + GATE_WEIGHT_REGULARIZATION * gate_weight_regularization_loss
        regularization_loss = (1 - LOSS_CE_STRENGTH) * regularization_loss
        
        loss = loss_ce + regularization_loss
        loss.backward()
        optimizer.step()

        # TODO: rewrite this as regularization
        for l in model.layers:
            for const_gate_ix in [0,15]:
                l.w.data[const_gate_ix, :] = l.w.data[const_gate_ix, :] * (1 - LEARNING_RATE*DECAY_CONST_GATES)


    if (i + 1) % PRINTOUT_EVERY == 0:
        passthrough_log = ", ".join([f"{value * 100:.1f}%" for value in model.get_passthrough_fraction().tolist()])
        log(f"Iteration {i + 1:10} - Loss {loss:.3f} - RegLoss {(1-loss_ce/loss)*100:.0f}% - Pass {passthrough_log}")
        WANDB_KEY and wandb.log({"training_step": i, "loss": loss, "connection_regularization_loss":connection_regularization_loss, "gate_weight_regularization_loss":gate_weight_regularization_loss, 
            "regularization_loss_fraction":(1-loss_ce/loss)*100, "passthrough_regularization_loss":passthrough_regularization_loss})
        # log(f"loss_ce={F.cross_entropy(model_output, y).detach().item()}")
        # log(f"connection_regularization_loss={connection_regularization_loss}")
        # log(f"gate_weight_regularization_loss={gate_weight_regularization_loss}")
        # log(f"passthrough_regularization_loss={passthrough_regularization_loss}")
    if (i + 1) % VALIDATE_EVERY == 0:
        current_epoch = (i+1) // EPOCH_STEPS

        train_loss, train_acc = validate('train')
        # log(f"EPOCH={current_epoch}/{EPOCHS}     TRN loss={train_loss:.3f} acc={train_acc*100:.2f}%")
        model_binarized = binarize_model()
        _, bin_train_acc = validate(dataset="train", model=model_binarized)
        log(f"EPOCH={current_epoch}/{EPOCHS} BIN TRN acc={bin_train_acc*100:.2f}%, train_acc_diff={train_acc*100-bin_train_acc*100:.2f}%")
        
        
        val_loss, val_acc = validate()
        _, bin_val_acc = validate(model=model_binarized)
        # log(f"EPOCH={current_epoch}/{EPOCHS} VAL loss={val_loss:.3f} acc={val_acc*100:.2f}%")
        log(f"EPOCH={current_epoch}/{EPOCHS} BIN VAL acc={bin_val_acc*100:.2f}%,   val_acc_diff={val_acc*100-bin_val_acc*100:.2f}%")

        WANDB_KEY and wandb.log({"epoch": current_epoch, 
            "train_loss": train_loss, "train_acc": train_acc*100,
            "val_loss": val_loss, "val_acc": val_acc*100,
            "bin_train_acc": bin_train_acc*100, "train_acc_diff": train_acc*100-bin_train_acc*100,
            "bin_val_acc": bin_val_acc*100, "val_acc_diff": val_acc*100-bin_val_acc*100,
            })

log(f"Network architecture: {NET_ARCHITECTURE}")

time_end = time.time()
log(f"Training took {time_end - time_start:.2f} seconds, per iteration: {(time_end - time_start) / TRAINING_STEPS * 1000:.2f} milliseconds")

test_loss, test_acc = validate('test')
log(f"TEST loss={test_loss:.3f} acc={test_acc*100:.2f}%")




model_binarized = binarize_model()
bin_test_loss, bin_test_acc = validate(dataset="test", model=model_binarized)
log(f"BIN TEST loss={bin_test_loss:.3f} acc={bin_test_acc*100:.2f}%")

with torch.no_grad():
    X = val_images
    for layer_idx in range(0, len(model_binarized.layers)):
        X = model_binarized.layers[layer_idx](X)
    model.dataset_input = val_images
    model.dataset_output = X

model_filename = (
    f"{datetime.now(ZoneInfo(TIMEZONE)).strftime('%Y%m%d-%H%M%S')}"
    f"_binTestAcc{round(bin_test_acc * 10000)}"
    f"_seed{SEED}_epochs{EPOCHS}_{len(NET_ARCHITECTURE)}x{NET_ARCHITECTURE[0]}"
    f"_b{BATCH_SIZE}_lr{LEARNING_RATE * 1000:.0f}.pth"
)
torch.save(model.state_dict(), model_filename)
log(f"Saved to {model_filename}")

WANDB_KEY and wandb.log({
            "final_train_loss": train_loss, "final_train_acc": train_acc*100,
            "final_val_loss": val_loss, "final_val_acc": val_acc*100,
            "final_test_loss": test_loss, "final_test_acc": test_acc*100,
            "final_bin_test_loss": bin_test_loss, "final_bin_test_acc": bin_test_acc*100,
    })
WANDB_KEY and wandb.finish()