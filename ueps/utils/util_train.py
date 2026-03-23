import torch
from torch.nn.parallel import DistributedDataParallel as DDP

f16_type = torch.bfloat16

def train_acc_setting(model, use_ddp, use_compile, precision_type, rank=0):
    """Train acceleration (DDP, Compile, AMP) related setting.

    Parameters:
        model: model instance
        use_ddp (bool): if True, use DDP
        use_compile (bool): if True, use torch.compile()
        precision_type (str): choose from {"amp", "tensor_core", "full"}
        rank (int): needed for DDP

    Returns:
        model: model for reading states
        model_train: model for train
        model_eval: model for evaluation
        train_epoch: function for training one epoch
        scaler: used for amp

    model_train / model_eval / model are the same in plain mode.
    model differs from model_train / model_eval when use_ddp=True.
    """
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.optimize_ddp = True

    if precision_type == "amp":
        scaler = torch.amp.GradScaler('cuda')
        torch.backends.cuda.matmul.allow_tf32 = True

    elif precision_type == "tensor_core":
        scaler = None
        torch.backends.cuda.matmul.allow_tf32 = True

    elif precision_type == "full":
        scaler = None
        torch.backends.cuda.matmul.allow_tf32 = False

    if use_ddp:
        model = DDP(model, device_ids=[rank])

        if use_compile:
            torch._dynamo.reset()
            model = torch.compile(model)

        model_train = model
        model_eval = model
        model = model.module
    else:
        if use_compile:
            torch._dynamo.reset()
            model_eval = torch.compile(model)
            model_train = model_eval
        else:
            model_eval = model
            model_train = model

    return model, model_train, model_eval, scaler

def get_opt(model, opt_type, lr):
    if opt_type == "momentum":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif opt_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                                      weight_decay=0.1)
    elif opt_type == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise RuntimeError(f"opt_type {opt_type} is not supported")

    return optimizer

def get_lr_scheduler(optimizer, lrs_type, N, lr):
    if lrs_type == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[round(N*0.8)],
            gamma=0.1,
            )
    elif lrs_type == "warmup":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=round(N*0.1),
            )
    elif lrs_type == "cos1":
        scheduler1 = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=round(N*0.01),
            )
        scheduler2 = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=N,
            )
        scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=round(N*0.3),
            eta_min=lr*0.1,
            )
        scheduler4 = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=0.1,
            total_iters=N,
            )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2, scheduler3, scheduler4],
            milestones=[round(N*0.01), round(N*0.65), round(N*0.95)],
            )
    elif lrs_type == "none":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[round(N*2)],
            gamma=1.0,
            )
    else:
        RuntimeError("Incorrect lrs_type")

    return scheduler
