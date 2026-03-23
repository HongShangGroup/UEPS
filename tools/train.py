import sys, os, shutil
import logging
os.environ["MKL_THREADING_LAYER"] = "GNU"

import argparse, yaml
import time, datetime
import numpy as np
import torch
import torch.distributed as dist
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from torchao.float8 import Float8LinearConfig
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.logging import get_logger
from accelerate.utils import AORecipeKwargs

from ueps.datasets import build_data
from ueps.models import build_model, count_param, get_loss
from ueps.utils import get_opt, get_lr_scheduler

logger = get_logger(__name__)


def main(config):
	output_dir = config["output_dir"]

	dataloader_config = DataLoaderConfiguration(split_batches=True, non_blocking=True)
	with_tracking = config.get("with_tracking", True)
	if with_tracking:
		accelerator = Accelerator(gradient_accumulation_steps=int(config["accmu_steps"]), dataloader_config=dataloader_config, log_with="tensorboard", project_dir=output_dir)
	else:
		accelerator = Accelerator(gradient_accumulation_steps=int(config["accmu_steps"]), dataloader_config=dataloader_config)

	if accelerator.mixed_precision == "fp8":
		fp8_config = Float8LinearConfig(
				enable_fsdp_float8_all_gather=True,
				pad_inner_dim=True,
			)
		accelerator.ao_recipe_handler = AORecipeKwargs(config=fp8_config)

	# Make one log on every process with the configuration for debugging.
	logging.basicConfig(
		format="%(asctime)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO,
	)

	if accelerator.is_main_process:
		if os.path.exists(output_dir) and os.path.isdir(output_dir):
			shutil.rmtree(output_dir)
		os.makedirs(output_dir)

		# Setup logging to file
		file_handler = logging.FileHandler(os.path.join(output_dir, f"{output_dir.split('/')[-1]}.log"))
		file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
		logging.getLogger().addHandler(file_handler)
		
		logger.info(accelerator.state)
		logger.info(f"Experiment directory: {output_dir}")

		with open(os.path.join(output_dir, "config.yaml"), "w") as f:
			yaml.dump(config, f)

	seed = config.get("seed", 42)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	world_size = accelerator.num_processes
	accmu_steps = int(config["accmu_steps"])
	global_batch_size = config["batch_size"]
	batch_size_per_gpu = global_batch_size // world_size
	train_loader, valid_loader, test_loader = build_data(config, world_size)

	logger.info(f"{config['which_data']} train has {len(train_loader.dataset)} images")
	logger.info(f"{config['which_data']} valid has {len(valid_loader.dataset)} images")
	logger.info(f"{config['which_data']} test has {len(test_loader.dataset)} images")
	if accelerator.distributed_type == "FSDP":
		logger.info("Using FSDP distributed training")
		logger.info(f"Batch size: {global_batch_size}")
		logger.info(f"Gradient accumulation steps: {accmu_steps}, total effective batch size: {global_batch_size * accmu_steps}")
	elif accelerator.distributed_type == "MULTI_GPU":
		logger.info("Using DDP distributed training")
		logger.info(f"Total world size: {world_size}, batch size per gpu: {batch_size_per_gpu}, global batch size: {global_batch_size}")
		logger.info(f"Gradient accumulation steps: {accmu_steps}, total effective batch size: {global_batch_size * accmu_steps}")

	model = build_model(config)
	_, param_info = count_param(model)
	logger.info(param_info)

	# Losses
	loss_fn_train = get_loss(config["loss_type_train"])
	loss_fn_eval = get_loss(config["loss_type_eval"])

	# Optimizer & LR scheduler
	optimizer = get_opt(model, config["opt_type"], config["lr"])

	max_steps = config.get("max_steps", "epoch")
	if max_steps == "epoch":
		assert config.get("epochs") is not None, "Please set epochs if max_steps is epoch"
		max_steps = int(len(train_loader) * config["epochs"])

	scheduler = get_lr_scheduler(optimizer, config["lrs_type"], max_steps//accmu_steps, config["lr"])

	# Prepare with accelerator
	model, optimizer, train_loader, valid_loader, scheduler = accelerator.prepare(
		model, optimizer, train_loader, valid_loader, scheduler
	)
	device = accelerator.device
	if with_tracking:
		tracker_config = {k: v if isinstance(v, (int, float, str, bool, torch.Tensor)) else str(v) for k, v in config.items()}
		accelerator.init_trackers("tensorboard", config=tracker_config)

	# load checkpoint (optional)
	overall_steps = 0
	overall_epochs = 0
	if config["resume_from"]:
		if config["resume_from"] is not None or config["resume_from"] != "":
			accelerator.load_state(config['resume_from'])
			ckpt_name = os.path.basename(config['resume_from'])
			resume_step = int(ckpt_name.replace("checkpoint_", ""))
			overall_steps += resume_step
			finish_epoch = resume_step // len(train_loader)
			overall_epochs += finish_epoch
			resume_step -= finish_epoch * len(train_loader)

	accmu_steps = config["accmu_steps"]
	log_every = int(config.get("log_every", int(len(train_loader))))
	ckpt_every = int(config.get("ckpt_every", int(len(train_loader))))
	ckpt_dir = os.path.join(output_dir, "checkpoints")
	ckpt_best_path_pt = os.path.join(output_dir, "ckpt_pick.pth")
	ckpt_best_path_accel = os.path.join(ckpt_dir, "ckpt_pick")
	if accelerator.is_main_process:
		if not os.path.exists(ckpt_dir):
			os.makedirs(ckpt_dir)

	best_steps = overall_steps
	best_epochs = None
	best_loss = 100.0
	start_time = time.time()
	train_start_time = start_time
	lr_arr = []

	log_steps = 0
	log_loss = 0
	model.train()
	while True:
		if config["resume_from"] and resume_step is not None:
			active_dataloader = accelerator.skip_first_batches(train_loader, resume_step)
			resume_step = None
		else:
			active_dataloader = train_loader
		logger.info(f"Epoch {overall_epochs}")
		for step, batch in enumerate(active_dataloader):
			with accelerator.accumulate(model):
				x, y, mask, _, _ = batch
				x, y, mask = x.to(device), y.to(device), mask.to(device)
				pred = model(x, mask)
				loss = loss_fn_train(pred, y)
				accelerator.backward(loss)
				if accelerator.sync_gradients:
					accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()
				lr_arr.append(scheduler.get_last_lr()[0])

			log_loss += loss.item()
			log_steps += 1
			overall_steps += 1
			if overall_steps % log_every == 0:
				accelerator.wait_for_everyone()
				end_time = time.time()
				secs_per_step = (end_time - start_time) / log_steps
				avg_loss = torch.tensor(log_loss / log_steps, device=device)
				dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
				avg_loss = avg_loss.detach().float()
				logger.info(
					f"step {overall_steps}/{max_steps} | Train Loss: {avg_loss:.12f} | Secs/Step: {secs_per_step:.5f}"
				)
				if log_every == len(train_loader):
					logger.info(f"finish one epoch training, avg loss {avg_loss:.12f}")
				accelerator.log({"train_loss": avg_loss, "lr": scheduler.get_last_lr()[0]}, step=overall_steps)
				log_loss = 0
				log_steps = 0
				start_time = time.time()
				
			if overall_steps % ckpt_every == 0:
				accelerator.wait_for_everyone()
				ckpt_path = os.path.join(ckpt_dir, f"checkpoint_{overall_steps}")
				accelerator.save_state(output_dir=ckpt_path, safe_serialization=False)
				
				if accelerator.is_main_process:
					for folder in os.listdir(ckpt_dir):
						# Only process directories that look like checkpoints and are NOT the current one
						if folder.startswith("checkpoint_") and folder != f"checkpoint_{overall_steps}":
							folder_path = os.path.join(ckpt_dir, folder)
							if os.path.isdir(folder_path):
								try:
									shutil.rmtree(folder_path)
								except OSError as e:
									logger.error(f"Error deleting {folder_path}: {e}")
				accelerator.wait_for_everyone()

				# Evaluate
				logger.info(f"Start evaluation at step {overall_steps}")
				model.eval()
				val_loss = 0.0
				val_steps = 0
				with torch.no_grad():
					for vstep, vbatch in enumerate(valid_loader):
						x, y, mask, _, _ = vbatch
						x, y, mask = x.to(device), y.to(device), mask.to(device)
						pred = model(x, mask)
						loss = loss_fn_eval(pred, y)
						val_loss += loss.item()
						val_steps += 1
				val_loss = torch.tensor(val_loss / val_steps, device=device)
				dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
				model.train()
				
				logger.info(f"Evaluation avg loss: {val_loss:.12f}")
				accelerator.log({"val_loss": val_loss}, step=overall_steps)
				
				if val_loss < best_loss:
					best_loss = val_loss
					best_steps = overall_steps
					if ckpt_every == len(train_loader):
						best_epochs = overall_epochs

					accelerator.save_state(output_dir=ckpt_best_path_accel, safe_serialization=False)
					state_dict = accelerator.get_state_dict(model)
					state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
					
					if accelerator.is_main_process:
						all_states = {
							"steps": best_steps,
							"best_loss": best_loss,
							"state_dict": state_dict,
						}
						torch.save(all_states, ckpt_best_path_pt)
						logger.info(f"New best loss {best_loss:.12f} at step {best_steps}, saving checkpoint to {ckpt_best_path_accel}")
				accelerator.wait_for_everyone()

			if overall_steps >= max_steps:
				break
		if overall_steps >= max_steps:
			break
		logger.info(f"At Step {overall_steps} finished Epoch {overall_epochs}")
		overall_epochs += 1

	if accelerator.is_main_process:
		total_time = time.time() - train_start_time
		total_time_str = str(datetime.timedelta(seconds=int(total_time)))
		logger.info(f"finish all training in {total_time_str}")
		if best_epochs is not None:
			print(f"validation pick epoch {best_epochs}")

	accelerator.end_training()


def invoke_main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, default="../configs/fastmrimc-knee_e2evarnet.yaml")
	parser.add_argument("--output_dir", default="../../../model_export/newgroup/newname")
	args = parser.parse_args()

	with open(args.config, "r") as f:
		config = yaml.safe_load(f)
	config["output_dir"] = args.output_dir

	main(config)
	logger.info("All Done.")

if __name__ == "__main__":
	invoke_main()
