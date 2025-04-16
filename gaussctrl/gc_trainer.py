# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code to train GaussCtrl model
"""
from dataclasses import dataclass, field
import dataclasses
from typing import Type, Tuple, Dict, Literal
import functools
import time
from rich import box, style
from rich.panel import Panel
from rich.table import Table
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

import torch
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.viewer.server.viewer_elements import ViewerButton
from nerfstudio.utils.decorators import check_main_thread
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer_legacy.server.viewer_state import ViewerLegacyState
from nerfstudio.viewer.viewer import Viewer as ViewerState
from gaussctrl.uco_utils import create_transforms_from_uco, load_uco_data

TRAIN_INTERATION_OUTPUT = Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
TORCH_DEVICE = str

def plot_loss(total_loss):
        # Extract keys (training steps) and values (loss values)
        steps = list(total_loss.keys())
        loss_values = list(total_loss.values())

        # Create a plot
        plt.figure(figsize=(10, 5))
        plt.plot(steps, loss_values, marker='o', linestyle='-', color='b')

        # Add labels and title
        plt.xlabel('Training Step')
        plt.ylabel('Loss Value')
        plt.title('Training Loss over Steps')
        plt.savefig('/home/ddinucci/Desktop/gaussctrl/check/retrain/training_loss.png')
        return
        
@dataclass
class GaussCtrlTrainerConfig(TrainerConfig):
    """Configuration for the GaussCtrlTrainer."""
    _target: Type = field(default_factory=lambda: GaussCtrlTrainer)
    steps_per_save: int = 500
    """Number of steps between saves."""

class GaussCtrlTrainer(Trainer):
    """Trainer for GaussCtrl"""

    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:

        super().__init__(config, local_rank, world_size)
        # reset button
        self.reset_button = ViewerButton(name="Reset Button", cb_hook=self.reset_callback)

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val") -> None:
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datasets into memory
                'inference': does not load any dataset into memory
        """
        if True: #Aggiungi una condizione per entrare qua dentro
            #create_transforms_from_uco()
            self.uco_dataloader = load_uco_data(self.config.pipeline.scene_name, batch=1, subset_lists_file=self.config.pipeline.sql_path)
        self.pipeline = self.config.pipeline.setup( #TODO qui carica il dataset, carica anche le immagini di GT, non so se siano utili
            device=self.device,
            test_mode=test_mode,
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
            uco_data=self.uco_dataloader,
        )
        
        self.optimizers = self.setup_optimizers()
        self._load_checkpoint(uco_data=self.uco_dataloader['splats']) #TODO load checkpoint here
        self.pipeline.render_reverse() #TODO HERE LOAD RENDERED IMAGES AND DEPTH MAP
        if self.pipeline.test_mode == "val":
            self.pipeline.edit_images()

        # set up viewer if enabled
        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None
        if self.config.is_viewer_legacy_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerLegacyState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
            )
            banner_messages = [f"Legacy viewer at: {self.viewer_state.viewer_url}"]
        if self.config.is_viewer_enabled() and self.local_rank == 0:
            datapath = self.config.data
            if datapath is None:
                datapath = self.base_dir
            self.viewer_state = ViewerState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=self,
                train_lock=self.train_lock,
                share=self.config.viewer.make_share_url,
            )
            banner_messages = self.viewer_state.viewer_info
        self._check_viewer_warnings()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers,
                grad_scaler=self.grad_scaler,
                pipeline=self.pipeline,
            )
        )

        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / self.config.logging.relative_log_dir
        writer.setup_event_writer(
            self.config.is_wandb_enabled(),
            self.config.is_tensorboard_enabled(),
            self.config.is_comet_enabled(),
            log_dir=writer_log_path,
            experiment_name=self.config.experiment_name,
            project_name=self.config.project_name,
        )
        writer.setup_local_writer(
            self.config.logging, max_iter=self.config.max_num_iterations, banner_messages=banner_messages
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(self.config), step=0)
        profiler.setup_profiler(self.config.logging, writer_log_path)

    def reset_callback(self, handle: ViewerButton) -> None:
        """Reset the model to the original checkpoint"""
        
        # load checkpoint
        self._load_checkpoint()

        # reset dataset
        self.config.pipeline.datamanager.image_batch['image'] = self.config.pipeline.datamanager.original_image_batch['image'].clone()
        self.config.pipeline.datamanager.image_batch['image_idx'] = self.config.pipeline.datamanager.original_image_batch['image_idx'].clone()

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers
        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        pipeline_state_dict = {k: v for k, v in self.pipeline.state_dict().items() if "ip2p." not in k}
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else pipeline_state_dict,
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        # possibly delete old checkpoints
        if self.config.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()

    def train(self) -> None:
        """Train the model."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"
        
        self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
            self.base_dir / "dataparser_transforms.json"
        )
        # Create the directory if it doesn't exist
        self.render_dir = self.base_dir / "train_render"
        Path(self.render_dir).mkdir(parents=True, exist_ok=True)
        self._init_viewer_state()
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.pipeline.config.render_rate + 1000
            total_loss = {}
            for step in range(self._start_step, self._start_step + num_iterations):
                while self.training_state == "paused":
                    time.sleep(0.01)
                with self.train_lock:
                    with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                        self.pipeline.train()

                        # training callbacks before the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                            )

                        # time the forward pass
                        loss, loss_dict, metrics_dict = self.train_iteration(step)

                        # training callbacks after the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                            )
                        if step%10 == 0:
                            total_loss[step] = loss_dict['main_loss'].detach().cpu().numpy().item()

                self._update_viewer_state(step)

                # a batch of train rays
                if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)
                    # The actual memory allocated by Pytorch. This is likely less than the amount
                    # shown in nvidia-smi since some unused memory can be held by the caching
                    # allocator and some context needs to be created on GPU. See Memory management
                    # (https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
                    # for more details about GPU memory management.
                    writer.put_scalar(
                        name="GPU Memory (MB)", scalar=torch.cuda.max_memory_allocated() / (1024**2), step=step
                    )

                # Do not perform evaluation if there are no validation images
                if self.pipeline.datamanager.eval_dataset:
                    self.eval_iteration(step)

                if step_check(step, self.config.steps_per_save):
                    self.save_checkpoint(step)

                writer.write_out_storage()
        #plot_loss(total_loss)        

        # save checkpoint at the end of training
        self.save_checkpoint(step)

        # write out any remaining events (e.g., total train time)
        writer.write_out_storage()

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("Config File", str(self.config.get_base_dir() / "config.yml"))
        table.add_row("Checkpoint Directory", str(self.checkpoint_dir))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Training Finished :tada:[/bold]", expand=False))

        # after train end callbacks
        for callback in self.callbacks:
            callback.run_callback_at_location(step=step, location=TrainingCallbackLocation.AFTER_TRAIN)

        if not self.config.viewer.quit_on_train_completion:
            self._train_complete_viewer()

    

    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """

        needs_zero = [
            group for group in self.optimizers.parameters.keys() if step % self.gradient_accumulation_steps[group] == 0
        ]
        self.optimizers.zero_grad_some(needs_zero)
        cpu_or_cuda_str: str = self.device.split(":")[0]
        cpu_or_cuda_str = "cpu" if cpu_or_cuda_str == "mps" else cpu_or_cuda_str
        
        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            _, loss_dict, metrics_dict, batch = self.pipeline.get_train_loss_dict(step=step)
            loss = functools.reduce(torch.add, loss_dict.values())

            if step%50 == 0:
                self.save_imgs(_, batch, step, self.render_dir)

        self.grad_scaler.scale(loss).backward()  # type: ignore
        needs_step = [
            group
            for group in self.optimizers.parameters.keys()
            if step % self.gradient_accumulation_steps[group] == self.gradient_accumulation_steps[group] - 1
        ]
        self.optimizers.optimizer_scaler_step_some(self.grad_scaler, needs_step)

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad  # type: ignore
                    total_grad += grad

            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)  # type: ignore

        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            self.optimizers.scheduler_step_all(step)

        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict  # type: ignore
    
    def save_imgs(self, model_outputs, batch, step, path):
        # Convert the tensors to NumPy arrays and scale them to the range [0, 255]
        unedited_image_np = (batch['unedited_image'].cpu().numpy() * 255).astype(np.uint8)
        batch_image_np = (batch['image'].cpu().numpy() * 255).astype(np.uint8)
        model_outputs_rgb_np = (model_outputs['rgb'].detach().cpu().numpy() * 255).astype(np.uint8)

        # Create PIL images from the NumPy arrays
        unedited_image_pil = Image.fromarray(unedited_image_np)
        batch_image_pil = Image.fromarray(batch_image_np)
        model_outputs_rgb_pil = Image.fromarray(model_outputs_rgb_np)

        # Concatenate the images horizontally
        combined_width = unedited_image_pil.width + batch_image_pil.width + model_outputs_rgb_pil.width
        combined_height = unedited_image_pil.height
        combined_image = Image.new('RGB', (combined_width, combined_height))
        combined_image.paste(unedited_image_pil, (0, 0))
        combined_image.paste(batch_image_pil, (unedited_image_pil.width, 0))
        combined_image.paste(model_outputs_rgb_pil, (unedited_image_pil.width + batch_image_pil.width, 0))

        # Save the combined image
        combined_image.save(f'{path}/step_{step}.png')
        return