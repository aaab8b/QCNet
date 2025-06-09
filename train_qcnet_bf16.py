# Copyright (c) 2023, Zikang Zhou. All rights reserved.
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
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import torch
from datamodules import ArgoverseV2DataModule
from predictors import QCNet
import torch.nn as nn
from torch.amp import autocast

# v2
def convert_conv_linear_to_bf16(module: nn.Module,exclude_module:list):
    for name, submodule in module.named_children():
        if name in exclude_module:
            print("{} is excluded from converting to bf16.".format(name))
            continue
        if isinstance(submodule, (nn.Conv2d, nn.Linear)):
            if hasattr(submodule, '_original_forward'):
                continue
            submodule._original_forward = submodule.forward
            def new_forward(m):
                def forward_hooked(x):
                    with autocast('cuda', dtype=torch.bfloat16):
                        return m._original_forward(x).to(torch.float32)
                return forward_hooked
            submodule.forward = new_forward(submodule)
        else:
            convert_conv_linear_to_bf16(submodule,exclude_module)
    return module


if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=64)
    QCNet.add_model_specific_args(parser)
    args = parser.parse_args()

    model = QCNet(**vars(args))
    datamodule = {
        'argoverse_v2': ArgoverseV2DataModule,
    }[args.dataset](**vars(args))
    exclude_module=["encoder","t2m_propose_attn_layers","pl2m_propose_attn_layers","a2m_propose_attn_layers",
    "m2m_propose_attn_layer","t2m_refine_attn_layers","pl2m_refine_attn_layers","a2m_refine_attn_layers","m2m_refine_attn_layer"]
    print("exclude bf16 module:{}".format(exclude_module))
    convert_conv_linear_to_bf16(model,exclude_module)
    model=model = model.to(memory_format=torch.channels_last)

    model_checkpoint = ModelCheckpoint(monitor='val_minFDE', save_top_k=5, mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                         strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
                         callbacks=[model_checkpoint, lr_monitor], max_epochs=args.max_epochs)
    trainer.fit(model, datamodule)
