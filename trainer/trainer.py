import torch
import tqdm
import torch.nn as nn
import nomenclature
import numpy as np
from torch.optim.swa_utils import AveragedModel, SWALR

from torchinfo import summary

class NotALightningTrainer(object):

    def __init__(self,
            args,
            callbacks,
            logger
        ):
        self.args = args

        self.epoch = 0
        self.global_step = 0

        self.logger = logger
        self.logger.trainer = self

        self.callbacks = callbacks
        for callback in callbacks:
            callback.trainer = self

        self.should_stop = False


    def stop(self):
        self.should_stop = True

    def fit(self, model, train_dataloader, evaluators):
        model.log = self.logger.log

        optimizer = model.configure_optimizers()
        model.trainer = self

        if not hasattr(model.model, 'module'):
            # distributed data parallel??
            model.model = nn.DataParallel(model.model, device_ids=[0, 1])
            model.model = model.model.to(nomenclature.device)

        summary(model.model, input_shape = (self.args.batch_size, 54))

        self.logger.watch(model.model)
        self.model_hook = model.model

        # self.scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.epochs):
            if self.should_stop:
                break

            for callback in self.callbacks:
                callback.on_epoch_start()

            pbar = tqdm.tqdm(train_dataloader, total = len(train_dataloader))

            model.training_epoch_start(epoch)
            for i, data in enumerate(pbar):
                self.global_step += 1
                optimizer.zero_grad(set_to_none = True)

                for callback in self.callbacks:
                    callback.on_batch_start()

                for key in data.keys():
                    data[key] = data[key].to(nomenclature.device)

                loss = model.training_step(data, i)
                loss = loss / self.args.accumulation_steps

                # self.scaler.scale(loss).backward()
                loss.backward()

                if (i + 1) % self.args.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.5)

                    optimizer.step()
                    # self.scaler.step(optimizer)
                    # self.scaler.update()

                    for callback in self.callbacks:
                        callback.on_batch_end()

                pbar.set_description(f'Epoch {self.epoch} / {self.args.epochs} | ' + ' | '.join([f'{k}={np.round(v, 4)}' for k,v in self.logger.on_step_metrics.items()]))

            model.training_epoch_end()
            self.epoch += 1
            model.model.train(False)
            with torch.no_grad():
                outputs = []
                for evaluator in evaluators:
                    value = evaluator.trainer_evaluate(self.global_step)
                    self.logger.log(f'val_acc_{evaluator.__class__.__name__}', value, on_step = True, force_log = True)

            model.model.train(True)

            for callback in self.callbacks:
                callback.on_epoch_end()
