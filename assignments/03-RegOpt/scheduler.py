from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Create a new scheduler.

    """

    def __init__(self, optimizer, model_size=512, factor=1, warmup=4000):
        """
        Initialize a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self.factor = factor
        self._rate = 0
        super(CustomLRScheduler, self).__init__(optimizer, -1)

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.model_size ** (-0.5) * min(
            step ** (-0.5), step * self.warmup ** (-1.5)
        )

    def get_lr(self) -> List[float]:
        """
        Create a new get_lr method for scheduler.

        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # if self.last_epoch % self.decay_steps == 0 and self.last_epoch != 0:
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] *= self.decay_rate
        #         # print("param_group['lr'] = ", param_group['lr'])

        return [param_group["lr"] for param_group in self.optimizer.param_groups]
