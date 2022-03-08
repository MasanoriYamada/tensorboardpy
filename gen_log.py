import torch
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard import SummaryWriter


def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None):
    torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
    if type(hparam_dict) is not dict or type(metric_dict) is not dict:
        raise TypeError('hparam_dict and metric_dict should be dictionary.')
    exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

    self.file_writer.add_summary(exp)
    self.file_writer.add_summary(ssi)
    self.file_writer.add_summary(sei)
    for k, v in metric_dict.items():
        if v is not None:
            self.add_scalar(k, v)
setattr(SummaryWriter, "add_hparams", add_hparams)


for i in range(5):
    save_metrics = {'train/acc': None, 'train/loss': None}
    writer = SummaryWriter(f'runs/{i}')
    for step in range(50):
        writer.add_scalar('train/acc', 10*i+step, step)
        writer.add_scalar('train/loss', 10*i - step, step)
    writer.add_hparams({'lr': 0.1*i, 'bsize': i, 'gpu': True, 'opt': 'adam'}, save_metrics)
writer.close()
