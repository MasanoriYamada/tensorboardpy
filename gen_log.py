from torch.utils.tensorboard import SummaryWriter


for i in range(5):
    save_metrics = {'train/acc': None, 'train/loss': None}
    writer = SummaryWriter(f'runs/{i}')
    for step in range(50):
        writer.add_scalar('train/acc', 10*i+step, step)
        writer.add_scalar('train/loss', 10*i - step, step)
    writer.add_hparams({'lr': 0.1*i, 'bsize': i, 'gpu': True, 'opt': 'adam'}, save_metrics)
writer.close()
