import torch

from .accuracy import *
from model.helpers import AverageMeter


def validate(val_loader1, val_loader2, val_loader3, val_loader4, model, args, text_tensor):
    model.eval()
    acc_top11 = AverageMeter()
    acc_top51 = AverageMeter()
    trajectory_success_rate_meter1 = AverageMeter()
    MIoU1_meter1 = AverageMeter()
    MIoU2_meter1 = AverageMeter()
    A0_acc1 = AverageMeter()
    AT_acc1 = AverageMeter()

    for i_batch, sample_batch in enumerate(val_loader1):
        with torch.no_grad():
            _, start, end, _, _, _, _, labels, _ = sample_batch
            batch_size_current, T = sample_batch[-2].shape  # [bs, (T+1), ob_dim]
            labels = labels.cuda()
            # T1 = 2
            cond = {0: start.cuda().contiguous().float(),
                     T - 1: end.cuda().contiguous().float()}

            output = model(cond, T, if_jump=True)
            actions_pred = output.contiguous()

            x_output = actions_pred[:, :,
                       args.horizon_dim + args.class_dim:args.horizon_dim + args.class_dim + args.action_dim]

            pred_logits = None

            for j in range(T):
                frames_embedding = x_output[:, j, :]
                sim_logits = torch.nn.functional.cosine_similarity(frames_embedding.unsqueeze(1),
                                                                   text_tensor.unsqueeze(0), dim=2)  # 256 666
                sim_logits = sim_logits / 0.1
                sim_logits_softmax = torch.nn.functional.softmax(sim_logits, dim=1)
                sim_logits_softmax = sim_logits_softmax.unsqueeze(1)
                if pred_logits is None:
                    pred_logits = sim_logits_softmax
                else:
                    pred_logits = torch.cat((pred_logits, sim_logits_softmax), dim=1)

            pred_logits = pred_logits.view(-1, pred_logits.shape[-1])
            (acc1, acc5), trajectory_success_rate, MIoU1, MIoU2, a0_acc, aT_acc = \
                accuracy(pred_logits.cpu(), labels.view(-1).cpu(), topk=(1, 5),
                         max_traj_len=T)

        acc_top11.update(acc1.item(), batch_size_current)
        acc_top51.update(acc5.item(), batch_size_current)
        trajectory_success_rate_meter1.update(trajectory_success_rate.item(), batch_size_current)
        MIoU1_meter1.update(MIoU1, batch_size_current)
        MIoU2_meter1.update(MIoU2, batch_size_current)
        A0_acc1.update(a0_acc, batch_size_current)
        AT_acc1.update(aT_acc, batch_size_current)


    return torch.tensor(acc_top11.avg), torch.tensor(acc_top51.avg), \
           torch.tensor(trajectory_success_rate_meter1.avg), \
           torch.tensor(MIoU1_meter1.avg), torch.tensor(MIoU2_meter1.avg), \
           torch.tensor(A0_acc1.avg), torch.tensor(AT_acc1.avg)
