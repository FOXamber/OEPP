import copy

from model.helpers import AverageMeter
from .accuracy import *


def cycle(dl):
    while True:
        for data in dl:
            yield data


class EMA():
    """
        empirical moving average
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            datasetloader1,
            datasetloader2,
            datasetloader3,
            datasetloader4,
            ema_decay=0.995,
            train_lr=1e-5,
            gradient_accumulate_every=1,
            step_start_ema=400,
            update_ema_every=10,
            log_freq=100,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataloader1 = cycle(datasetloader1)
        self.dataloader2 = cycle(datasetloader2)
        self.dataloader3 = cycle(datasetloader3)
        self.dataloader4 = cycle(datasetloader4)
        self.optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=train_lr, weight_decay=0.0)

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, n_train_steps, if_calculate_acc, args, scheduler, train_text_tensor):
        self.model.train()
        self.ema_model.train()
        losses1 = AverageMeter()
        self.optimizer.zero_grad()
        cost = torch.nn.CrossEntropyLoss()

        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch1 = next(self.dataloader1)
                _, start, end, _, _, _, _, labels, _ = batch1
                bs1, T1 = batch1[-2].shape  # [bs, (T+1), ob_dim]
                labels = labels.cuda()
                img_tensors1 = torch.zeros((bs1, T1, args.class_dim + args.action_dim + args.observation_dim + args.horizon_dim))
                img_tensors1[:, 0, args.horizon_dim+args.class_dim+args.action_dim:] = start.cuda().contiguous().float()
                img_tensors1[:, -1, args.horizon_dim+args.class_dim+args.action_dim:] = end.cuda().contiguous().float()
                img_tensors1 = img_tensors1.cuda()

                if args.class_dim != 0:
                    assert 0
                else:
                    cond1 = {0: img_tensors1[:, 0, args.horizon_dim + args.class_dim + args.action_dim:].float(),
                             T1 - 1: img_tensors1[:, -1, args.horizon_dim + args.class_dim + args.action_dim:].float()}

                if args.horizon_dim != 0:
                    horizon_onehot1 = torch.zeros((bs1, args.horizon_dim))
                    horizon_onehot1[:, 0] = 1.
                    temp1 = horizon_onehot1.unsqueeze(1)
                    horizon_onehot1 = temp1.repeat(1, T1, 1)
                    img_tensors1[:, :, :args.horizon_dim] = horizon_onehot1
                    cond1['horizon'] = horizon_onehot1

                x1 = img_tensors1.float()
                x_output = self.model.module.loss(x1, cond1)
                loss = 0
                x_output = x_output[:, :, args.horizon_dim + args.class_dim:args.horizon_dim + args.class_dim + args.action_dim]

                for j in range(T1):
                    frames_embedding = x_output[:, j, :]
                    label = labels[:, j]
                    gt_embedding = train_text_tensor[label]
                    mse_loss = torch.nn.MSELoss(reduction='sum')
                    m_loss = mse_loss(frames_embedding, gt_embedding) / bs1
                    sim_logits = torch.nn.functional.cosine_similarity(frames_embedding.unsqueeze(1),
                                                                       train_text_tensor.unsqueeze(0), dim=2)  # 256 666
                    sim_logits = sim_logits / 0.1
                    sim_logits_softmax = torch.nn.functional.softmax(sim_logits, dim=1)
                    loss_ce = cost(sim_logits_softmax, label)
                    loss += m_loss / m_loss.item() * args.para_mse + loss_ce / loss_ce.item() * (1 - args.para_ce)

                loss = loss / self.gradient_accumulate_every
                loss.backward()
                losses1.update(loss.item(), bs1)
                self.optimizer.step()
                self.optimizer.zero_grad()

            scheduler.step()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            self.step += 1

        if if_calculate_acc:
            with torch.no_grad():
                x_output = self.ema_model(cond1, T1, if_jump=True)
                x_output = x_output[:, :,
                           args.horizon_dim + args.class_dim:args.horizon_dim + args.class_dim + args.action_dim]

                pred_logits = None

                for j in range(T1):
                    frames_embedding = x_output[:, j, :]
                    sim_logits = torch.nn.functional.cosine_similarity(frames_embedding.unsqueeze(1),
                                                                       train_text_tensor.unsqueeze(0), dim=2)  # 256 666
                    sim_logits = sim_logits / 0.1
                    sim_logits_softmax = torch.nn.functional.softmax(sim_logits, dim=1)
                    sim_logits_softmax = sim_logits_softmax.unsqueeze(1)
                    # print(sim_logits_softmax)
                    if pred_logits is None:
                        pred_logits = sim_logits_softmax
                    else:
                        pred_logits = torch.cat((pred_logits, sim_logits_softmax), dim=1)

                pred_logits = pred_logits.view(-1, pred_logits.shape[-1])
                (acc11, acc51), trajectory_success_rate1, MIoU11, MIoU21, a0_acc1, aT_acc1 = \
                        accuracy(pred_logits.cpu(), labels.view(-1).cpu(), topk=(1, 5),
                             max_traj_len=T1)

                return torch.tensor(losses1.avg), acc11, acc51, torch.tensor(trajectory_success_rate1), \
                       torch.tensor(MIoU11), torch.tensor(MIoU21), a0_acc1, aT_acc1


        else:
            return torch.tensor(losses1.avg)
