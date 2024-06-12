import torch
import torch.nn.functional as F

def accuracy(output, target, topk=(1,), max_traj_len=0):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # [k, bs*T]
        correct = pred.eq(target.view(1, -1).expand_as(pred))  # [k, bs*T]

        correct_a = correct[:1].view(-1, max_traj_len)  # [bs, T]
        correct_a0 = correct_a[:, 0].reshape(-1).float().mean().mul_(100.0)
        correct_aT = correct_a[:, -1].reshape(-1).float().mean().mul_(100.0)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        correct_1 = correct[:1]     # (1, bs*T)

        # Success Rate
        trajectory_success = torch.all(correct_1.view(correct_1.shape[1] // max_traj_len, -1), dim=1)
        trajectory_success_rate = trajectory_success.sum() * 100.0 / trajectory_success.shape[0]

        # MIoU
        _, pred_token = output.topk(1, 1, True, True)  # [bs*T, 1]
        pred_inst = pred_token.view(correct_1.shape[1], -1)  # [bs*T, 1]
        pred_inst_set = set()
        target_inst = target.view(correct_1.shape[1], -1)  # [bs*T, 1]
        target_inst_set = set()
        for i in range(pred_inst.shape[0]):
            # print(pred_inst[i], target_inst[i])
            pred_inst_set.add(tuple(pred_inst[i].tolist()))
            target_inst_set.add(tuple(target_inst[i].tolist()))
        MIoU1 = 100.0 * len(pred_inst_set.intersection(target_inst_set)) / len(pred_inst_set.union(target_inst_set))

        batch_size = batch_size // max_traj_len
        pred_inst = pred_token.view(batch_size, -1)  # [bs, T]
        pred_inst_set = set()
        target_inst = target.view(batch_size, -1)  # [bs, T]
        target_inst_set = set()
        MIoU_sum = 0
        for i in range(pred_inst.shape[0]):
            # print(pred_inst[i], target_inst[i])
            pred_inst_set.update(pred_inst[i].tolist())
            target_inst_set.update(target_inst[i].tolist())
            MIoU_current = 100.0 * len(pred_inst_set.intersection(target_inst_set)) / len(
                pred_inst_set.union(target_inst_set))
            MIoU_sum += MIoU_current
            pred_inst_set.clear()
            target_inst_set.clear()

        MIoU2 = MIoU_sum / batch_size
        return res, trajectory_success_rate, MIoU1, MIoU2, correct_a0, correct_aT


def accuracy_observation(observations_pred, observations):   # 均为[bs, T, dim]
    with torch.no_grad():
        bs, T, _ = observations.shape
        losses = torch.zeros((bs, T, T))
        for i in range(T):
            pred = observations_pred[:, i]
            for j in range(T):
                ob = observations[:, j].cuda()
                loss = F.mse_loss(pred, ob, reduction='none')
                loss = torch.sum(loss, dim=-1)  # [bs]
                losses[:, i, j] = loss

        select_id = torch.zeros(bs, T)
        ind = torch.arange(0, bs)

        for i in range(T):
            losses = losses.view(bs, -1)
            current_id = torch.argmin(losses, dim=-1)  # [bs] 选出最大/小的一个数字
            x = current_id // T
            y = current_id % T
            select_id[ind, x] = y.float()
            losses = losses.view(bs, T, T)
            losses[ind, x, :] = torch.inf
            losses[ind, :, y] = torch.inf

        select_id += 1

        select_id_full = torch.zeros((bs, T+2))
        select_id_full[:, 1:-1] = select_id
        select_id_full[:, -1] = T+1

        # gt : 0 1 2 ...
        gt_hanming = torch.arange(0, T+2).unsqueeze(0).repeat(bs, 1)
        hanming = (select_id_full != gt_hanming).float()
        hanming = torch.sum(hanming) / bs

        pair_count_sum = 0.

        for i in range(T+1):
            current = select_id_full[:, i].unsqueeze(1)
            larger = select_id_full[:, i+1:]
            pair_count = (current < larger).float()
            pair_count_sum += torch.sum(pair_count)

        pair_count_sum /= bs
        pair_count_sum *= (2 / ((T+2) * (T+1)))

        return hanming, pair_count_sum


def accuracy_observation2(observations_pred, observations):   # 均为[bs, T, dim]
    with torch.no_grad():
        bs, T, _ = observations.shape
        similars = torch.zeros((bs, T, T))
        for i in range(T):
            pred = observations_pred[:, i]
            for j in range(T):
                ob = observations[:, j].cuda()
                similar = torch.cosine_similarity(pred, ob, dim=-1)
                # loss = torch.sum(loss, dim=-1)  # [bs]
                similars[:, i, j] = similar

        select_id = torch.zeros(bs, T)
        ind = torch.arange(0, bs)

        for i in range(T):
            similars = similars.view(bs, -1)
            current_id = torch.argmax(similars, dim=-1)  # [bs] 选出最大/小的一个数字
            x = current_id // T
            y = current_id % T
            select_id[ind, x] = y.float()
            similars = similars.view(bs, T, T)
            similars[ind, x, :] = -torch.inf
            similars[ind, :, y] = -torch.inf

        select_id += 1

        select_id_full = torch.zeros((bs, T+2))
        select_id_full[:, 1:-1] = select_id
        select_id_full[:, -1] = T+1

        # gt : 0 1 2 ...
        gt_hanming = torch.arange(0, T+2).unsqueeze(0).repeat(bs, 1)
        hanming = (select_id_full != gt_hanming).float()
        hanming = torch.sum(hanming) / bs

        pair_count_sum = 0.

        for i in range(T+1):
            current = select_id_full[:, i].unsqueeze(1)
            larger = select_id_full[:, i+1:]
            pair_count = (current < larger).float()
            pair_count_sum += torch.sum(pair_count)

        pair_count_sum /= bs
        pair_count_sum *= (2 / ((T+2) * (T+1)))

        return hanming, pair_count_sum



