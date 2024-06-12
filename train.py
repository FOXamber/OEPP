
import torch
import yaml
import os
import json
import random
import argparse
import numpy as np
from datetime import datetime


from dataset.dataset import Seq_action
from model.MLP import MLP
from model.attention import TransformerEncoder


def get_text_tensor(action_pool, action_text_dict):
    text_list = []
    for action in action_pool:
        text_embedding = torch.tensor(action_text_dict[action])
        text_list.append(text_embedding)
    text_tensor = torch.cat([tensor for tensor in text_list], dim=0)
    return text_tensor


def task_match(gt_action, pre_action, task_info):
    is_match = 0
    for task in task_info:
        gt_true = 0
        pre_true = 0
        action_list = task_info[task]['action_list']
        for action in action_list:
            if gt_action == action:
                gt_true = 1
            if pre_action == action:
                pre_true = 1
        if gt_true == 1 and pre_true == 1:
            is_match = 1
    return is_match


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="attention_config.yaml")

    #
    args = parser.parse_args()


    with open(args.config,'r') as file:
        config = yaml.safe_load(file)


    split = config['split']
    T = config['T']
    is_pad = config['is_pad']
    feat = config['feature']
    model_n = config['model']['model_n']
    epochs = config['training']['epochs']

    random_seed = config['seed']
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    output_dir = os.path.join('results', model_n,datetime.now().strftime("%m%d%H%M%S"))
    os.makedirs(output_dir)
    output_config = yaml.dump(config)
    with open(os.path.join(output_dir,'config.yaml'),'w') as file:
        file.write(output_config)


    with open('data/task_info.json') as f:
        task_info = json.load(f)
    with open('data/base_action_pool_' + str(split) + '.json') as f:
        train_action_pool = json.load(f)
    with open('data/novel_action_pool_' + str(split) + '.json') as f:
        test_action_pool = json.load(f)
    with open('data/total_action_pool.json') as f:
        total_action_pool = json.load(f)
    print("train_action_pool_size: ", len(train_action_pool))
    print("test_action_pool_size: ", len(test_action_pool))
    print("total_action_pool_size: ", len(total_action_pool))

    if feat == 's3d':
        with open('data/s3d_action_feat_dict.json') as f:
            actions_text_dict = json.load(f)
        feat_dim = 512
    elif feat == 'videoclip':
        with open('data/vc_action_feat_dict.json') as f:
            actions_text_dict = json.load(f)
        feat_dim = 768
    else:
        with open('data/action_feat_dict.json') as f:
            actions_text_dict = json.load(f)
        feat_dim = 512

    train_text_tensor = get_text_tensor(train_action_pool, actions_text_dict).cuda()
    test_text_tensor = get_text_tensor(test_action_pool, actions_text_dict).cuda()
    total_text_tensor = get_text_tensor(total_action_pool, actions_text_dict).cuda()
    print(train_text_tensor.shape)
    print(test_text_tensor.shape)
    print(total_text_tensor.shape)
    #文本特征先存储，train/test/total

    train_train_dataset = Seq_action(root='data/',split=split,feat=feat, T=T, is_pad=is_pad, is_total=0, is_val=0)
    test_novel_dataset = Seq_action(root='data/',split=split, feat=feat, T=T, is_pad=is_pad, is_total=0, is_val=1)
    test_base_dataset = Seq_action(root='data/',split=split, feat=feat, T=T, is_pad=is_pad, is_total=0, is_val=2)
    train_val_dataset = Seq_action(root='data/', split=split, feat=feat, T=T, is_pad=is_pad,
                                   is_total=0, is_val=3)

    print("train_train: ",len(train_train_dataset))
    print("test_base: ",len(test_base_dataset))
    print("test_novel: ",len(test_novel_dataset))
    print("train_tcal: ",len(train_val_dataset))

    train_train_loader = torch.utils.data.DataLoader(dataset=train_train_dataset, batch_size=256, shuffle=True, drop_last=False)
    test_base_loader = torch.utils.data.DataLoader(dataset=test_base_dataset, batch_size=1, shuffle=False, drop_last=False)
    test_novel_loader = torch.utils.data.DataLoader(dataset=test_novel_dataset, batch_size=1, shuffle=False, drop_last=False)
    train_val_loader = torch.utils.data.DataLoader(dataset=train_val_dataset, batch_size=1, shuffle=False,
                                                    drop_last=False)


    if model_n == 'MLP':
        model = MLP(dim=feat_dim,T=T).cuda()
    elif model_n == 'attention':
        model = TransformerEncoder(input_dim=feat_dim, hidden_dim=feat_dim, num_layers=config['model']['num_layers'], num_heads=config['model']['num_heads'], T=config['T']).cuda()

    print(model)

    with open(os.path.join(output_dir,'output.txt'), 'w') as f:
        print(model, file=f)
        print("\n", file=f)

    cost = torch.nn.CrossEntropyLoss()
    mse_cost = torch.nn.MSELoss()
    # soft_dtw_cost = SoftDTW(use_cuda=True,gamma=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)

    best_test_sr = 0
    for epoch in range(epochs):

        train_acc_num = 0
        train_total_num = 0
        total_loss = 0
        model.train()
        for data in train_train_loader:
            _, _, _, frames,  action_list, action_tensors, _, labels, _ = data

            frames = frames.cuda()
            labels = labels.cuda()
            pre_labels = torch.zeros(frames.shape[0],T).cuda()
            action_tensors = action_tensors.cuda()
            if model_n == 'MLP':
                frames_embedding_list = model(frames)  # [[B,512],[B,512]]
            elif model_n == 'attention':
                frames_embedding_list = model(frames)  # [[B,512],[B,512]]
            # print(len(frames_embedding_list))
            ce_loss = 0
            mse_loss = 0
            for i in range(T):
                frames_embedding = frames_embedding_list[i]
                label = labels[:,i]
                action_tensor = action_tensors[:,i,:]
                sim_logits = torch.nn.functional.cosine_similarity(frames_embedding.unsqueeze(1),
                                                               train_text_tensor.unsqueeze(0), dim=2)  # 256 666
                sim_logits = sim_logits / 0.1
                sim_logits_softmax = torch.nn.functional.softmax(sim_logits, dim=1)
                # print(sim_logits_softmax)
                ce_loss += cost(sim_logits_softmax, label)
                mse_loss += mse_cost(frames_embedding,action_tensor)
                gt_indices = label
                pred_indices = torch.argmax(sim_logits_softmax, dim=1)
                pre_labels[:,i] = pred_indices

                correct_predictions = torch.eq(pred_indices, gt_indices)
                num_correct = torch.sum(correct_predictions).item()
                train_acc_num += num_correct
                train_total_num += label.shape[0]

            loss = config['loss']['ce_w']*ce_loss + config['loss']['mse_w']* mse_loss
            # loss = 0.8 * ce_loss + 0.2 * mse_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        print("epoch: ", epoch)
        print("loss: ", total_loss / len(train_train_loader))
        print("train_acc: ", train_acc_num / train_total_num)

        val_acc_num = 0
        val_total_num = 0
        val_task_match = 0
        val_miou = 0
        val_sr_num = 0

        model.eval()
        for data in train_val_loader:
            set_pred = set()
            set_true = set()
            correct_f = 0
            _, _, _, frames,  action_list,_, _, labels, _ = data

            frames = frames.cuda()
            labels = labels.cuda()
            if model_n == 'MLP':
                # action_tensors = action_tensors.cuda()
                frames_embedding_list = model(frames)  # [[B,512],[B,512]]

            elif model_n == 'attention':
                frames_embedding_list = model(frames)  # [[B,512],[B,512]]
            # print(len(frames_embedding_list))
            # end_embedding = end_model(end_frames)
            for i in range(T):
                frames_embedding = frames_embedding_list[i]
                label = labels[:, i]
                sim_logits = torch.nn.functional.cosine_similarity(frames_embedding.unsqueeze(1),
                                                                   train_text_tensor.unsqueeze(0), dim=2)  # 256 666
                sim_logits = sim_logits / 0.1
                sim_logits_softmax = torch.nn.functional.softmax(sim_logits, dim=1)
                # print(sim_logits_softmax)
                # loss += cost(sim_logits_softmax, label)
                gt_indices = label
                pred_indices = torch.argmax(sim_logits_softmax, dim=1)
                set_pred.add(pred_indices[0].tolist())
                set_true.add(gt_indices[0].tolist())

                correct_predictions = torch.eq(pred_indices, gt_indices)
                num_correct = torch.sum(correct_predictions).item()
                if num_correct == 1:
                    correct_f +=1
                val_acc_num += num_correct
                val_total_num += label.shape[0]
            if correct_f == T:
                val_sr_num += 1
            iou = 100.0 * len(set_pred.intersection(set_true)) / len(set_pred.union(set_true))
            val_miou += iou
            # total_loss += loss
        print("val_acc: ", val_acc_num / val_total_num)
        print("val_sr: ", val_sr_num/len(train_val_dataset))
        print("val_miou: ", val_miou/len(train_val_dataset))

        if best_test_sr < (val_sr_num / len(train_val_dataset)):
            best_test_sr = val_sr_num / len(train_val_dataset)
            model_dir = str(split)+'_'+str(T)+'.pth'
            # model_dir = os.path.join('results',model_dir)
            model_dir = os.path.join(output_dir,model_dir)
            torch.save(model, model_dir)

        print("best_val_sr: ", best_test_sr)

    #--------test-----------
    model = torch.load(model_dir)
    model.eval()
    print("-------test----------")

    val_acc_num = 0
    val_total_num = 0
    val_task_match = 0
    val_miou = 0
    val_sr_num = 0

    for data in train_val_loader:
        set_pred = set()
        set_true = set()
        correct_f = 0
        _, _, _, frames, action_list, _, _, labels, _ = data

        frames = frames.cuda()
        labels = labels.cuda()
        if model_n == 'MLP':
            # action_tensors = action_tensors.cuda()
            frames_embedding_list = model(frames)  # [[B,512],[B,512]]

        elif model_n == 'attention':
            frames_embedding_list = model(frames)  # [[B,512],[B,512]]
        # print(len(frames_embedding_list))
        # end_embedding = end_model(end_frames)
        for i in range(T):
            frames_embedding = frames_embedding_list[i]
            label = labels[:, i]
            sim_logits = torch.nn.functional.cosine_similarity(frames_embedding.unsqueeze(1),
                                                               train_text_tensor.unsqueeze(0), dim=2)  # 256 666
            sim_logits = sim_logits / 0.1
            sim_logits_softmax = torch.nn.functional.softmax(sim_logits, dim=1)
            # print(sim_logits_softmax)
            # loss += cost(sim_logits_softmax, label)
            gt_indices = label
            pred_indices = torch.argmax(sim_logits_softmax, dim=1)
            set_pred.add(pred_indices[0].tolist())
            set_true.add(gt_indices[0].tolist())

            correct_predictions = torch.eq(pred_indices, gt_indices)
            num_correct = torch.sum(correct_predictions).item()
            if num_correct == 1:
                correct_f += 1
            val_acc_num += num_correct
            val_total_num += label.shape[0]
        if correct_f == T:
            val_sr_num += 1
        iou = 100.0 * len(set_pred.intersection(set_true)) / len(set_pred.union(set_true))
        val_miou += iou
        # total_loss += loss
    print("val_acc: ", val_acc_num / val_total_num)
    print("val_sr: ", val_sr_num / len(train_val_dataset))
    print("val_miou: ", val_miou / len(train_val_dataset))

    with open(os.path.join(output_dir, 'output.txt'), 'a') as f:
        print("val_acc: ", val_acc_num / val_total_num,file=f)
        print("\n", file=f)
        print("val_sr: ", val_sr_num / len(train_val_dataset),file=f)
        print("\n", file=f)
        print("val_miou: ", val_miou / len(train_val_dataset),file=f)
        print("\n", file=f)

    base_acc_num = 0
    base_total_num = 0
    base_task_match = 0
    base_miou = 0
    base_sr_num = 0

    for data in test_base_loader:
        set_pred = set()
        set_true = set()
        correct_f = 0
        _, _, _, frames, action_list, _, _, labels, _ = data

        frames = frames.cuda()
        labels = labels.cuda()
        if model_n == 'MLP':
            # action_tensors = action_tensors.cuda()
            frames_embedding_list = model(frames)  # [[B,512],[B,512]]

        elif model_n == 'attention':
            frames_embedding_list = model(frames)  # [[B,512],[B,512]]
        # print(len(frames_embedding_list))
        # end_embedding = end_model(end_frames)
        for i in range(T):
            frames_embedding = frames_embedding_list[i]
            label = labels[:, i]
            sim_logits = torch.nn.functional.cosine_similarity(frames_embedding.unsqueeze(1),
                                                               train_text_tensor.unsqueeze(0), dim=2)  # 256 666
            sim_logits = sim_logits / 0.1
            sim_logits_softmax = torch.nn.functional.softmax(sim_logits, dim=1)
            # print(sim_logits_softmax)
            # loss += cost(sim_logits_softmax, label)
            gt_indices = label
            pred_indices = torch.argmax(sim_logits_softmax, dim=1)
            set_pred.add(pred_indices[0].tolist())
            set_true.add(gt_indices[0].tolist())

            correct_predictions = torch.eq(pred_indices, gt_indices)
            num_correct = torch.sum(correct_predictions).item()
            if num_correct == 1:
                correct_f += 1
            base_acc_num += num_correct
            base_total_num += label.shape[0]
        if correct_f == T:
            base_sr_num += 1
        iou = 100.0 * len(set_pred.intersection(set_true)) / len(set_pred.union(set_true))
        base_miou += iou
        # total_loss += loss
    print("base_acc: ", base_acc_num / base_total_num)
    print("base_sr: ", base_sr_num / len(test_base_dataset))
    print("base_miou: ", base_miou / len(test_base_dataset))
    with open(os.path.join(output_dir, 'output.txt'), 'a') as f:
        print("base_acc: ", base_acc_num / base_total_num,file=f)
        print("\n", file=f)
        print("base_sr: ", base_sr_num / len(test_base_dataset),file=f)
        print("\n", file=f)
        print("base_miou: ", base_miou / len(test_base_dataset),file=f)
        print("\n", file=f)

    novel_acc_num = 0
    novel_total_num = 0
    novel_task_match = 0
    novel_miou = 0
    novel_sr_num = 0

    for data in test_novel_loader:
        set_pred = set()
        set_true = set()
        correct_f = 0
        _, _, _, frames, action_list, _, _, labels, _ = data

        frames = frames.cuda()
        labels = labels.cuda()
        if model_n == 'MLP':
            # action_tensors = action_tensors.cuda()
            frames_embedding_list = model(frames)  # [[B,512],[B,512]]

        elif model_n == 'attention':
            frames_embedding_list = model(frames)  # [[B,512],[B,512]]
        # print(len(frames_embedding_list))
        # end_embedding = end_model(end_frames)
        loss = 0
        for i in range(T):
            frames_embedding = frames_embedding_list[i]
            label = labels[:, i]
            sim_logits = torch.nn.functional.cosine_similarity(frames_embedding.unsqueeze(1),
                                                               test_text_tensor.unsqueeze(0), dim=2)  # 256 666
            sim_logits = sim_logits / 0.1
            sim_logits_softmax = torch.nn.functional.softmax(sim_logits, dim=1)
            # print(sim_logits_softmax)
            gt_indices = label
            pred_indices = torch.argmax(sim_logits_softmax, dim=1)
            set_pred.add(pred_indices[0].tolist())
            set_true.add(gt_indices[0].tolist())

            correct_predictions = torch.eq(pred_indices, gt_indices)
            num_correct = torch.sum(correct_predictions).item()
            if num_correct == 1:
                correct_f += 1
            novel_acc_num += num_correct
            novel_total_num += label.shape[0]
        if correct_f == T:
            novel_sr_num += 1
        iou = 100.0 * len(set_pred.intersection(set_true)) / len(set_pred.union(set_true))
        novel_miou += iou

    print("novel_acc: ", novel_acc_num / novel_total_num)
    print("novel_sr: ", novel_sr_num / len(test_novel_dataset))
    print("novel_miou: ", novel_miou / len(test_novel_dataset))
    with open(os.path.join(output_dir, 'output.txt'), 'a') as f:
        print("novel_acc: ", novel_acc_num / novel_total_num,file=f)
        print("\n", file=f)
        print("novel_sr: ", novel_sr_num / len(test_novel_dataset),file=f)
        print("\n", file=f)
        print("novel_miou: ", novel_miou / len(test_novel_dataset),file=f)
        print("\n", file=f)









