
import torch
import os
import json
import numpy as np
class Video(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 split,
                 feat,
                 is_val=0):
        self.root = root
        self.split = split
        self.is_val = is_val
        self.feat = feat
        self.M = 3
        if self.feat == 's3d':
            self.zeros_frame = torch.zeros(512)
        elif self.feat == 'videoclip':
            self.zeros_frame = torch.zeros(768)
        else:
            self.zeros_frame = torch.zeros(512)


        #读取json文件
        if self.is_val == 0:#train
            video_json = 'train_train_base_dataset_' + str(self.split) + '.json'
        elif self.is_val == 1:#test_novel
            video_json = 'novel_dataset_' + str(self.split) + '.json'
        elif self.is_val == 2:#test_base
            video_json = 'test_base_dataset_' + str(self.split) + '.json'
        elif self.is_val == 3:#val
            video_json = 'train_val_base_dataset_' + str(self.split) + '.json'
        video_json_dir = os.path.join(self.root, video_json)
        if os.path.exists(video_json_dir):
            with open(video_json_dir, 'r') as f:
                self.json_data = json.load(f)
            print('Loaded {}'.format(video_json_dir))
        else:
            print("no json data")


    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        video_id = self.json_data[index]

        if self.feat == 's3d':
            if video_id['dataset'] == 'COIN':
                name = video_id['task_name'] + '_' + str(video_id['task_id_old']) + '_' + video_id['vid'] + '.npy'
                images_d = np.load(os.path.join('/data0/wuyilu/data/COIN/full_npy', name), encoding='bytes', allow_pickle=True)  #T*512
                images = images_d['frames_features']
            else:
                name = video_id['task_id_old']+'_'+video_id['vid'] + '.npy'
                images_d = np.load(os.path.join('/data0/wuyilu/data/ori_processed_data', name), encoding='bytes',
                                   allow_pickle=True)
                images = images_d['frames_features']
        elif self.feat == 'videoclip':
            name = video_id['dataset'] + '_' + video_id['vid'] + '.npy'
            images = np.load(os.path.join('/data0/wuyilu/data/OEPP_videoclip', name))
        # print(images.shape)

        start_frames_list = []
        end_frames_list = []
        action_list = []

        for step in video_id['anno']:
            start_frames = []
            end_frames = []
            segment = step['segment']
            action = step['action']
            # print(action)
            action_list.append(str(action))
            start = int(segment[0])
            end = int(segment[1])
            if end >= images.shape[0]:
                end = images.shape[0]-1
            for i in range(self.M):
                if start+i >= images.shape[0]:#
                    start_frames.extend(self.zeros_frame)
                else:
                    start_frames.extend(images[start+i])
                if end-self.M+1+i < 0:
                    end_frames.extend(self.zeros_frame)
                else:
                    end_frames.extend(images[end-self.M+1+i])
            # print(len(start_frames))
            start_frames = torch.tensor(start_frames)
            end_frames = torch.tensor(end_frames)
            start_frames_list.append(start_frames)
            end_frames_list.append(end_frames)

        return video_id['vid'], start_frames_list, end_frames_list, action_list


class Seq_action(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 split,
                 feat,
                 T,#T,action动作的个数
                 is_pad,#动作个数较少的序列是否需要补充动作
                 is_total,#是否是total的动作池子
                 is_val#训练集/测试集/验证集
                 ):
        self.root = root
        self.split = split
        self.feat = feat
        self.T = T
        self.is_pad = is_pad
        self.is_total = is_total
        self.is_val = is_val
        self.videos = Video(root=self.root, split=self.split, feat= self.feat,is_val=self.is_val)
        self.seq_list = []
        print("len of videos: ", len(self.videos))
        if self.feat == 'videoclip':
            with open('/data0/wuyilu/data/OEPP_videoclip/action_feat_dict.json') as f:
                self.actions_text_dict = json.load(f)
        elif self.feat == 's3d':
            with open('/data0/wuyilu/otpp/s3d/action_feat_dict.json') as f:
                self.actions_text_dict = json.load(f)

        for video in self.videos:
            vid, start_frames_list, end_frames_list , action_list = video
            length = len(action_list)
            # print(length)
            if self.T <= length:
                for i in range(length-self.T+1):
                    # print(i)
                    start_id = i
                    end_id = i + self.T - 1
                    start_f = start_frames_list[start_id]
                    end_f = end_frames_list[end_id]
                    actions = action_list[start_id:end_id+1]
                    self.seq_list.append({'vid':vid, 'start_frames':start_f,'end_frames':end_f,'actions':actions})
            else:
                if self.is_pad == 1:
                    start_f = start_frames_list[0]
                    end_f = end_frames_list[-1]
                    pad_action_list = []
                    for i in range(self.T-length):
                        pad_action_list.append(action_list[0])
                    actions = pad_action_list+action_list
                    self.seq_list.append({'vid':vid, 'start_frames': start_f, 'end_frames': end_f, 'actions': actions})
        print("total sequences length:",len(self.seq_list))

    def get_labels(self,actions):
        if self.is_val == 0 or self.is_val == 2 or self.is_val == 3:  # 训练集和验证集是train_action_pool
            with open('/data0/wuyilu/otpp/data_p_1114/base_action_pool_' + str(self.split) + '.json') as f:
                self.action_pool = json.load(f)
        else:
            with open('/data0/wuyilu/otpp/data_p_1114/novel_action_pool_' + str(self.split) + '.json') as f:
                self.action_pool = json.load(f)
        if self.is_total == 1:
            with open('/data0/wuyilu/otpp/data_p_1114/total_action_pool.json') as f:
                self.action_pool = json.load(f)
        l = torch.zeros(self.T,dtype=int)
        for index in range(len(actions)):
            for i in range(len(self.action_pool)):
                if actions[index] == self.action_pool[i]:
                    l[index] = torch.tensor(i)
        return l
    def get_action_tensor(self,action_list):
        tensor_list = []
        for action in action_list:
            text_embedding = torch.tensor(self.actions_text_dict[action])
            tensor_list.append(text_embedding)
        text_tensor = torch.cat([tensor for tensor in tensor_list], dim=0)
        return text_tensor
    def __len__(self):
        return len(self.seq_list)
    def __getitem__(self, index):
        seq = self.seq_list[index]
        vid = seq['vid']
        start_frames = seq['start_frames']
        end_frames = seq['end_frames']
        start_end_frames = torch.cat((start_frames,end_frames),dim=0)
        action_list = seq['actions']
        action_tensor = self.get_action_tensor((action_list))
        mid_action_list = action_list[1:-1]
        labels = self.get_labels(action_list)
        mid_labels = labels[1:-1]
        return vid, start_frames,end_frames,start_end_frames,action_list,action_tensor,mid_action_list,labels,mid_labels



if __name__ == '__main__':
    train_dataset = Seq_action(root='/data0/wuyilu/otpp/data_p_1114', split=1,T=6, feat='videoclip',is_pad=0, is_total=0, is_val=0)
