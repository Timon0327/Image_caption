import os
import json
from tqdm import tqdm
import numpy as np



def data_split(token_path, tti_path, train_set_path, val_set_path):
    val_set = []
    train_set = []
    test_set = []
    with open(token_path) as t:
        token = json.load(t)
        all_token = token['images']
    with open(tti_path) as int:
        coco = json.load(int)
        for i in tqdm(range(len(all_token))):
            # img_id = all_token[i]['cocoid']
            split = all_token[i]['filepath']
            if split == 'val2014':
                # img_id = all_token[i]['filename'].split('.')[0].split('COCO_val2014_000000')[1]
                img_id = all_token[i]['cocoid']
                for num in range(len(all_token[i]['sentences'])):
                    idx = [img_id, num]
                    val_set.append(idx)

            elif split == 'train2014':
                # img_id = all_token[i]['filename'].split('.')[0].split('COCO_train2014_000000')[1]
                img_id = all_token[i]['cocoid']
                # for s in range(len(img_id)):
                #     if img_id[s] != '0':
                #         b = s
                #         break
                # img_id = img_id[b:]
                for num_1 in range(len(all_token[i]['sentences'])):
                    idx = [img_id, num_1]
                    train_set.append(idx)
        return train_set, val_set

    # np.save(train_set_path, train_set)
    # np.save(val_set_path, val_set)

def token_to_int(token_path, save_ref_path):
    with open(token_path) as t:
        token = json.load(t)
        all_token = token['images']
        # with open(tti_path) as int:
        #     coco = json.load(int)
        length = []
        for i in tqdm(range(len(all_token))):
            sentence_list = {}
            ref_dict = {}
            img_id = all_token[i]['cocoid']
            for j in range(len(all_token[i]['sentences'])):
                sentence = all_token[i]['sentences'][j]['tokens']
                sentence_list[j] = sentence
                #     comput max len
                # length.append(len(sentence))
        # num = max(length)
        # print(num)
            ref_dict = {'img_id': img_id, 'sentence': sentence_list}
        return ref_dict
        # print('a')
        #     np.save(save_ref_path + str(img_id) + '.npy', ref_dict)


def frequency_word(token_path, word_dict_path):
    with open(token_path) as f:
        token_data = json.load(f)
    with open(word_dict_path) as d:
        word_dict = json.load(d)
        dict = word_dict['ix_to_word']
        # dict = torch.tensor(dict)
        num = np.zeros(9487)
        all_token = token_data['images']
        for file in tqdm(all_token):
            all_sentence = file['sentences']
            for token in all_sentence:
                all_word = token['tokens']
                # print('a')
                for word in all_word:
                    for i in range(len(dict)):
                        if word == dict[str(i+1)]:
                            # print(word)
                            num[i] += 1
        return num
    
   

def s_u_split(train_path, val_path, word_data_path, coco_ref):
    train_data = np.load(train_path)
    val_data = np.load(val_path)
    # with open(word_data_path) as f:
    #     word_dataset = json.load(f)
    #     all_token = word_dataset['images']
    ref_list = os.listdir(coco_ref)
    # train_sentences = []
    # val_sentences = []
    share_word = []
    train_word = []
    val_word = []
    # for file in tqdm(train_data):
    #     train_name = str(file[0]) + '.npy'
    #     data_tra = np.load(os.path.join(coco_ref, train_name), allow_pickle=True).item()
    #     sentence = data_tra['sentence']
    #     train_sentences.append(sentence)
    #     # np.save('')
    # for file in tqdm(val_data):
    #     val_name = str(file[0]) + '.npy'
    #     data_val = np.load(os.path.join(coco_ref, val_name), allow_pickle=True).item()
    #     sentence = data_val['sentence']
    #     val_sentences.append(sentence)
    #
    # np.save('/mnt/Peng/Projects/image caption/datasets/coco/data/word_split/train_sentences.npy', train_sentences)
    # np.save('/mnt/Peng/Projects/image caption/datasets/coco/data/word_split/val_sentences.npy', val_sentences)

    train_sentences_path = '/mnt/Peng/Projects/image caption/datasets/coco/data/word_split/train_sentences.npy'
    val_sentences_path = '/mnt/Peng/Projects/image caption/datasets/coco/data/word_split/val_sentences.npy'
    train_sentences = list(np.load(train_sentences_path, allow_pickle=True))
    val_sentences = list(np.load(val_sentences_path, allow_pickle=True))


    for s in tqdm(train_sentences):
        for i in range(len(s)):
            for word in s[i]:

        # for s_1 in s:
        #     for word in s_1:
                for s_val in val_sentences:
                    for j in range(len(s_val)):
                        for word_val in s_val[j]:
                            if word == word_val:
                                share_word.append(word)

    for s in tqdm(train_sentences):
        for n in range(len(s)):
            for word in s[n]:
                for share in share_word:
                    if word != share:
                        train_word.append(word)
    for s in tqdm(val_sentences):
        for m in range(len(s)):
            for word in s[m]:
                for share in share_word:
                    if word != share:
                        val_word.append(word)
    print('Done!!!')

    return  share_word, train_word, val_word





if __name__ == '__main__':
    token_path = '/mnt/Peng/Projects/image caption/datasets/coco/dataset.json'
    tti_path = '/mnt/Peng/Projects/image caption/datasets/coco/cocotalk.json'
    train_set_path = '/mnt/Peng/Projects/image caption/datasets/coco/data/data_split/val2014_set.npy'
    val_set_path = '/mnt/Peng/Projects/image caption/datasets/coco/data/data_split/train2014_set.npy'
    # data_split = data_split(token_path, tti_path, train_set_path, val_set_path)
    save_ref_path = '/mnt/Peng/Projects/image caption/datasets/coco/data/coco_ref/'
    # save_ref = token_to_int(token_path, save_ref_path)
    num = frequency_word(token_path=token_path, word_dict_path=tti_path)
    np.save('/mnt/Peng/Projects/image caption/datasets/coco/word_freq.npy', num)
    
    
    share_word, train_word, val_word = s_u_split(train_path=train_set_path, val_path=val_set_path, word_data_path=token_path, coco_ref=save_ref_path)
    np.save('/mnt/Peng/Projects/image caption/datasets/coco/data/word_split/share_word.npy', share_word)
    np.save('/mnt/Peng/Projects/image caption/datasets/coco/data/word_split/train_word.npy', train_word)
    np.save('/mnt/Peng/Projects/image caption/datasets/coco/data/word_split/val_word.npy', val_word)
    
