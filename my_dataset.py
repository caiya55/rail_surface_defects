from torch.utils.data import Dataset
import torch
import numpy as np
import os
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
import random
from random import choice

# 随机对比度和亮度 (概率：0.5)
def random_bright(img, p=0.5, lower=0.8, upper=1.2):
    if random.random() < p:
        mean = np.mean(img)
        img = img - mean
        img = img * random.uniform(lower, upper) + mean * random.uniform(lower, upper)  # 亮度
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
    return img

def find_bbox(mask):
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    stats = stats[stats[:,4].argsort()]
    return stats[:-1]

def random_translate(img, mask, p=0.5):
    # 随机平移
    if random.random() < p:
        h_img, w_img, _ = img.shape
        bboxes = find_bbox(mask)
        # 得到可以包含所有bbox的最大bbox
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w_img - max_bbox[2]
        max_d_trans = h_img - max_bbox[3]
        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))
 
        M = np.array([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (w_img, h_img))
        mask = cv2.warpAffine(mask, M, (w_img, h_img))
        mask = mask[:,:,None]
    return img, mask

class DefectDataset(Dataset):
    def __init__(self, data_pathes, istrain=True, debug=False):
        '''data path is ../RSDDs dataset/Type-I RSDDs dataset'''
        neg_paths = np.load('type1_negative_lists.npy')
        pos_paths = np.load('type1_positive_lists.npy')

        if not os.path.exists('neg_idx_restore.npy'):              
            neg_idxs = np.random.permutation(len(neg_paths))
            pos_idxs = np.random.permutation(len(pos_paths))
            np.save('neg_idxs.npy', neg_idxs)
            np.save('pos_idxs.npy', pos_idxs)
        else:
            neg_idxs = np.load('neg_idx_restore.npy')
            pos_idxs = np.load('pos_idxs.npy')
        neg_s = int(len(neg_idxs)*0.8)
        pos_s = int(len(pos_idxs)*0.8)
        self.istrain = istrain
        if istrain:
          train_pos = [os.path.join(data_pathes, 'crop_images', pos_paths[a]) for a in pos_idxs[:pos_s]]
          train_neg = [os.path.join(data_pathes, 'crop_images', neg_paths[a]) for a in neg_idxs[:neg_s]]
          
          label_pos = [os.path.join(data_pathes, 'crop_groundtruth', pos_paths[a]) for a in pos_idxs[:pos_s]]
          label_neg = [os.path.join(data_pathes, 'crop_groundtruth', neg_paths[a]) for a in neg_idxs[:pos_s]]
          data_lists = train_neg + train_pos
          label_lists = label_neg + label_pos
        else:
          test_pos = [os.path.join(data_pathes, 'crop_images', pos_paths[a]) for a in pos_idxs[pos_s:]]
          test_neg = [os.path.join(data_pathes, 'crop_images', neg_paths[a]) for a in neg_idxs[neg_s:]]
          
          label_pos = [os.path.join(data_pathes, 'crop_groundtruth', pos_paths[a]) for a in pos_idxs[pos_s:]]
          label_neg = [os.path.join(data_pathes, 'crop_groundtruth', neg_paths[a]) for a in neg_idxs[neg_s:]]
          data_lists = test_pos + test_neg
          label_lists = label_pos + label_neg
          
        self.len = len(data_lists)
        self.data = data_lists
        self.labels = label_lists
        self.debug = debug
        # 相关预处理的初始化
        '''class torchvision.transforms.ToTensor'''
        # 把shape=(H,W,C)的像素值范围为[0, 255]的PIL.Image或者numpy.ndarray数据
        # 转换成shape=(C,H,W)的像素数据，并且被归一化到[0.0, 1.0]的torch.FloatTensor类型。

        # self.normalize=transforms.Normalize()
       
        
    def __getitem__(self, index):

        image_name =  self.data[index]
        label_name = self.labels[index]
        img = cv2.imread(image_name,0)
        label = cv2.imread(label_name,0)       

        
        '''crop the image'''
        img = cv2.resize(img, (224,224))
        img = img[:,:,None]
        
        label = 255*(label>200).astype(np.uint8)
        
        label = cv2.resize(label, (224,224),interpolation=cv2.INTER_NEAREST)
        # label = 255*(label>0.5)
        label = label[:,:,None]

        '''data augmentation'''
        # randomly flip image and labels
        # left right
        
        if self.istrain == True and np.random.uniform() > 0.5:
            img[:] = img[:, ::-1, :]
            label[:] = label[:, ::-1, :]
        # up down
        if self.istrain == True and np.random.uniform() > 0.5:
            img[:] = img[::-1, :, :]
            label[:] = label[::-1, :, :]
        if self.istrain == True:
            img = random_bright(img)
            
            # img, label = random_translate(img, label[:,:,0])
            
        label = label / 255.0
        label = label.astype(np.float32)
        
        img = img / 255.0
        img = img.astype(np.float32)
        img = img.transpose((2,0,1))
        label = label.transpose((2,0,1))
        
        if not self.debug:
            return img, label
        else:
            return img, label, label_name
 
    def __len__(self):
        return self.len

class DefectDataset2(Dataset):
    def __init__(self, data_pathes, istrain=True, debug=False):
        '''data path is ../RSDDs dataset/Type-II RSDDs dataset'''
        neg_paths = np.load('type2_negative_lists.npy')
        pos_paths = np.load('type2_positive_lists.npy')

        if not os.path.exists('t2_neg_idxs.npy'):              
            neg_idxs = np.random.permutation(len(neg_paths))
            pos_idxs = np.random.permutation(len(pos_paths))
            np.save('t2_neg_idxs.npy', neg_idxs)
            np.save('t2_pos_idxs.npy', pos_idxs)
        else:
            neg_idxs = np.load('t2_neg_idxs.npy')
            pos_idxs = np.load('t2_pos_idxs.npy')
        neg_s = int(len(neg_idxs)*0.8)
        pos_s = int(len(pos_idxs)*0.8)
        self.istrain = istrain
        if istrain:
          train_pos = [os.path.join(data_pathes, 'crop_images', pos_paths[a]) for a in pos_idxs[:pos_s]]
          train_neg = [os.path.join(data_pathes, 'crop_images', neg_paths[a]) for a in neg_idxs[:neg_s]]
          
          label_pos = [os.path.join(data_pathes, 'crop_groundtruth', pos_paths[a]) for a in pos_idxs[:pos_s]]
          label_neg = [os.path.join(data_pathes, 'crop_groundtruth', neg_paths[a]) for a in neg_idxs[:pos_s]]
          data_lists = train_neg #+ train_pos
          label_lists = label_neg #+ label_pos
        else:
          test_pos = [os.path.join(data_pathes, 'crop_images', pos_paths[a]) for a in pos_idxs[pos_s:]]
          test_neg = [os.path.join(data_pathes, 'crop_images', neg_paths[a]) for a in neg_idxs[neg_s:]]
          
          label_pos = [os.path.join(data_pathes, 'crop_groundtruth', pos_paths[a]) for a in pos_idxs[pos_s:]]
          label_neg = [os.path.join(data_pathes, 'crop_groundtruth', neg_paths[a]) for a in neg_idxs[neg_s:]]
          data_lists = test_pos + test_neg
          label_lists = label_pos + label_neg
          
        self.len = len(data_lists)
        self.data = data_lists
        self.labels = label_lists
        self.debug = debug
        # 相关预处理的初始化
        '''class torchvision.transforms.ToTensor'''
        # 把shape=(H,W,C)的像素值范围为[0, 255]的PIL.Image或者numpy.ndarray数据
        # 转换成shape=(C,H,W)的像素数据，并且被归一化到[0.0, 1.0]的torch.FloatTensor类型。

        # self.normalize=transforms.Normalize()
 
    def __getitem__(self, index):

        image_name =  self.data[index]
        label_name = self.labels[index]
        img = cv2.imread(image_name,0)
        label = cv2.imread(label_name,0)       

        '''crop the image'''
        img = cv2.resize(img, (224,224))
        img = img[:,:,None]
        
        label = 255*(label>200).astype(np.uint8)
        
        label = cv2.resize(label, (224,224),interpolation=cv2.INTER_NEAREST)
        # label = 255*(label>0.5)
        label = label[:,:,None]

        '''data augmentation'''
        # randomly flip image and labels
        # left right
        if self.istrain == True and np.random.uniform() > 0.5:
            img[:] = img[:, ::-1, :]
            label[:] = label[:, ::-1, :]
        # up down
        if self.istrain == True and np.random.uniform() > 0.5:
            img[:] = img[::-1, :, :]
            label[:] = label[::-1, :, :]
        if self.istrain == True:
            img = random_bright(img)
            # img, label = random_translate(img, label[:,:,0])
            
        label = label / 255.0
        label = label.astype(np.float32)
        
        img = img / 255.0
        img = img.astype(np.float32)
        img = img.transpose((2,0,1))
        label = label.transpose((2,0,1))
        
        if not self.debug:
            return img, label
        else:
            return img, label, label_name
 
    def __len__(self):
        return self.len

def hist_match(source, template): 
    """ 
    Adjust the pixel values of a grayscale image such that its histogram 
    matches that of a target image 

    Arguments: 
    ----------- 
     source: np.ndarray 
      Image to transform; the histogram is computed over the flattened 
      array 
     template: np.ndarray 
      Template image; can have different dimensions to source 
    Returns: 
    ----------- 
     matched: np.ndarray 
      The transformed output image 
    """ 

    oldshape = source.shape 
    source = source.ravel() 
    template = template.ravel() 

    # get the set of unique pixel values and their corresponding indices and 
    # counts 
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, 
              return_counts=True) 
    t_values, t_counts = np.unique(template, return_counts=True) 

    # take the cumsum of the counts and normalize by the number of pixels to 
    # get the empirical cumulative distribution functions for the source and 
    # template images (maps pixel value --> quantile) 
    s_quantiles = np.cumsum(s_counts).astype(np.float64) 
    s_quantiles /= s_quantiles[-1] 
    t_quantiles = np.cumsum(t_counts).astype(np.float64) 
    t_quantiles /= t_quantiles[-1] 

   # interpolate linearly to find the pixel values in the template image 
    # that correspond most closely to the quantiles in the source image 
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values) 

    return interp_t_values[bin_idx].reshape(oldshape) 


def ecdf(x):
    """convenience function for computing the empirical CDF"""
    vals, counts = np.unique(x, return_counts=True)
    ecdf = np.cumsum(counts).astype(np.float64)
    ecdf /= ecdf[-1]
    return vals, ecdf

class joint_dataset(Dataset):
    def __init__(self, data_path):
        '''data path is ../RSDDs dataset/Type-I RSDDs dataset'''
        data_paths = os.listdir(data_path)
        self.paths = [ os.path.join(data_path, pa) for pa in data_paths]
        self.len = len(data_paths)
        
        '''data path is ../RSDDs dataset/Type-II RSDDs dataset'''
        neg_paths = np.load('type2_negative_lists.npy')
        pos_paths = np.load('type2_positive_lists.npy')
        type_pathes = '../RSDDs dataset/Type-II RSDDs dataset'
        
        if not os.path.exists('t2_neg_idxs.npy'):              
            neg_idxs = np.random.permutation(len(neg_paths))
            pos_idxs = np.random.permutation(len(pos_paths))
            np.save('t2_neg_idxs.npy', neg_idxs)
            np.save('t2_pos_idxs.npy', pos_idxs)
        else:
            neg_idxs = np.load('t2_neg_idxs.npy')
            pos_idxs = np.load('t2_pos_idxs.npy')
        neg_s = int(len(neg_idxs)*0.8)
        pos_s = int(len(pos_idxs)*0.8)
        
        # self.istrain = istrain
        train_neg = [os.path.join(type_pathes, 'crop_images', neg_paths[a]) for a in neg_idxs[:neg_s]]
        label_neg = [os.path.join(type_pathes, 'crop_groundtruth', neg_paths[a]) for a in neg_idxs[:pos_s]]
        data_lists = train_neg 
        label_lists = label_neg
        
        
    def __getitem__(self, index):
        path = self.paths[index]
        image = cv2.imread(path, 0)
        image = image[:,:,None]
        image = image / 255.0
        image = image.astype(np.float32)
        image = image.transpose((2,0,1))
        return image, path
    def __len__(self):
        return self.len
    
class joint_train_dataset_mix(Dataset):
    def __init__(self, data_path):
        '''data path is ../RSDDs dataset/Type-I RSDDs dataset'''
        data_paths = os.listdir(data_path)
        self.paths = [ os.path.join(data_path, pa) for pa in data_paths if pa.endswith('pnggt.png')]

        '''type II template '''
        template_pathes = '../RSDDs dataset/Type-II RSDDs dataset/crop_images'
        data_paths = os.listdir(template_pathes)
        paths = [ os.path.join(template_pathes, pa) for pa in data_paths]
        self.templat_image = cv2.imread(paths[0], 0)

        neg_paths = np.load('type2_negative_lists.npy')
        pos_paths = np.load('type2_positive_lists.npy')
        type_pathes = '../RSDDs dataset/Type-II RSDDs dataset'
        
        if not os.path.exists('t2_neg_idxs.npy'):              
            neg_idxs = np.random.permutation(len(neg_paths))
            pos_idxs = np.random.permutation(len(pos_paths))
            np.save('t2_neg_idxs.npy', neg_idxs)
            np.save('t2_pos_idxs.npy', pos_idxs)
        else:
            neg_idxs = np.load('t2_neg_idxs.npy')
            pos_idxs = np.load('t2_pos_idxs.npy')
        neg_s = int(len(neg_idxs)*0.8)
        pos_s = int(len(pos_idxs)*0.8)
        
        # self.istrain = istrain
        train_neg = [os.path.join(type_pathes, 'crop_images', neg_paths[a]) for a in neg_idxs[:neg_s]]
        label_neg = [os.path.join(type_pathes, 'crop_groundtruth', neg_paths[a]) for a in neg_idxs[:pos_s]]
        data_lists = train_neg 
        label_lists = label_neg
        self.labels = label_lists
        self.paths = self.paths+data_lists
        self.len = len(self.paths)
        
    def __getitem__(self, index):
        path = self.paths[index]
        if index < 249:
            raw_image = cv2.imread(path, 0)
            '''histogram matching'''
            image = hist_match(raw_image, self.templat_image)
            
            image = image[:,:,None]
            image = image / 255.0
            image = image.astype(np.float32)
            image = image.transpose((2,0,1))
            
            label_name = path[:-10]+'.png'
            label = cv2.imread(label_name,0)  
            label = label[:,:,None]
            label = label / 255.0
            label = label.astype(np.float32)
            label = label.transpose((2,0,1))    
            
        else:
            img = cv2.imread(path, 0)
            label_name = self.labels[index-249]
            label = cv2.imread(label_name,0)       
    
            '''crop the image'''
            img = cv2.resize(img, (224,224))
            img = img[:,:,None]
            label = 255*(label>200).astype(np.uint8)
        
            label = cv2.resize(label, (224,224),interpolation=cv2.INTER_NEAREST)
            # label = 255*(label>0.5)
            label = label[:,:,None]
            
            img = img / 255.0
            img = img.astype(np.float32)
            image = img.transpose((2,0,1))
            label = label.transpose((2,0,1))
        return image, label, path,label_name
    def __len__(self):
        return self.len

class joint_train_dataset(Dataset):
    def __init__(self, data_path):
        '''data path is ../RSDDs dataset/Type-I RSDDs dataset'''
        data_paths = os.listdir(data_path)
        neg_idxs = np.load('joint_neg.npy')
        paths = [ os.path.join(data_path, pa) for pa in data_paths if pa.endswith('pnggt.png')]
        self.paths = [paths[neg_idxs[i]] for i in range(len(neg_idxs))]
        # '''type II template '''
        # template_pathes = '../RSDDs dataset/Type-II RSDDs dataset/crop_images'
        # data_paths = os.listdir(template_pathes)
        # paths = [ os.path.join(template_pathes, pa) for pa in data_paths]
        # self.templat_image = cv2.imread(paths[0], 0)

        '''type I template '''
        template_pathes = '../RSDDs dataset/Type-I RSDDs dataset/crop_images'
        data_paths = os.listdir(template_pathes)
        paths = [ os.path.join(template_pathes, pa) for pa in data_paths]
        self.templat_image = cv2.imread(paths[0], 0)


        self.len = len(self.paths)
        
    def __getitem__(self, index):
        path = self.paths[index]
        raw_image = cv2.imread(path, 0)
        '''histogram matching'''
        image = raw_image
        # image = hist_match(raw_image, self.templat_image)
        # image = raw_image
        image = image[:,:,None]
        label_name = path[:-10]+'.png'
        label = cv2.imread(label_name,0)  
        label = label[:,:,None]
        '''data augmentation'''
        # randomly flip image and labels
        # left right
        
        # if np.random.uniform() > 0.5:
        #     image[:] = image[:, ::-1, :]
        #     label[:] = label[:, ::-1, :]
        # # up down
        # if np.random.uniform() > 0.5:
        #     image[:] = image[::-1, :, :]
        #     label[:] = label[::-1, :, :]
 
        # image = random_bright(image)
        

        image = image / 255.0
        image = image.astype(np.float32)
        image = image.transpose((2,0,1))
        
        label = label / 255.0
        label = label.astype(np.float32)
        label = label.transpose((2,0,1))    

        return image, label, path
    def __len__(self):
        return self.len
    
if __name__=='__main__':
    print('sss')
    # data_pathes = '../RSDDs dataset/Type-I RSDDs dataset'

    # train = DefectDataset(data_pathes, istrain=True, debug=True) #训练数据集
    data_path = '../results_joints'
    join_train = joint_train_dataset_mix(data_path) #训练数据集
    trainloader = torch.utils.data.DataLoader(join_train, batch_size=1, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
    for i, data in enumerate(trainloader, 0):
        length = len(trainloader)
        inputs, label, name, label_name = data
        inputss = (inputs.squeeze().cpu().numpy())
        inputss = (inputss*255).astype(np.uint8)
        name1  = name[0].split('\\')[-1][:-10] + '.png'
        out_path = os.path.join('../results_domain',name1)
        # cv2.imwrite(out_path, inputss)
        break
    # '''type II'''
    # data_pathes = '../RSDDs dataset/Type-II RSDDs dataset'

    # train = DefectDataset2(data_pathes, istrain=True, debug=True) #训练数据集
    # data_path = '../results_joints'

    # trainloader = torch.utils.data.DataLoader(train, batch_size=5, shuffle=True, num_workers=2)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取
    # for i, data in enumerate(trainloader, 0):
    #     length = len(trainloader)
    #     inputs, label, name = data
    #     break
    # '''type II template '''
    # template_pathes = '../RSDDs dataset/Type-II RSDDs dataset/crop_images'
    # data_paths = os.listdir(template_pathes)
    # paths = [ os.path.join(template_pathes, pa) for pa in data_paths]
    # templat_image = cv2.imread(paths[0], 0)
    # plt.imshow(templat_image)
    
    # source_pathes = '../results_joints'
    # data_paths = os.listdir(source_pathes)
    # paths = [ os.path.join(source_pathes, pa) for pa in data_paths if pa.endswith('pnggt.png')]
    # path = paths[0]

    # source_image = cv2.imread(path, 0)
    # plt.imshow(source_image)   
    
    # matched = hist_match(source_image, templat_image)
    # plt.imshow(matched)
    # plt.show()
    
    # x1, y1 = ecdf(source_image.ravel())
    # x2, y2 = ecdf(templat_image.ravel())
    # x3, y3 = ecdf(matched.ravel())

    # colors = ['darkorange', 'brown', 'slategray', 'darkslategray', 'indigo']
    # fig=plt.figure()
    # plt.plot(x1, y1 * 100, '-', color=colors[0], lw=3, label='Source')
    # plt.plot(x2, y2 * 100, '-k', color=colors[1],lw=3, label='Template')
    # plt.plot(x3, y3 * 100, '--r', color=colors[2], lw=3, label='Matched')   
    # plt.xlabel('Pixel value')
    # plt.ylabel('Cumulative %')
    # plt.grid('True')
    # plt.legend(loc=5)
    # plt.savefig('matching figure.png', dpi=400)
    # plt.show()
    
    # cv2.imwrite('matching_exm.png', matched)
    
    
    
    
    