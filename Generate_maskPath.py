import os
'''
生成伪语义信息的路径文件
'''

read_file = '../1sw/data/train_augvoc.txt'
save_file = '../1sw/data/train_gen_v14_3.txt'
mask_path = '/home/lhw/sdb/xx/WSSS/1sw/output/IRN/sem_seg/v1209/CAM_CASA_WGAP_tf_v14_3_ws3'
result = []
img_gt_name_list = open(read_file).read().splitlines()
with open(save_file, 'w') as f:
    for path in img_gt_name_list:
        path = path.split(" ")[0]
        f.write(path)
        filename = path.split('/')[-1].split('.')[0]
        f.write(' ')
        f.write(os.path.join(mask_path, filename+'.png'))
        f.write('\n')