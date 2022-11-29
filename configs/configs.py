# 1. data
dataset_type = 'HSI Data Sets'
# 220 145 145 16
# data_root = '/home/students/master/2022/liuy/PyCharm-Remote/learn/HSIC_student_Liuyi/datasets/Indian_pines.mat'
# image_name = 'indian_pines_corrected'
# gt_name = 'R'

# 103*610*340 9
# data_root = '/home/students/master/2022/liuy/PyCharm-Remote/learn/Dataset/PaviaU.mat'
# image_name = 'paviaU'
# gt_name = 'paviaU_gt'

# 176 512 614 13
data_root = '/home/students/master/2022/liuy/PyCharm-Remote/learn/Dataset/KSC.mat'
image_name = 'KSC'
gt_name = 'KSC_gt'

# 144 349 1905 15
# data_root = '../datasets/HoustonU.mat'
# image_name = 'HoustonU'
# gt_name = 'HoustonU_GT'

# Black_River 135,684,453 9
# data_root = '/home/students/master/2022/liuy/PyCharm-Remote/learn/HSIC_student_Liuyi/datasets/blackriver.mat'
# image_name = 'dc'
# gt_name = 'R'

epoch = 200
train_set_num = 150
patch_size = 25
pca_num = 5
phase = ['train', 'test', 'no_gt']

num_workers = 12

data = dict(
    data_path=data_root,
    image_name=image_name,
    gt_name=gt_name,
    train_set_num=train_set_num,
    patch_size=patch_size,
    pca=pca_num,
    train_data=dict(
        phase=phase[0]
    ),
    test_data=dict(
        phase=phase[1]
    ),
    no_gt_data=dict(
        phase=phase[2]
    )
)

# 2. model

model = dict(
    inCDCN_fea_num=200,
    inPMN_fea_num=5,
    out_fea_num=13
)

# 3. train

lr = 1e-3

train = dict(
    optimizer=dict(
        typename='RMSprop',
        lr=lr,
        momentum=0.9,
        weight_decay=1e-5
    ),
    train_model=dict(
        gpu_train=True,
        gpu_num=1,
        workers_num=num_workers,
        epoch=epoch,
        batch_size=128,
        lr=lr,
        lr_adjust=True,
        lr_gamma=0.1,
        lr_step=100,
        save_folder='./weights/',
        save_name='model_CNN2D',
        reuse_model=False,
        reuse_file='./weights/model_CNN2D_Final.pth'
    )
)

test = dict(
    batch_size=2048,
    gpu_train=True,
    gpu_num=1,
    workers_num=num_workers,
    model_weights='./weights/model_CNN2D_Final.pth',
    save_folder='./result'
)
