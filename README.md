# ST-CGAN
Stacked conditional GAN for reflection removal

Original Thesis:

Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal

Jifeng Wang Xiang Li Jian Yang

https://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Stacked_Conditional_Generative_CVPR_2018_paper.pdf

Try to trans it for reflection removal

Requirements:(All network reimplements are same of similar)

* 1.Pytorch 1.3.0
* 2.Torchvision 0.2.0
* 3.Python 3.6.10
* 4.glob
(Dataset)
* 5.PIL
* 6.tqdm(For training)
* 7.Opencv-Python
* 8.tensorboardX
* 9.pip install resnest --pre

Dataset Modified:

Line 25,26,27

imgpath='/public/zebanghe2/joint/train/mix'

transpath='/public/zebanghe2/joint/train/transmission'

maskpath='/public/zebanghe2/joint/train/sodmask'

Train

python train.py

Epoch Number:Line 72

Batch Size:Line 67

Test
python test.py

If any problem, please ask in issue.

Jerry He
