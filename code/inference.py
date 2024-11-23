import os
import numpy as np
import torch
from eval_metrics import get_similarity_metric
from dataset import create_Kamitani_dataset, create_BOLD5000_dataset
from dc_ldm.ldm_for_fmri import fLDM
from einops import rearrange
from PIL import Image
import torchvision.transforms as transforms
import json


def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))


def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')


def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0  # to -1 ~ 1
    return img


def get_eval_metric(samples, avg=True, subject_name="default_subject", results_path="../results"):
    metric_list = ['mse', 'pcc', 'ssim', 'psm']
    res_list = []

    if samples is None or len(samples) == 0:
        raise ValueError("Samples list is empty. Cannot calculate metrics.")

    gt_images = [img[0] for img in samples]
    gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
    samples_to_run = np.arange(1, len(samples[0])) if avg else [1]

    for m in metric_list:
        res_part = []
        for s in samples_to_run:
            pred_images = [img[s] for img in samples]
            pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
            res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m)
            res_part.append(np.mean(res))
        res_list.append(np.mean(res_part))

    res_part = []
    for s in samples_to_run:
        pred_images = [img[s] for img in samples]
        pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
        res = get_similarity_metric(pred_images, gt_images, 'class', None,
                                     n_way=50, num_trials=1000, top_k=1, device='cuda')
        res_part.append(np.mean(res))
    res_list.append(np.mean(res_part))
    res_list.append(np.max(res_part))
    metric_list.append('top-1-class')
    metric_list.append('top-1-class (max)')

    metrics_data = {
        "metrics": {metric: res for metric, res in zip(metric_list, res_list)}
    }
    
    return res_list, metric_list



def process_subject(subject_name, weights_file, dataset_name, root, custom_name=None):
 
    model_path = os.path.join(root, 'pretrains', dataset_name, weights_file)
    print(f"Attempting to load model from: {model_path}")
    try:
        sd = torch.load(model_path, map_location='cpu')
    except FileNotFoundError:
        print(f"Model file not found: {model_path}")
        return
    except Exception as e:
        print(f"Error loading model for {subject_name}: {e}")
        return

    config = sd['config']
    config.root_path = root
    config.kam_path = os.path.join(root, 'data/Kamitani/npz')
    config.bold5000_path = os.path.join(root, 'data/BOLD5000')
    config.pretrain_mbm_path = os.path.join(root, 'pretrains', dataset_name, 'fmri_encoder.pth')
    config.pretrain_gm_path = os.path.join(root, 'pretrains/ldm/label2img')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_transform_test = transforms.Compose([
        normalize, transforms.Resize((256, 256)),
        channel_last
    ])

    try:
        if dataset_name == 'GOD':
            print(f"Dataset: {dataset_name}, Subject: {subject_name}, Weights: {weights_file}")
            if subject_name not in config.kam_subs:
                print(f"Invalid subject for GOD: {subject_name} and {weights_file}")
                return
                print(f"Dataset: {dataset_name}, Subject: {subject_name}, Weights: {weights_file}")
            _, dataset_test = create_Kamitani_dataset(
                config.kam_path, config.roi, config.patch_size,
                fmri_transform=torch.FloatTensor, image_transform=img_transform_test,
                subjects=[subject_name], test_category=config.test_category
            )
        elif dataset_name == 'BOLD5000':
            print(f"Dataset: {dataset_name}, Subject: {subject_name}, Weights: {weights_file}")
            if subject_name not in config.bold5000_subs:
                print(f"Invalid subject for BOLD5000: {subject_name} and {weights_file}")
                return
                print(f"Dataset: {dataset_name}, Subject: {subject_name}, Weights: {weights_file}")
            _, dataset_test = create_BOLD5000_dataset(
                config.bold5000_path, config.patch_size,
                fmri_transform=torch.FloatTensor, image_transform=img_transform_test,
                subjects=[subject_name]
            )

    except Exception as e:
        print(f"Error loading dataset for {subject_name} in {dataset_name}: {e}")
        return
       

    num_voxels = dataset_test.num_voxels
    print(f"Loaded dataset with {len(dataset_test)} samples.")

    pretrain_mbm_metafile = torch.load(config.pretrain_mbm_path, map_location='cpu')
    generative_model = fLDM(
        pretrain_mbm_metafile, num_voxels, device=device,
        pretrain_root=config.pretrain_gm_path, logger=config.logger,
        ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond
    )
    generative_model.model.load_state_dict(sd['model_state_dict'])
    print("Loaded LDM successfully.")

    state = sd['state']
    
    #dataset_test_subset = [dataset_test[i] for i in range(3)]
    
    grid, samples, samples_list, gt_list = generative_model.generate(
    dataset_test, config.num_samples, config.ddim_steps, config.HW, limit=None, state=state
    )

    results_path = os.path.abspath("../results")
    os.makedirs(results_path, exist_ok=True) 
    
    # grid
    eval_dir = os.path.abspath("../results/eval")
    os.makedirs(eval_dir, exist_ok=True)
    grid_img_path = os.path.join(eval_dir, f"{subject_name}_grid.png")

    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(grid_img_path)
    print(f"Grid image saved to {grid_img_path}")

    # eval metrics
    metric, metric_list = get_eval_metric(samples, avg=True)    
    # Create the metric dictionary
    metric_dict = {f'summary/pair-wise_{k}': v for k, v in zip(metric_list[:-2], metric[:-2])}
    metric_dict[f'summary/{metric_list[-2]}'] = metric[-2]
    metric_dict[f'summary/{metric_list[-1]}'] = metric[-1]
    
    print(metric_dict)

    
    results_file_path = os.path.join(results_path, f"metrics_{subject_name}_{weights_file[3]}.txt")
    with open(results_file_path, "w") as file:
        for key, value in metric_dict.items():
            file.write(f"{key}: {value}\n")

    # images    
    for i in range(len(samples_list)):
        sample = process_image(samples_list[i])
        gt = process_image(gt_list[i])
        output_dir = os.path.join(results_path, dataset_name + '_' + subject_name + '_' + weights_file[3])
        os.makedirs(output_dir, exist_ok = True)
        
        sample.save(os.path.join(output_dir,f"sample{i+1}.png"))
        gt.save(os.path.join(output_dir,f"gt{i+1}.png"))


def process_image(tensor):
    
    tensor = tensor.detach().cpu()
    if tensor.dim() == 4 and tensor.size(0) == 1:  
        tensor = tensor.squeeze(0)
    tensor = rearrange(tensor, 'c h w -> h w c')
    tensor = (tensor * 255).clamp(0, 255).numpy().astype(np.uint8)

    img = Image.fromarray(tensor)
    return img
    


if __name__ == '__main__':
    datasets = {
        "GOD": [("sbj_1", "sub1.pth"), ("sbj_3", "sub3.pth")],
        "BOLD5000": [("CSI1", "CSI1.pth"), ("CSI4", "CSI4.pth")]
    }
    counter = 0
    root = os.path.abspath("/student/mgallai/mind-vis")

    for dataset_name, subjects in datasets.items():
        for subject in subjects:
            counter+=1
            print("*************************")
            print(f"************{counter}************")
            print("*************************")
            subject_name, weights_file = subject
            process_subject(subject_name, weights_file, dataset_name, root)
