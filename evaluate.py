
import json
from html_to_ui import *
from helper.metrics import compute_docsim,compute_average_iou, compute_maximum_iou,compute_perceptual_iou,compute_iou
from helper.metrics import compute_overlap,compute_alignment
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image

#from convertHTML.utils import convert_xywh_to_ltrb
def convert_xywh_to_ltrb(bbox,W,H):
    xc, yc, w, h = bbox
    x1 = xc 
    y1 = yc
    x2 = xc + w
    y2 = yc + h
    return [x1/W, y1/H, x2/W, y2/H]

def convert_real_to_xywh(bbox, W, H):
    bbox = [float(i) for i in bbox]
    x1, y1, width, height = bbox
    
    xc = x1 + width / 2.
    yc = y1 + height / 2.
    b = [xc / W, yc / H, width / W, height / H]
    return b

def convert_to_array(generations: List[str]):
    final_res = []
    for sample in generations:
        bboxs, cates = sample.get("bbox"), sample.get("categories")
        bboxs, cates = bboxs.tolist(), cates.tolist()
        #bboxs = np.array(bboxs, dtype=np.float32)
        #cates = np.array(cates, dtype=np.int32)
        bboxs = torch.tensor(bboxs, dtype=torch.float)
        cates = torch.tensor(cates, dtype=torch.float)
        final_res.append((bboxs, cates))
    return final_res

def remove_repeat(bbox, label):
    if bbox.size(0) == 0:
        return bbox, label 
    bbox_label = torch.cat((bbox, label.unsqueeze(1)), dim=1)  
    unique_bbox_label = []  
    for item in bbox_label:   
        same_bbox_label_exists = False  
        for unique_item in unique_bbox_label:  
            if torch.all(torch.eq(item, unique_item)):  
                same_bbox_label_exists = True  
                break  
        if not same_bbox_label_exists:  
            unique_bbox_label.append(item) 
     
    unique_bbox_label = torch.stack(unique_bbox_label) 
    unique_bbox = unique_bbox_label[:, :-1]  
    unique_label = unique_bbox_label[:, -1] 
    
    return unique_bbox, unique_label

def print_scores(scores: Dict):
    scores = {k: scores[k] for k in sorted(scores)}

    tex = ""
    for k, v in scores.items():
        # if k == "Alignment" or k == "Overlap" or "Violation" in k:
        #     v = [_v * 100 for _v in v]
        mean, std = np.mean(v), np.std(v)
        stdp = std * 100.0 / mean
        print(f"\t{k}: {mean:.4f} ({stdp:.4f}%)")
        tex += f"& {mean:.4f}\\std{{{stdp:.1f}}}\% "
    print(tex)
    
def preprocess_batch(layouts, max_len: int):
    layout = defaultdict(list)
    empty_ids = []  # 0: empty 1: full
    # 최대 차원으로 맞춰주기.
    for sample in layouts:
        if not isinstance(sample["bbox"], torch.Tensor):
            bbox, label = torch.tensor(sample["bbox"]), torch.tensor(sample["categories"])
        else:
            bbox, label = sample["bbox"], sample["categories"]
        bbox, label = remove_repeat(bbox, label)
        pad_len = max_len - label.size(0)

        if pad_len == max_len:
            empty_ids.append(0)
            pad_bbox = torch.tensor(np.full((max_len, 4), 0.0), dtype=torch.float)
            pad_label = torch.tensor(np.full((max_len,), 0), dtype=torch.long)
            mask = torch.tensor([False for _ in range(max_len)])
        else:
            empty_ids.append(1)  # not empty
            pad_bbox = torch.tensor(
                np.concatenate([bbox, np.full((pad_len, 4), 0.0)], axis=0),
                dtype=torch.float,
            )
            pad_label = torch.tensor(
                np.concatenate([label, np.full((pad_len,), 0)], axis=0),
                dtype=torch.long,
            )
            mask = torch.tensor(
                [True for _ in range(bbox.shape[0])] + [False for _ in range(pad_len)]
            ) 

        layout["bbox"].append(pad_bbox)
        layout["label"].append(pad_label)
        layout["mask"].append(mask)
        
    bbox = torch.stack(layout["bbox"], dim=0)
    label = torch.stack(layout["label"], dim=0)
    mask = torch.stack(layout["mask"], dim=0)
    
    padding_mask = ~mask.bool()  
    return bbox, label, padding_mask, mask, empty_ids 

def main(args):
    generation_path= "log_dir/train_stage2_with_all_dataset_non_text/generated_sample/test_numerical.jsonl"
    with open(generation_path, "r",encoding="utf-8") as f:
        generations =[json.loads(line) for line in f] 
    golden_path="data/cgl_dataset/for_posternuwa/html_format_img_instruct_all/test_numerical.jsonl"
    with open(golden_path, "r") as f:
        golden = [json.loads(line) for line in f]
    img_path = "data/cgl_dataset/cgl_inpainting_all"
    all_generations = []
    all_bbox_golden =[]
    all_labels_golden = []
    all_bbox_generation=[]
    all_labels_generation =[]
    
    W,H = 513,750
    block_lst = ["id","golden","cond_bbox_input_seqs","continual_gen_input_seqs",'image','gt_bbox']
    for i,sample in enumerate(generations):
        if "id" in sample:
            id_ = sample.pop("id")
        else:
            id_ = i
        
        golden_bboxs,golden_labels = get_bbox(golden[id_].get('labels')[0])
        if args.save_sample_img & (i%10==0):
            path = os.path.join(img_path,golden[id_]['name'][0][:-4]+".png")
            img = Image.open(path)
            draw_gt = draw_bbox(img,golden_bboxs,golden_labels,verbose=False)
            save_path = os.path.join("log_dir/train_stage2_with_all_dataset_non_text/samples",f"real_{i}.jpg")
            draw_gt.save(save_path)
        golden_bboxs = [convert_real_to_xywh(bbox, W, H) for bbox in golden_bboxs] #x,y,w,h
        golden_bboxs = torch.tensor(golden_bboxs,dtype=torch.float)
        golden_labels = torch.tensor(golden_labels, dtype=torch.long)
        per_sample = {
            "id": id_, 
            "golden": {
                "bbox": golden_bboxs,
                "categories": golden_labels,
            },
        } 
        
        for k, v in sample.items():
            if k in block_lst:
                continue
            gen_bboxs, gen_categories = get_bbox(v)
            
            if args.save_sample_img & (i%10==0) & (k == "unconditional"):
                path = os.path.join(img_path,golden[id_]['name'][0][:-4]+".png")
                img = Image.open(path)
                draw_fake = draw_bbox(img,gen_bboxs,gen_categories,verbose=False)
                save_path = os.path.join("log_dir/train_stage2_with_all_dataset_non_text/samples",f"fake_{i}.jpg")
                draw_fake.save(save_path)

            gen_bboxs = [convert_real_to_xywh(bbox, W, H) for bbox in gen_bboxs]
            gen_bboxs = torch.tensor(gen_bboxs, dtype=torch.float)
            
            gen_categories = torch.tensor(gen_categories, dtype=torch.long)
            per_sample[k] = {"bbox": gen_bboxs, "categories": gen_categories}

        all_bbox_golden.append(golden_bboxs)
        all_labels_golden.append(golden_labels)
        all_generations.append(per_sample)  

    all_keys = list(all_generations[0].keys())
    print(f">>> All keys are {all_keys} | Begin to extract features from generations")
    
    scores_all = dict()
    for gk in all_keys:
        # create saver for each k
        if gk in block_lst:
            continue
        feats_gen = [] 
        batch_metrics = defaultdict(float)
        filter_ids = [] # filter the empty bbox 
        k_generations = [tmp[gk] for tmp in all_generations]  
        for i in range(0, len(k_generations), args.batch_size):
            i_end = min(i + args.batch_size, len(k_generations))
            batch = k_generations[i:i_end]
            max_len = max(len(g["categories"]) for g in batch)
            if max_len == 0:  # prevent not generations
                max_len == 1
            bbox, label, padding_mask, mask, empty_ids = preprocess_batch(batch, max_len)
            filter_ids.extend(empty_ids)
            
            mask_empty_ids = torch.tensor(empty_ids, dtype=torch.bool)
            bbox = torch.masked_select(bbox, mask_empty_ids.unsqueeze(1).unsqueeze(2)).reshape(-1, bbox.size(1), bbox.size(2)).contiguous()
            label = torch.masked_select(label, mask_empty_ids.unsqueeze(1)).reshape(-1, label.size(1)).contiguous()
            padding_mask = torch.masked_select(padding_mask, mask_empty_ids.unsqueeze(1)).reshape(-1, padding_mask.size(1)).contiguous()
            mask = torch.masked_select(mask, mask_empty_ids.unsqueeze(1)).reshape(-1, mask.size(1)).contiguous()   
                
            for k, v in compute_alignment(bbox, mask).items():
                batch_metrics[k] += v.sum().item()
            for k, v in compute_overlap(bbox, mask).items():
                batch_metrics[k] += v.sum().item()  
        gen_data = convert_to_array(k_generations)

        scores = {}
        for k, v in batch_metrics.items():
            scores[k] = v / len(k_generations)
        
        
        scores.update(compute_average_iou(gen_data))
        scores["maximum_iou"] = compute_maximum_iou(zip(all_bbox_golden,all_labels_golden), gen_data)
        scores["DocSim"] = compute_docsim(zip(all_bbox_golden,all_labels_golden), gen_data)
        
        scores["Failed_rate"] = (len(filter_ids) - sum(filter_ids)) / len(filter_ids) * 100
        
        scores_all[gk] = defaultdict(list)
        for k, v in scores.items():
            scores_all[gk][k].append(v)
        print("-"*40)
        print(f">>> Logging Scores for condition {gk}")
        print_scores(scores_all[gk])

    #print("real_iou : ",compute_average_iou(zip(all_bbox_golden,all_labels_golden)))
    #print("generated_iou : ",compute_average_iou(zip(all_bbox_generation,all_labels_generation)))
    #print("docsim : ",compute_docsim(zip(all_bbox_golden,all_labels_golden),zip(all_bbox_generation,all_labels_generation)))
    #print("M-iou : ",compute_maximum_iou(zip(all_bbox_golden,all_labels_golden),zip(all_bbox_generation,all_labels_generation)))
    
    
    
    
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Generated Layout Code")
    parser.add_argument("--test_others", action='store_true', help="custom your evaluation own code here")  
    parser.add_argument("--file_dir", type=str, default="data/generated_results/rico")  
    parser.add_argument("--intermediate_saved_path", type=str, default=None) 
    #parser.add_argument("--golden_file", type=str, default="data/generated_results/rico/golden.jsonl")  
    #parser.add_argument("--fid_model_name_or_path", type=str, default="models/rico25-max25",)  
    #parser.add_argument("--cluster_model", type=str, default="models/rico25-max25/rico25_max25_kmeans_train_clusters.pkl")
    parser.add_argument("--dataset_name", type=str, default="cgl")
    parser.add_argument("--dataset_path", type=str, default="data/rico25-max25")
    parser.add_argument("--gen_res_path", type=str, default=None, help="/data1/poong/PosterNUWA/log_dir/test/generated_sample/test_numerical.jsonl")  
    parser.add_argument("--batch_size", type=int, default=3028)  
    parser.add_argument("--device", type=str, default="cuda:0")  
    parser.add_argument("--save_sample_img", type=bool, default=True)  

    args = parser.parse_args()
    
    int_to_lable = DATASET_META.get(args.dataset_name)
    label_to_int = dict([(v, k) for k, v in int_to_lable.items()])

    main(args)