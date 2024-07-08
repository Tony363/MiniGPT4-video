import yaml 
import json
import argparse
import os
import numpy as np
import torch
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,MulticlassF1Score
from minigpt4.common.eval_utils import prepare_texts, init_model
from minigpt4_video_inference import run,setup_seeds
from utils import init_logger


def get_arguments():
    """
    python3 inference_engagenet.py\
        --videos-dir /home/tony/engagenet_val\
        --cfg-path test_configs/mistral_test_config.yaml\
        --ckpt /home/tony/MiniGPT4-video/minigpt4/training_output/engagenet/mistral/202406160507/checkpoint_49.pth\
        --num-classes 4\
        --label-path /home/tony/engagenet_labels/validation_engagement_labels.json
    """
    parser = argparse.ArgumentParser(description="Inference parameters")
    parser.add_argument("--cfg-path", help="path to configuration file.",default="test_configs/llama2_test_config.yaml")
    parser.add_argument("--ckpt", type=str,default='checkpoints/video_llama_checkpoint_last.pth', help="path to checkpoint")
    parser.add_argument("--videos-dir", type=str,required=True, help="location of videos directory")
    parser.add_argument("--question",
                        type=str, 
                        default="This is a student performing tasks in an online setting. Choose whether the student is 'not engaged','barely engaged', 'engaged', or 'highly engaged'.",
                        help="question to ask")
    parser.add_argument(
        "--label-path", 
        type=str, 
        default='/home/tony/engagenet_labels/validation_engagement_labels.json',
        help="path to EngageNet Labels"
    )
    parser.add_argument("--num-classes", type=int, help="# of classes",default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512, help="max number of generated tokens")
    parser.add_argument("--lora_r", type=int, default=64, help="lora rank of the model")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
                "in xxx=yyy format will be merged into config file (deprecate), "
                "change to --cfg-options instead.",
    )
    return parser.parse_args()

def get_test_labels(
    label_path:str
)->dict:
    label = {}
    classes = np.array([
        [0,'Not-Engaged',"The student is Not-Engaged."],
        [1,'Barely-Engaged',"The student is Barely-Engaged."],
        [2,'Engaged',"The student is Engaged."],
        [4,'Highly-Engaged','The student is Highly-Engaged.']
    ])
    with open(label_path,'r') as f:
        captions = json.load(f)
        for pair in captions:
            label[pair['video_id']] = classes[[
                ('Not-Engaged'.lower() in pair['a'].lower()),
                ('Barely-Engaged'.lower() in pair['a'].lower()),
                ('Engaged'.lower() in pair['a'].lower()),
                ('Highly-Engaged'.lower() in pair['a'].lower())
            ]][0].tolist()
    save = open(os.path.join('/'.join(label_path.split('/')[:-1]),'eval_labels.json'),'w')
    json.dump(label,save,indent=4)
    save.close()
    return label

def load_metrics(num_classes:int)->torchmetrics.MetricCollection:
    metrics = torchmetrics.MetricCollection([
        MulticlassAccuracy(num_classes=num_classes, average="micro"),
        MulticlassPrecision(num_classes=num_classes, average="macro"),
        MulticlassRecall(num_classes=num_classes, average="macro"),
        MulticlassF1Score(num_classes=num_classes, average="macro"),
    ])
    return metrics

def main()->None:
    logger.info("Starting Inference")
    args = get_arguments()
    with open(args.cfg_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        
    setup_seeds(config['run']['seed'])
    logger.info("SEED - {}".format(config['run']['seed']))
    label = get_test_labels(
        label_path=args.label_path
    )
    num_classes,max_new_tokens = args.num_classes,args.max_new_tokens
        
    model, vis_processor = init_model(args)
    model.to(config['run']['device'])
    
    metrics = load_metrics(args.num_classes)
    metrics.to(config['run']['device'])
    
    pred_samples,pred_set = [],{}
    question = args.question
    inference_samples = len(os.listdir(args.videos_dir))
    pred_table,target_table = torch.zeros(inference_samples).to(config['run']['device']),\
        torch.zeros(inference_samples).to(config['run']['device'])

    for sample,vid_path in enumerate(os.listdir(args.videos_dir)):
        vid_id = vid_path.split(".mp4")[0]
        vid_path = os.path.join(args.videos_dir, vid_path)
        logger.info("Processing video - {}".format(vid_path))
        answer = run(vid_path, question, model, vis_processor,max_new_tokens, gen_subtitles=False)
        pred_table[sample] = target_table[sample]
        if label[vid_id][1] not in answer.lower():
            pred_table[sample] = (target_table[sample] - 1) % num_classes

        pred_set['video_name'] = vid_id
        pred_set['Q'] = question
        pred_set['A'] = answer
        pred_samples.append(pred_set)
    
    performance = metrics(pred_table, target_table)
    logger.info(f"ACC - {performance['MulticlassAccuracy']}")
    logger.info(f"PR - {performance['MulticlassPrecision']}")
    logger.info(f"RE - {performance['MulticlassRecall']}")
    logger.info(f"F1 - {performance['MulticlassF1Score']}")

    with open('for_gpt_pred.json','w') as f:
        json.dump(pred_samples,f,indent=4) 
    return

if __name__ == "__main__":
    program = os.path.basename(__file__)
    if os.path.exists(f"logs/{os.path.splitext(program)[0]}.log"):
        os.remove(f"logs/{os.path.splitext(program)[0]}.log")
    logger = init_logger(program)
    main()
