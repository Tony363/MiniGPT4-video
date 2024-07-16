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
        --videos-dir /home/tony/engagenet_val/videos\
        --cfg-path test_configs/mistral_test_config.yaml\
        --ckpt minigpt4/training_output/engagenet/mistral/202406160507/checkpoint_49.pth\
        --num-classes 4\
        --gpu-id 1\
        --label-path /home/tony/engagenet_labels/validation_engagement_labels.json
        
    python3 inference_engagenet.py\
        --videos-dir /home/tony/engagenet_val/videos\
        --cfg-path test_configs/llama2_test_config.yaml\
        --ckpt minigpt4/training_output/engagenet/llama2/202406211922/checkpoint_49.pth\
        --num-classes 4\
        --gpu-id 1\
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
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")

    return parser.parse_args()

def get_test_labels(
    label_path:str
)->dict:
    label = {}
    classes = np.array([
        'The student is Not-Engaged.'.lower(),
        'The student is Barely-engaged.'.lower(),
        'The student is Engaged.'.lower(),
        'The student is Highly-Engaged.'.lower()
    ])
    with open(label_path,'r') as f:
        captions = json.load(f)
        for pair in captions:
            label[pair['video_id']] = classes[classes == pair['a'].lower()][0]
    save = open(os.path.join('/'.join(label_path.split('/')[:-1]),'eval_labels.json'),'w')
    json.dump(label,save,indent=4)
    save.close()
    return label,classes

def load_metrics(num_classes:int)->torchmetrics.MetricCollection:
    metrics = torchmetrics.MetricCollection([
        MulticlassAccuracy(num_classes=num_classes, average="macro"),
        MulticlassPrecision(num_classes=num_classes, average="macro"),
        MulticlassRecall(num_classes=num_classes, average="macro"),
        MulticlassF1Score(num_classes=num_classes, average="macro"),
    ])
    return metrics

def main()->None:
    logger.info("Starting Inference")
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu_id}"
    
    with open(args.cfg_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        
    setup_seeds(config['run']['seed'])
    logger.info("SEED - {}".format(config['run']['seed']))
    label,classes = get_test_labels(
        label_path=args.label_path
    )
    num_classes,max_new_tokens = args.num_classes,args.max_new_tokens
    model, vis_processor = init_model(args)
    model.to(config['run']['device'])
    
    metrics = load_metrics(args.num_classes)
    metrics.to(config['run']['device'])
    
    pred_samples = []
    question = args.question
    pred_table,target_table = torch.zeros(num_classes).to(config['run']['device']),\
        torch.zeros(num_classes).to(config['run']['device'])

    for sample,vid_path in enumerate(os.listdir(args.videos_dir)):
        if not ".mp4" in vid_path:
            continue
        
        vid_id = vid_path.split(".mp4")[0]
        vid_path = os.path.join(args.videos_dir, vid_path)
        logger.info("Processing video - {}".format(vid_id))
        answer = run(vid_path, question, model, vis_processor,max_new_tokens, gen_subtitles=False)
        
        logger.info("SAMPLE:{} {} - {}".format(sample,label[vid_id],answer))
        target_table[0] = np.where(classes == label[vid_id])[0][0]
        pred_table[0] = target_table[0] if label[vid_id].split(' ')[-1] == answer.lower() else (target_table[check] - 1) % num_classes
        logger.info(f"CORRECT:{pred_table[0]} - {target_table[0]} - {pred_table[0] == target_table[0]}")

        wrongs = classes[classes != label[vid_id]]
        for check,wrong in enumerate(wrongs):
            logger.info(f"CHECK: {wrong.split(' ')[-1]} - {answer.lower()} - {wrong.split(' ')[-1] in answer.lower()}")
            pred_table[check + 1] = target_table[check + 1] = np.where(classes == wrong)[0][0]
            if wrong.split(' ')[-1] == answer.lower():
                pred_table[check + 1] = (target_table[check + 1] - 1) % num_classes
            logger.info(f"CHECK: {pred_table[check + 1]} - {target_table[check + 1]}")

        
        performance = metrics.forward(pred_table, target_table)
        logger.info(f"ACC - {performance['MulticlassAccuracy']}")
        logger.info(f"PR - {performance['MulticlassPrecision']}")
        logger.info(f"RE - {performance['MulticlassRecall']}")
        logger.info(f"F1 - {performance['MulticlassF1Score']}")
        
        pred_set = {
            'video_name':vid_id,
            'Q':question,
            'A':label[vid_id],
            'pred':answer
        }
        pred_samples.append(pred_set)
    
    performance = metrics.compute()
    logger.info(f"ACC - {performance['MulticlassAccuracy']}")
    logger.info(f"PR - {performance['MulticlassPrecision']}")
    logger.info(f"RE - {performance['MulticlassRecall']}")
    logger.info(f"F1 - {performance['MulticlassF1Score']}")
    metrics.reset()
    
    model_card = args.cfg_path.split(".yaml")[0].split(os.sep)[-1]
    with open(f'gpt_evaluation/{model_card}_eval.json','w') as f:
        json.dump(pred_samples,f,indent=4) 
    return

if __name__ == "__main__":
    '''
    SED prompt: /home/lupang/minigptv2/minigpt4/output/minigpt4_stage2_finetune/20240418113/checkpoint_98.pth
    DPO: /home/lupang/minigptv2/minigpt4/output/minigpt4_stage2_finetune/20240415103/checkpoint_63.pth
    Original prompt: /home/lupang/minigptv2/minigpt4/output/minigpt4_stage2_finetune/20240415103/original_prompt.pth
    
    mistral
    [inference_engagenet.py | INFO | 2024-07-10] F1 - 0.6666666865348816
    [inference_engagenet.py | INFO | 2024-07-10] ACC - 0.6757702827453613
    [inference_engagenet.py | INFO | 2024-07-10] PR - 0.7449684739112854
    [inference_engagenet.py | INFO | 2024-07-10] RE - 0.6757702827453613
    [inference_engagenet.py | INFO | 2024-07-10] F1 - 0.6665439009666443
    
    /home/tony/MiniGPT4-video/gpt_evaluation/mistral_test_config_eval.json 1071
    completed_files: 0
    incomplete_files: 1071
    completed_files: 1071
    incomplete_files: 0
    All evaluation completed!
    Average score for correctness: 4.010270774976657
    completed_files: 0
    incomplete_files: 1071
    completed_files: 1071
    incomplete_files: 0
    All evaluation completed!
    Average score for detailed orientation: 3.7282913165266107
    completed_files: 0
    incomplete_files: 1071
    completed_files: 1071
    incomplete_files: 0
    All evaluation completed!
    Average score for contextual understanding: 3.9719887955182074
    completed_files: 0
    incomplete_files: 1071
    completed_files: 1071
    incomplete_files: 0
    All evaluation completed!
    Average score temporal understanding: 3.918767507002801
    All evaluations completed!
    
    llama2
    [inference_engagenet.py | INFO | 2024-07-10] F1 - 0.6666666865348816                                                                                                  
    [inference_engagenet.py | INFO | 2024-07-10] ACC - 0.6104108095169067                                                                                                 
    [inference_engagenet.py | INFO | 2024-07-10] PR - 0.6712498664855957                                                                                                  
    [inference_engagenet.py | INFO | 2024-07-10] RE - 0.6104108095169067                                                                                                  
    [inference_engagenet.py | INFO | 2024-07-10] F1 - 0.5968020558357239                                                                                                  
    /home/tony/MiniGPT4-video/gpt_evaluation/llama2_test_config_eval.json 1071
    completed_files: 0
    incomplete_files: 1071
    completed_files: 1071
    incomplete_files: 0
    All evaluation completed!
    Average score for correctness: 3.138188608776844
    completed_files: 0
    incomplete_files: 1071
    Error processing file 'subject_74_65ovfrvz97_vid_1_16_0': Gateway timeout. {"error":{"code":524,"message":"Gateway timeout.","param":null,"type":"cf_gateway_timeout"}} 524 {'error': {'code': 524, 'message': 'Gateway timeout.', 'param': None, 'type': 'cf_gateway_timeout'}} {'Date': 'Tue, 16 Jul 2024 13:09:29 GMT', 'Content-Type': 'application/json', 'Content-Length': '92', 'Connection': 'keep-alive', 'Strict-Transport-Security': 'max-age=15552000; includeSubDomains; preload', 'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'SAMEORIGIN', 'Referrer-Policy': 'same-origin', 'Cache-Control': 'private, max-age=0, no-store, no-cache, must-revalidate, post-check=0, pre-check=0', 'Expires': 'Thu, 01 Jan 1970 00:00:01 GMT', 'Server': 'cloudflare', 'CF-RAY': '8a4237ac3adb1986-EWR', 'alt-svc': 'h3=":443"; ma=86400'}
    completed_files: 1070
    incomplete_files: 1
    completed_files: 1071
    incomplete_files: 0
    All evaluation completed!
    Average score for detailed orientation: 3.2324929971988796
    completed_files: 0
    incomplete_files: 1071
    completed_files: 1071
    incomplete_files: 0
    All evaluation completed!
    Average score for contextual understanding: 3.4313725490196076
    completed_files: 0
    incomplete_files: 1071
    completed_files: 1071
    incomplete_files: 0
    All evaluation completed!
    Average score temporal understanding: 2.955182072829132
    All evaluations completed!
    '''
    program = os.path.basename(__file__)
    if os.path.exists(f"logs/{os.path.splitext(program)[0]}.log"):
        os.remove(f"logs/{os.path.splitext(program)[0]}.log")
    logger = init_logger(program)
    main()
