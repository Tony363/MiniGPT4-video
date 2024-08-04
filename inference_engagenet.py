import yaml 
import json
import argparse
import os
import numpy as np
import torch
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,MulticlassF1Score
from minigpt4.common.eval_utils import prepare_texts, init_model
from minigpt4_video_inference import run,setup_seeds,prepare_input
from minigpt4.conversation.conversation import CONV_VISION
from utils import init_logger


def get_arguments():
    """
    python3 inference_engagenet.py\
        --videos-dir /home/tony/engagenet_val/videos\
        --cfg-path test_configs/mistral_test_config.yaml\
        --ckpt /home/tony/nvme2tb/mistral_rppg_mamba_half/202407300139/checkpoint_49.pth\
        --num-classes 4\
        --gpu-id 1\
        --label-path /home/tony/engagenet_labels/validation_engagement_labels.json\
        --consistency-qa /home/tony/MiniGPT4-video/gpt_evaluation/consistency_qa_engagenet.json\
        --rppg-dir /home/tony/engagenet_val/rppg_mamba/tensors
        
    python3 inference_engagenet.py\
        --videos-dir /home/tony/engagenet_val/videos\
        --cfg-path test_configs/llama2_test_config.yaml\
        --ckpt /home/tony/MiniGPT4-video/checkpoints/video_llama_checkpoint_best.pth\
        --num-classes 4\
        --gpu-id 0\
        --label-path /home/tony/engagenet_labels/validation_engagement_labels.json\
        --consistency-qa /home/tony/MiniGPT4-video/gpt_evaluation/consistency_qa_engagenet.json
        --rppg-dir /home/tony/engagenet_val/rppg_mamba/tensors

    """
    parser = argparse.ArgumentParser(description="Inference parameters")
    parser.add_argument("--cfg-path", help="path to configuration file.",default="test_configs/llama2_test_config.yaml")
    parser.add_argument("--ckpt", type=str,default='checkpoints/video_llama_checkpoint_last.pth', help="path to checkpoint")
    parser.add_argument("--videos-dir", type=str,required=True, help="location of videos directory")
    parser.add_argument("--num-classes", type=int, help="# of classes",default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512, help="max number of generated tokens")
    parser.add_argument("--lora_r", type=int, default=64, help="lora rank of the model")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument("--rppg-dir", type=str,required=False, help="location of rppg directory",default=None)
    parser.add_argument("--question",
                        type=str, 
                        default="Choose whether the student is 'not engaged','barely engaged', 'engaged', or 'highly engaged'.",
                        help="question to ask"
    )
    parser.add_argument(
       "--sys_instruct",
        type=str, 
        default="""
<s>[INST]
You are an intelligent chatbot that looks at a series of images and chooses between 'not engaged', 'barely engaged', 'engaged', or 'highly engaged'.
The only choices are from following responses:
    {'answer':0} for not-engaged
    {'answer':1} for barely-engaged
    {'answer':2} for engaged
    {'answer':3} for highly-engaged
[/INST]</s>
""",
        help="system prompt" 
    )
    parser.add_argument(
        "--label-path", 
        type=str, 
        default='/home/tony/engagenet_labels/validation_engagement_labels.json',
        help="path to EngageNet Labels"
    )

    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
                "in xxx=yyy format will be merged into config file (deprecate), "
                "change to --cfg-options instead.",
    )
    parser.add_argument(
        '-consistency-qa','--consistency-qa', 
        type=str,
        default='consistency-qa.json',
        help='json of qa pairs', 
        required=True
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")

    return parser.parse_args()

def get_test_labels(
    label_path:str
)->dict:
    label = {}
    classes = np.array([
        ['Not-Engaged'],
        ['Barely-engaged'],
        ['Engaged'],
        ['Highly-Engaged']
    ])
    mapping = {
        'Not-Engaged':0,
        'Barely-engaged':1,
        'Engaged':2,
        'Highly-Engaged':3
    }
    with open(label_path,'r') as f:
        captions = json.load(f)
        for pair in captions:
            label[pair['video_id']] = pair['a']
    save = open(os.path.join('/'.join(label_path.split('/')[:-1]),'eval_labels.json'),'w')
    json.dump(label,save,indent=4)
    save.close()
    return label,classes,mapping

def load_metrics(num_classes:int)->torchmetrics.MetricCollection:
    metrics = torchmetrics.MetricCollection([
        MulticlassAccuracy(num_classes=num_classes, average="macro"),
        MulticlassPrecision(num_classes=num_classes, average="macro"),
        MulticlassRecall(num_classes=num_classes, average="macro"),
        MulticlassF1Score(num_classes=num_classes, average="macro"),
    ])
    return metrics

def prepare_conversation(
    vid_path:str,
    vis_processor:object,
    conv:CONV_VISION,
    sys_instruct:str,
    question:str
)->tuple:
    conv = CONV_VISION.copy()
    conv.system = sys_instruct
    
    prepared_images,prepared_instruction = prepare_input(vis_processor,vid_path,None,question)
    conv.append_message(conv.roles[0], prepared_instruction)
    conv.append_message(conv.roles[1], None)
    q = [conv.get_prompt()]
    return prepared_images,prepared_instruction,q

def main()->None:
    logger.info("Starting Inference")
    args = get_arguments()

    with open(args.cfg_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        
    with open(args.consistency_qa,'r') as f:
        qa_pairs = json.load(f)
        
    setup_seeds(config['run']['seed'])
    logger.info("SEED - {}".format(config['run']['seed']))
    label,classes,mapping = get_test_labels(
        label_path=args.label_path
    )
    num_classes,max_new_tokens = args.num_classes,args.max_new_tokens
    model, vis_processor = init_model(args)
    model.to(config['model']['device'])
    
    video_paths = os.listdir(args.videos_dir)
    
    metrics = load_metrics(args.num_classes)
    metrics.to(config['model']['device'])
    
    pred_samples = []
    samples = len(video_paths)
    logger.info(f"RPPG INFERENCE - {args.rppg_dir is not None}")
    rppg = None
    for sample,vid_path in enumerate(video_paths):
        if not ".mp4" in vid_path:
            continue
        
        vid_id = vid_path.split(".mp4")[0]
        vid_path = os.path.join(args.videos_dir, vid_path)
        logger.info("Processing video - {}".format(vid_id))
        
        rppg_path = os.path.join(args.rppg_dir, f"{vid_id}_0.pt")
        if args.rppg_dir is not None and os.path.exists(rppg_path):
            rppg = torch.load(rppg_path)
            samples['rppg'] = rppg.to(config['model']['device']) 
        
        
        prepared_images,q_prepared_instruction,q_prompt = prepare_conversation(vid_path,vis_processor,CONV_VISION,args.sys_instruct,args.question)
        a1 = model.generate(
            prepared_images, 
            q_prompt, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            lengths=[len(prepared_images)],
            num_beams=1,
            rppg=rppg
        ) 
        
        q1,q2 = qa_pairs[vid_id]['Q1'],qa_pairs[vid_id]['Q2']
        prepared_images,q1_prepared_instruction,q1_prompt = prepare_conversation(vid_path,vis_processor,CONV_VISION,args.sys_instruct,q1)
        a1 = model.generate(
            prepared_images, 
            q1_prompt, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            lengths=[len(prepared_images)],
            num_beams=1,
            rppg=rppg
        ) 
        prepared_images,q2_prepared_instruction,q2_prompt = prepare_conversation(vid_path,vis_processor,CONV_VISION,args.sys_instruct,q2)
        a2 = model.generate(
            prepared_images, 
            q2_prompt, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            lengths=[len(prepared_images)],
            num_beams=1,
            rppg=rppg
        )    
        samples = {
            "image":prepared_images.unsqueeze(0),
            "instruction_input":[q_prepared_instruction],
            "choices":classes,
            "num_choices":[num_classes],
            "length":[45],
        }
        pred_ans = model.predict_class(samples)
        logger.info(f"{sample}: {pred_ans[0]} - {label[vid_id]}")
        
        pred,target = torch.tensor([mapping[pred_ans[0]]]).to(config['model']['device']),torch.tensor([mapping[label[vid_id]]]).to(config['model']['device'])
        performance = metrics.forward(pred,target)
        logger.info(f"ACC - {performance['MulticlassAccuracy']}")
        logger.info(f"PR - {performance['MulticlassPrecision']}")
        logger.info(f"RE - {performance['MulticlassRecall']}")
        logger.info(f"F1 - {performance['MulticlassF1Score']}")
        
        pred_set:dict={
            'video_name':vid_id,
            'Q':args.question,
            'Q1': q1,
            'Q2':q2,
            'A':label[vid_id],
            'pred':pred_ans[0],
            'pred1':a1,
            'pred2':a2
        }
        pred_samples.append(pred_set)
        rppg = None
        
    performance = metrics.compute()
    logger.info(f"FINAL ACC - {performance['MulticlassAccuracy']}")
    logger.info(f"FINAL PR - {performance['MulticlassPrecision']}")
    logger.info(f"FINAL RE - {performance['MulticlassRecall']}")
    logger.info(f"FINAL F1 - {performance['MulticlassF1Score']}")
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
