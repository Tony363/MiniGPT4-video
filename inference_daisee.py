import yaml 
import json
import argparse
import os
import re
import random
import numpy as np
import torch
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall,MulticlassF1Score
from minigpt4.common.eval_utils import prepare_texts, init_model
from minigpt4_video_inference import run,setup_seeds,prepare_input
from minigpt4.conversation.conversation import CONV_VISION
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
import minigpt4.tasks as tasks

from tqdm import tqdm
from utils import init_logger


def get_arguments():
    """
    python3 inference_daisee.py\
        --videos-dir /home/tony/nvme2tb/DAiSEE/dataset/test/videos\
        --cfg-path test_configs/mistral_daisee_test_config.yaml\
        --ckpt minigpt4/training_output/engagenet/mistral_daisee/202501091440/checkpoint_8.pth\
        --num-classes 4\
        --gpu-id 0\
        --label-path /home/tony/MiniGPT4-video/daisee_captions/test_filter_cap.json
        --question-prompts /home/tony/MiniGPT4-video/prompts/daisee_questions.txt

    python3 inference_daisee.py\
        --videos-dir /home/tony/nvme2tb/DAiSEE/dataset/test/videos\
        --cfg-path test_configs/mistral_daisee_base_config.yaml\
        --ckpt checkpoints/video_mistral_checkpoint_best.pth\
        --num-classes 4\
        --gpu-id 0\
        --label-path /home/tony/MiniGPT4-video/daisee_captions/test_filter_cap.json
        --question-prompts /home/tony/MiniGPT4-video/prompts/daisee_questions.txt

    python3 inference_daisee.py\
        --videos-dir /home/tony/nvme2tb/EngageNet/Test/videos\
        --cfg-path test_configs/mistral_engagenet_base_config.yaml\
        --ckpt checkpoints/video_mistral_checkpoint_best.pth\
        --num-classes 4\
        --gpu-id 0\
        --label-path /home/tony/MiniGPT-4/engagenet_captions/test_filter_cap.json


    """
    parser = argparse.ArgumentParser(description="Inference parameters")
    parser.add_argument("--cfg-path", help="path to configuration file.",default="test_configs/llama2_test_config.yaml")
    parser.add_argument("--ckpt", type=str,default='checkpoints/video_llama_checkpoint_last.pth', help="path to checkpoint")
    parser.add_argument("--videos-dir", type=str,required=True, help="location of videos directory")
    parser.add_argument("--num-classes", type=int, help="# of classes",default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512, help="max number of generated tokens")
    parser.add_argument("--lora_r", type=int, default=64, help="lora rank of the model")
    parser.add_argument("--lora_alpha", type=int, default=16, help="lora alpha")
    parser.add_argument(
        "--label-path", 
        type=str, 
        default='/home/tony/MiniGPT4-video/daisee_captions/test_filter_cap.json',
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
        '-question-prompts','--question_prompts', 
        type=str,
        default='prompts/daisee_questions.txt',
        help='questions for consistency check', 
        required=False
    )
    parser.add_argument(
        "--eval-prompts", 
        type=str, 
        default='prompts/instruction_align.txt', 
        help="text file of instruction prompts",
        required=False
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")

    return parser.parse_args()

def get_test_labels(
    label_path:str
)->dict:
    # mapping = {
    #     'The student is Not-Engaged':0,
    #     'The student is Barely-Engaged':1,
    #     'The student is Engaged':2,
    #     'The student is Highly-Engaged':3
    # }
    mapping = {
        'The student is not-engaged':0,
        'The student is barely-engaged':1,
        'The student is engaged':2,
        'The student is highly-engaged':3
    }
    with open(label_path,'r') as f:
        label = json.load(f)

    return label['annotations'],mapping

def load_metrics(num_classes:int)->torchmetrics.MetricCollection:
    metrics = torchmetrics.MetricCollection([
        MulticlassAccuracy(num_classes=num_classes, average="micro"),
        MulticlassPrecision(num_classes=num_classes, average="macro"),
        MulticlassRecall(num_classes=num_classes, average="macro"),
        MulticlassF1Score(num_classes=num_classes, average="macro"),
    ])
    return metrics

def prepare_conversation(
    subject:str,
    vis_processor:object,
    conv:CONV_VISION,
    sys_instruct:str,
    question:str
)->tuple:
    conv = CONV_VISION.copy()
    conv.system = sys_instruct
    
    prepared_images,prepared_instruction = prepare_input(vis_processor,subject,None,question)
    conv.append_message(conv.roles[0], prepared_instruction)
    conv.append_message(conv.roles[1], None)
    q = [conv.get_prompt()]
    return prepared_images,prepared_instruction,q

def check_string_in_output(
    output:str, 
    search_string:str
)->bool:
    # Escape special characters in search_string if necessary
    output,search = re.sub(r'\W', '', output).lower(),re.sub(r'\W', '', search_string).lower()
    pattern = re.escape(search)
    match = re.search(pattern, output)
    return bool(match)


def main()->None:
    logger.info("Starting Inference")
    args = get_arguments()

    with open(args.eval_prompts, 'r', encoding='utf-8') as file:
        prompt = file.read()
    instruction_pool = prompt.split('\n\n')
    
    question = "Question: What is the student's engagement level?"
    with open(args.question_prompts,'r') as f:
        questions = f.read().split('\n')
        questions.remove(question)

    with open(args.cfg_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
        
    setup_seeds(config['run']['seed'])
    logger.info("SEED - {}".format(config['run']['seed']))
    label,mapping = get_test_labels(
        label_path=args.label_path
    )
    num_classes,max_new_tokens = args.num_classes,args.max_new_tokens
    model, vis_processor = init_model(args)
    model = model.to(config['run']['device'])
    model.eval()
    
    video_paths = os.listdir(args.videos_dir)
    vidid_filemap = {
        file.split('.')[0]:file 
        for file in video_paths
    }
    metrics = load_metrics(args.num_classes).to(config['run']['device'])

    inference_samples = len(label)
    pred_table,target_table = torch.zeros(inference_samples).to(config['run']['device']),torch.zeros(inference_samples).to(config['run']['device'])
    
    pred_samples = []
    for i,subject in enumerate(tqdm(label)):
        vid_id = vidid_filemap[subject['video_id']]
        video_path = os.path.join(args.videos_dir, vid_id)
        logger.info("Processing video - {}".format(video_path))

        target_table[i] = mapping[subject['caption']]
        pred_table[i] = target_table[i]

        instruction = random.choice(instruction_pool)
        q2 = random.choice(questions)

        prepared_images,q_prepared_instruction,q_prompt = prepare_conversation(video_path,vis_processor,CONV_VISION,instruction,question)

        a = model.generate(
            prepared_images, 
            q_prompt, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            lengths=[len(prepared_images)],
            num_beams=1,
        ) 
        logger.info(f"PRED - {a[0]}")
        logger.info(f"CAPTION - {subject['caption'].split(' ')[-1]}")
        if not check_string_in_output(a[0],subject['caption'].split(' ')[-1]): # and subject['caption'].split(' ')[-1].lower not in a.lower():
            pred_table[i] = (target_table[i] - 1) % args.num_classes

        _,q1_prepared_instruction,q1_prompt = prepare_conversation(video_path,vis_processor,CONV_VISION,instruction,q2)
        a1 = model.generate(
            prepared_images, 
            q1_prompt, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            lengths=[len(prepared_images)],
            num_beams=1,
        ) 

        performance = metrics.forward(pred_table[:i + 1],target_table[:i + 1])
        logger.info(f"ACC - {performance['MulticlassAccuracy']}")
        logger.info(f"PR - {performance['MulticlassPrecision']}")
        logger.info(f"RE - {performance['MulticlassRecall']}")
        logger.info(f"F1 - {performance['MulticlassF1Score']}")
        
        pred_set = {
            'video_name':vid_id,
            'Q':question,
            'Q1':question,
            'Q2':q2,
            'A':subject['caption'],
            'pred':a,
            'pred1':a,
            'pred2':a1
        }
        pred_samples.append(pred_set)
        
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
    MiniGPT4-Video - daisee Finetune
    [inference_daisee.py | INFO | 2025-01-14] FINAL ACC - 0.7163130640983582
    [inference_daisee.py | INFO | 2025-01-14] FINAL PR - 0.44986313581466675
    [inference_daisee.py | INFO | 2025-01-14] FINAL RE - 0.37715649604797363
    [inference_daisee.py | INFO | 2025-01-14] FINAL F1 - 0.3880458474159241

    Average score for correctness: 4.082399103139013
    Average score for detailed orientation: 3.6541479820627805
    Average score for contextual understanding: 3.951793721973094
    Average score temporal understanding: 3.7954035874439462
    Average score for consistency: 2.8542600896860986


    MiniGPT4-Video - daisee Base
    INFO:inference_daisee.py:FINAL ACC - 0.5688818097114563
    INFO:inference_daisee.py:FINAL PR - 0.4028744101524353
    INFO:inference_daisee.py:FINAL RE - 0.28821173310279846
    INFO:inference_daisee.py:FINAL F1 - 0.3005830943584442

    /home/tony/MiniGPT4-video/gpt_evaluation/mistral_daisee_base_config_eval.json 1784
    Average score for correctness: 3.374439461883408
    Average score for detailed orientation: 3.0812780269058297
    Average score for contextual understanding: 3.3323991031390134
    Average score temporal understanding: 3.0291479820627805
    Average score for consistency: 2.258408071748879

    MiniGPT4-Video - Engagenet base
    INFO:inference_daisee.py:FINAL ACC - 0.35479679703712463
    INFO:inference_daisee.py:FINAL PR - 0.24729931354522705
    INFO:inference_daisee.py:FINAL RE - 0.2997397482395172
    INFO:inference_daisee.py:FINAL F1 - 0.2471315562725067

    Average score for correctness: 3.2681388012618298
    Average score for detailed orientation: 3.0546792849631967
    Average score for contextual understanding: 3.2323869610935856
    Average score temporal understanding: 2.843322818086225
    Average score for consistency: 2.2018927444794953
    '''
    program = os.path.basename(__file__)
    if os.path.exists(f"logs/{os.path.splitext(program)[0]}.log"):
        os.remove(f"logs/{os.path.splitext(program)[0]}.log")
    logger = init_logger(program)
    main()
