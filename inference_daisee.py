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
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
import minigpt4.tasks as tasks

from tqdm import tqdm
from utils import init_logger


def get_arguments():
    """
    
    python3 inference_engagenet.py\
        --videos-dir /home/tony/nvme2tb/DAiSEE/dataset/test/videos\
        --cfg-path test_configs/llama2_test_config.yaml\
        --ckpt /home/tony/MiniGPT4-video/checkpoints/video_llama_checkpoint_best.pth\
        --num-classes 4\
        --gpu-id 0\
        --label-path /home/tony/engagenet_labels/validation_engagement_labels.json\
        --consistency-qa /home/tony/MiniGPT4-video/gpt_evaluation/consistency_qa_engagenet.json

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
    parser.add_argument(
        "--eval_prompts", 
        type=str, 
        default='prompts/instruction_align.txt', 
        help="text file of instruction prompts"
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")

    return parser.parse_args()

def get_test_labels(
    label_path:str
)->dict:
    mapping = {
        'The student is Not-Engaged.':0,
        'The student is Barely-engaged.':1,
        'The student is Engaged.':2,
        'The student is Highly-Engaged.':3
    }

    with open(label_path,'r') as f:
        label = json.load(f)
    save = open(os.path.join('/'.join(label_path.split('/')[:-1]),'eval_labels.json'),'w')
    json.dump(label,save,indent=4)
    save.close()
    return label,mapping

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

def main()->None:
    logger.info("Starting Inference")
    args = get_arguments()

    with open(args.eval_prompts, 'r', encoding='utf-8') as file:
        prompt = file.read()
    instruction_pool = prompt.split('\n\n')
    
    question = "Question: What is the student's engagement level?"
    with open(args.consistency_qa,'r') as f:
        qa_pairs = json.load(f)

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
    metrics = load_metrics(args.num_classes).to(config['run']['device'])

    inference_samples = len(label)
    pred_table,target_table = torch.zeros(inference_samples).to(config['run']['device']),torch.zeros(inference_samples).to(config['run']['device'])
    
    pred_samples = []
    for i,subject in enumerate(tqdm(label)):
        vid_id = subject.split(".mp4")[0]
        video_path = os.path.join(args.videos_dir, vid_id)
        logger.info("Processing video - {}".format(vid_path))

        target_table[i] = mapping[subject['caption']]
        pred_table[i] = target_table[i]
        instruction = random.choice(instruction_pool)
        questions = qa_pairs[vid_id]['Q1'],qa_pairs[vid_id]['Q2']

        prepared_images,q_prepared_instruction,q_prompt = prepare_conversation(video_path,vis_processor,CONV_VISION,instruction,question)
        a = model.generate(
            prepared_images, 
            q_prompt, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            lengths=[len(prepared_images)],
            num_beams=1,
        ) 
        if subject['caption'].split(' ')[-1].lower not in a.lower():
            pred_table[i] = (target_table[i] - 1) % args.num_classes

        _,q1_prepared_instruction,q1_prompt = prepare_conversation(subject,vis_processor,CONV_VISION,instruction,random.choice(questions))
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
            'Q':args.question,
            'Q1':q1,
            'Q2':q2,
            'A':subject['QA']['a'],
            'pred':a,
            'pred1':a1,
            'pred2':a2
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
    '''
    program = os.path.basename(__file__)
    if os.path.exists(f"logs/{os.path.splitext(program)[0]}.log"):
        os.remove(f"logs/{os.path.splitext(program)[0]}.log")
    logger = init_logger(program)
    main()
