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
        --videos-dir /home/tony/engagenet_val/videos\
        --cfg-path test_configs/mistral_test_config.yaml\
        --ckpt /home/tony/MiniGPT4-video/minigpt4/training_output/engagenet/mistral/202408102213/checkpoint_49.pth\
        --num-classes 4\
        --gpu-id 1\
        --label-path /home/tony/engagenet_labels/validation_engagement_labels.json\
        --consistency-qa /home/tony/MiniGPT4-video/gpt_evaluation/consistency_qa_engagenet.json\
    
    python3 inference_engagenet.py\
        --videos-dir /home/tony/engagenet_val/videos\
        --cfg-path test_configs/mistral_base_test_config.yaml\
        --ckpt /home/tony/MiniGPT4-video/checkpoints/video_mistral_checkpoint_best.pth\
        --num-classes 4\
        --gpu-id 0\
        --label-path /home/tony/engagenet_labels/validation_engagement_labels.json\
        --consistency-qa /home/tony/MiniGPT4-video/gpt_evaluation/consistency_qa_engagenet.json
    
    python3 inference_engagenet.py\
        --videos-dir /home/tony/engagenet_val/videos\
        --cfg-path test_configs/mistral_rppg_test_config.yaml\
        --ckpt /home/tony/MiniGPT4-video/minigpt4/training_output/engagenet/mistral_rppg_former/202408220314/checkpoint_17.pth\
        --num-classes 4\
        --label-path /home/tony/engagenet_labels/validation_engagement_labels.json\
        --consistency-qa /home/tony/MiniGPT4-video/gpt_evaluation/consistency_qa_engagenet.json\
        --rppg-dir /home/tony/nvme2tb/rhythmformer_rppg/validation/validation-tensors
        
        
    python3 inference_engagenet.py\
        --videos-dir /home/tony/engagenet_val/videos\
        --cfg-path test_configs/mistral_rppg2_test_config.yaml\
        --ckpt /home/tony/MiniGPT4-video/minigpt4/training_output/engagenet/mistral_rppg2_former/202408260303/checkpoint_9.pth\
        --num-classes 4\
        --label-path /home/tony/engagenet_labels/validation_engagement_labels.json\
        --consistency-qa /home/tony/MiniGPT4-video/gpt_evaluation/consistency_qa_engagenet.json\
        --rppg-dir /home/tony/nvme2tb/rhythmformer_rppg/validation/validation-tensors
        
    python3 inference_engagenet.py\
        --videos-dir /home/tony/engagenet_val/videos\
        --cfg-path test_configs/mistral_finetune_test_config.yaml\
        --ckpt minigpt4/training_output/engagenet/mistral/202408240305/checkpoint_9.pth\
        --num-classes 4\
        --label-path /home/tony/engagenet_labels/validation_engagement_labels.json\
        --consistency-qa /home/tony/MiniGPT4-video/gpt_evaluation/consistency_qa_engagenet.json
    
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
    parser.add_argument("--rppg-dir", type=str,required=False, help="location of rppg directory",default="")
    parser.add_argument("--question",
                        type=str, 
                        default="Choose whether the student is 'Not Engaged','Barely Engaged', 'Engaged', or 'Highly Engaged'",
                        help="question to ask"
    )
    
    parser.add_argument(
       "--sys_instruct",
        type=str, 
        default="<s> [INST] This is a video of a student performing tasks in an online setting.[/INST]</s>",
#         default="""
# <s>[INST]
# You are an intelligent chatbot that looks at a series of images and chooses between 'not engaged', 'barely engaged', 'engaged', or 'highly engaged'.
# The only choices are from following responses:
#     {'answer':0} for not-engaged
#     {'answer':1} for barely-engaged
#     {'answer':2} for engaged
#     {'answer':3} for highly-engaged
# [/INST]</s>
# """,
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
        ['The student is Not-Engaged.'],
        ['The student is Barely-engaged.'],
        ['The student is Engaged.'],
        ['The student is Highly-Engaged.']
    ])
    mapping = {
        'The student is Not-Engaged.':0,
        'The student is Barely-engaged.':1,
        'The student is Engaged.':2,
        'The student is Highly-Engaged.':3
    }
    with open(label_path,'r') as f:
        captions = json.load(f)
        for pair in captions:
            label[pair['video_id']] = pair['QA'['a']] if 'daisee' in label_path else pair['a']
    save = open(os.path.join('/'.join(label_path.split('/')[:-1]),'eval_labels.json'),'w')
    json.dump(label,save,indent=4)
    save.close()
    return label,classes,mapping

def load_metrics(num_classes:int)->torchmetrics.MetricCollection:
    metrics = torchmetrics.MetricCollection([
        MulticlassAccuracy(num_classes=num_classes, average="micro"),
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
    model = model.to(config['run']['device'])
    # model.eval()
    
    video_paths = os.listdir(args.videos_dir)
    
    metrics = load_metrics(args.num_classes)
    metrics.to(config['run']['device'])
    
    pred_samples = []
    rppg = None
    for sample,vid_path in enumerate(tqdm(video_paths)):
        if not ".mp4" in vid_path:
            continue
        
        vid_id = vid_path.split(".mp4")[0]
        vid_path = os.path.join(args.videos_dir, vid_path)
        logger.info("Processing video - {}".format(vid_id))
        
        prepared_images,q_prepared_instruction,q_prompt = prepare_conversation(vid_path,vis_processor,CONV_VISION,args.sys_instruct,args.question)

        # samples = {
        #     "image":prepared_images.unsqueeze(0).to(config['run']['device']),
        #     "instruction_input":[q_prepared_instruction],
        #     "choices":classes,
        #     "num_choices":[num_classes],
        #     "length":[len(prepared_images)],
        # }
        rppg_path = os.path.join(args.rppg_dir, f"{vid_id}_0.pt")
        logger.info(f"EXISTS - {os.path.exists(rppg_path)} {rppg_path}")
        if args.rppg_dir and os.path.exists(rppg_path):
            rppg = torch.load(rppg_path).to(config['run']['device'])
            # samples['rppg'] = rppg
            logger.info(f"RPPG INFERENCE - {rppg is not None}")
            
            a = model.generate(
                prepared_images, 
                q_prompt, 
                max_new_tokens=max_new_tokens, 
                do_sample=True, 
                lengths=[len(prepared_images)],
                num_beams=1,
                rppg=rppg
            ) 
            
            q1,q2 = qa_pairs[vid_id]['Q1'],qa_pairs[vid_id]['Q2']
            _,q1_prepared_instruction,q1_prompt = prepare_conversation(vid_path,vis_processor,CONV_VISION,args.sys_instruct,q1)
            a1 = model.generate(
                prepared_images, 
                q1_prompt, 
                max_new_tokens=max_new_tokens, 
                do_sample=True, 
                lengths=[len(prepared_images)],
                num_beams=1,
                rppg=rppg
            ) 
            _,q2_prepared_instruction,q2_prompt = prepare_conversation(vid_path,vis_processor,CONV_VISION,args.sys_instruct,q2)
            a2 = model.generate(
                prepared_images, 
                q2_prompt, 
                max_new_tokens=max_new_tokens, 
                do_sample=True, 
                lengths=[len(prepared_images)],
                num_beams=1,
                rppg=rppg
            )
        else:
            logger.info(f"RPPG INFERENCE - {rppg is not None}")
            a = model.generate(
                prepared_images, 
                q_prompt, 
                max_new_tokens=max_new_tokens, 
                do_sample=True, 
                lengths=[len(prepared_images)],
                num_beams=1,
            ) 
            
            q1,q2 = qa_pairs[vid_id]['Q1'],qa_pairs[vid_id]['Q2']
            _,q1_prepared_instruction,q1_prompt = prepare_conversation(vid_path,vis_processor,CONV_VISION,args.sys_instruct,q1)
            a1 = model.generate(
                prepared_images, 
                q1_prompt, 
                max_new_tokens=max_new_tokens, 
                do_sample=True, 
                lengths=[len(prepared_images)],
                num_beams=1,
            ) 
            _,q2_prepared_instruction,q2_prompt = prepare_conversation(vid_path,vis_processor,CONV_VISION,args.sys_instruct,q2)
            a2 = model.generate(
                prepared_images, 
                q2_prompt, 
                max_new_tokens=max_new_tokens, 
                do_sample=True, 
                lengths=[len(prepared_images)],
                num_beams=1,
            )    
        # pred_ans = model.predict_class(samples)
        # logger.info(f"{sample}: {pred_ans[0]} - {label[vid_id]}")
        
        prepared_images = prepared_images.to(config['run']['device'])

        
        # pred,target = torch.tensor([mapping[pred_ans[0]]]).to(config['run']['device']),torch.tensor([mapping[label[vid_id]]]).to(config['run']['device'])
        # performance = metrics.forward(pred,target)
        # logger.info(f"ACC - {performance['MulticlassAccuracy']}")
        # logger.info(f"PR - {performance['MulticlassPrecision']}")
        # logger.info(f"RE - {performance['MulticlassRecall']}")
        # logger.info(f"F1 - {performance['MulticlassF1Score']}")
        
        pred_set:dict={
            'video_name':vid_id,
            'Q':args.question,
            'Q1':q1,
            'Q2':q2,
            'A':label[vid_id],
            'pred':a,
            'pred1':a1,
            'pred2':a2
        }
        pred_samples.append(pred_set)
        rppg = None
        
    # performance = metrics.compute()
    # logger.info(f"FINAL ACC - {performance['MulticlassAccuracy']}")
    # logger.info(f"FINAL PR - {performance['MulticlassPrecision']}")
    # logger.info(f"FINAL RE - {performance['MulticlassRecall']}")
    # logger.info(f"FINAL F1 - {performance['MulticlassF1Score']}")
    # metrics.reset()
    
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
    /home/tony/MiniGPT4-video/gpt_evaluation/mistral_test_config_eval.json 1071
    Average score for correctness: 4.010270774976657
    Average score for detailed orientation: 3.7282913165266107
    Average score for contextual understanding: 3.9719887955182074
    Average score temporal understanding: 3.918767507002801
    All evaluations completed!
    
    llama2                                                                                                
    /home/tony/MiniGPT4-video/gpt_evaluation/llama2_test_config_eval.json 1071
    Average score for correctness: 3.138188608776844
    Average score for detailed orientation: 3.2324929971988796
    Average score for contextual understanding: 3.4313725490196076
    Average score temporal understanding: 2.955182072829132
    
    mistral-rppg-finetune no rppg
    Average score for correctness: 3.5676937441643326                                                                                   
    Average score for detailed orientation: 3.281045751633987                                                                            
    Average score for contextual understanding: 3.8823529411764706                                                                       
    Average score temporal understanding: 3.4248366013071894
    Average score for consistency: 2.7413632119514473
    
    mistral-rppg  with LoRA 4 bits
    Average score for correctness: 2.764705882352941   
    Average score for detailed orientation: 2.319327731092437   
    Average score for contextual understanding: 2.8683473389355743   
    Average score temporal understanding: 2.819794584500467    
    Average score for consistency: 1.6704014939309058         
    
    mistral-finetune with LoRA 4 bits
    Average score for correctness: 2.549019607843137                                                                                                                                                                                                                                
    Average score for detailed orientation: 2.1260504201680672      
    Average score for contextual understanding: 2.637721755368814
    Average score temporal understanding: 2.604108309990663         
    Average score for consistency: 0.8141923436041083   

    
    mistral base
    Average score for correctness: 3.1333333333333333
    Average score for detailed orientation: 2.3076923076923075
    Average score for contextual understanding: 2.5753846153846154
    Average score temporal understanding: 0.5528205128205128
    Average score for consistency: 2.3528205128205126
    
    mistral finetune
    Average score for correctness: 3.9292307692307693
    Average score for detailed orientation: 2.4707692307692306
    Average score for contextual understanding: 3.4123076923076923
    Average score temporal understanding: 0.838974358974359
    Average score for consistency: 2.8984615384615386
    
    Average score for correctness: 3.8984615384615386
    Average score for detailed orientation: 2.2594871794871794
    Average score for contextual understanding: 3.233846153846154
    Average score temporal understanding: 0.7938461538461539
    Average score for consistency: 2.838974358974359
    
    mistral rppg                      
    Average score for correctness: 2.7394871794871793
    Average score for detailed orientation: 2.365128205128205
    Average score for contextual understanding: 2.758974358974359
    Average score temporal understanding: 2.884102564102564
    Average score for consistency: 2.5025641025641026
    
    mistral rppg2
    Average score for correctness: 3.675897435897436
    Average score for detailed orientation: 3.7056410256410257
    Average score for contextual understanding: 3.8964102564102565
    Average score temporal understanding: 3.4728205128205127
    Average score for consistency: 3.797948717948718
    '''
    program = os.path.basename(__file__)
    if os.path.exists(f"logs/{os.path.splitext(program)[0]}.log"):
        os.remove(f"logs/{os.path.splitext(program)[0]}.log")
    logger = init_logger(program)
    main()
