#!/bin/bash
#SBATCH --partition=batch
#SBATCH --job-name=test
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH --time=23:00:00
#SBATCH --mem=110G
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
## run the application:
job_name="test" # Name of the experiment
cfg_path="/home/tony/MiniGPT4-video/train_configs/224_v2_llama2_engagenet_stage_2.yaml" #"train_configs/224_v2_mistral_engagenet_stage_2.yaml" # "/home/tony/MiniGPT4-video/train_configs/224_v2_llama2_engagenet_stage_2.yaml" #
number_of_gpus=2 # number of gpus
# cd ../../

read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
        PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
        ss -lpn | grep -q ":$PORT " || break
done
echo "Port is $PORT"
torchrun --master-port ${PORT} --nproc-per-node $number_of_gpus train.py --job_name ${job_name} --cfg-path ${cfg_path}