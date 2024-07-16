import openai
import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
from dotenv import load_dotenv
import time

def parse_args():

    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
    parser.add_argument("--api_key", required=True, help="OpenAI API key.")
    parser.add_argument("--num_tasks", required=True, type=int, help="Number of splits.")
    args = parser.parse_args()
    return args


def annotate(prediction_set, caption_files, output_dir):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    for file in caption_files:
        key = file[:-5] # Strip file extension
        qa_set = prediction_set[key]
        question = qa_set['q']
        answer = qa_set['a']
        pred = qa_set['pred']
        try:
            # Compute the correctness score
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": 
                            "You are an intelligent chatbot designed for evaluating the factual accuracy of generative outputs for video-based question-answer pairs. "
                            "Your task is to compare the predicted answer with the correct answer and determine if they are factually consistent. Here's how you can accomplish the task:"
                            "------"
                            "##INSTRUCTIONS: "
                            "- Focus on the factual consistency between the predicted answer and the correct answer. The predicted answer should not contain any misinterpretations or misinformation.\n"
                            "- The predicted answer must be factually accurate and align with the video content.\n"
                            "- Consider synonyms or paraphrases as valid matches.\n"
                            "- Evaluate the factual accuracy of the prediction compared to the answer."
                    },
                    {
                        "role": "user",
                        "content":
                            "Please evaluate the following video-based question-answer pair:\n\n"
                            f"Question: {question}\n"
                            f"Correct Answer: {answer}\n"
                            f"Predicted Answer: {pred}\n\n"
                            "Provide your evaluation only as a factual accuracy score where the factual accuracy score is an integer value between 0 and 5, with 5 indicating the highest level of factual consistency. "
                            "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the factual accuracy score in INTEGER, not STRING."
                            "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                            "For example, your response should look like this: {''score': 4.8}."
                    }
                ]
            )
            # Convert response to a Python dictionary.
            response_message = completion["choices"][0]["message"]["content"]
            response_dict = ast.literal_eval(response_message)
            result_qa_pair = [response_dict, qa_set]

            # Save the question-answer pairs to a json file.
            with open(f"{output_dir}/{key}.json", "w") as f:
                json.dump(result_qa_pair, f)

        except Exception as e:
            print(f"Error processing file '{key}': {e}")
            time.sleep(2)

def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    file = open(args.pred_path)
    pred_contents = json.load(file)
    print(args.pred_path,len(pred_contents))
    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # Iterate through each sample in pred_contents
    for sample in pred_contents:
        video_id = sample['video_name']
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample['video_name'] = f"{video_id}_{video_id_counts[video_id]}"
        new_pred_contents.append(new_sample)

    # Generating list of id's and corresponding files
    id_list = [x['video_name'] for x in new_pred_contents]
    caption_files = [f"{id}.json" for id in id_list]

    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        id = sample['video_name']
        question = sample['Q']
        answer = sample['A']
        pred = sample['pred']
        qa_set = {"q": question, "a": answer, "pred": pred}
        prediction_set[id] = qa_set

    # Set the OpenAI API key.
    # load_dotenv()
    # api_key = os.getenv("API_KEY")
    # openai.api_key = api_key
    openai.api_key = args.api_key
    num_tasks = args.num_tasks

    # While loop to ensure that all captions are processed.
    while True:
        try:
            # Files that have not been processed yet.
            completed_files = os.listdir(output_dir)
            print(f"completed_files: {len(completed_files)}")

            # Files that have not been processed yet.
            incomplete_files = [f for f in caption_files if f not in completed_files]
            print(f"incomplete_files: {len(incomplete_files)}")

            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                break
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1

            # Split tasks into parts.
            part_len = len(incomplete_files) // num_tasks
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, args.output_dir) for part in all_parts]

            # Use a pool of workers to process the files in parallel.
            with Pool() as pool:
                pool.starmap(annotate, task_args)

        except Exception as e:
            print(f"Error: {e}")

    # Combine all the processed files into one
    combined_contents = {}
    json_path = args.output_json

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                combined_contents[file_name[:-5]] = content

    # Write combined content to a json file
    with open(json_path, "w") as json_file:
        json.dump(combined_contents, json_file)
    print("All evaluation completed!")

    # Calculate average score
    score_sum = 0
    count = 0
    for key, result in combined_contents.items():
        count += 1
        score_match = result[0]['score']
        score = int(score_match)
        score_sum += score
    average_score = score_sum / count

    print("Average score for correctness:", average_score)


if __name__ == "__main__":
    '''
    dpo sedbal
    /home/tony/minigptv2/evaluations/dpo_eval_bal.json 396
    completed_files: 0
    incomplete_files: 396
    completed_files: 396
    incomplete_files: 0
    All evaluation completed!
    Average score for correctness: 4.843434343434343
    completed_files: 0
    incomplete_files: 396
    completed_files: 396
    incomplete_files: 0
    All evaluation completed!
    Average score for detailed orientation: 4.886363636363637
    completed_files: 0
    incomplete_files: 396
    completed_files: 396
    incomplete_files: 0
    All evaluation completed!
    Average score for contextual understanding: 4.876262626262626
    completed_files: 0
    incomplete_files: 396
    completed_files: 396
    incomplete_files: 0
    All evaluation completed!
    Average score for consistency: 3.6792929292929295
    All evaluations completed!
    
    
    dpo_eval_handpicked
    /home/tony/minigptv2/evaluations/dpo_eval_handpicked.json 85
    completed_files: 0
    incomplete_files: 85
    completed_files: 85
    incomplete_files: 0
    All evaluation completed!
    Average score for correctness: 4.2823529411764705
    
    completed_files: 0
    incomplete_files: 85
    completed_files: 85
    incomplete_files: 0
    All evaluation completed!
    Average score for detailed orientation: 4.517647058823529
    
    completed_files: 0
    incomplete_files: 85
    completed_files: 85
    incomplete_files: 0
    All evaluation completed!
    Average score for contextual understanding: 4.435294117647059
    All evaluations completed!
    
    
    sed_eval_handpicked
    /home/tony/minigptv2/evaluations/sed_eval_handpicked.json 85
    completed_files: 0
    incomplete_files: 85
    completed_files: 85
    incomplete_files: 0
    All evaluation completed!
    Average score for correctness: 4.1647058823529415
    completed_files: 0
    incomplete_files: 85
    completed_files: 85
    incomplete_files: 0
    All evaluation completed!
    Average score for detailed orientation: 4.376470588235295
    completed_files: 0
    incomplete_files: 85
    completed_files: 85
    incomplete_files: 0
    All evaluation completed!
    Average score for contextual understanding: 4.294117647058823
    All evaluations completed!
    
    sed_eval_bal
    /home/tony/minigptv2/evaluations/sed_eval_bal.json 396
    completed_files: 29
    incomplete_files: 367
    completed_files: 396
    incomplete_files: 0
    All evaluation completed!
    Average score for correctness: 4.77020202020202
    
    completed_files: 0
    incomplete_files: 396
    completed_files: 396
    incomplete_files: 0
    All evaluation completed!
    Average score for detailed orientation: 4.83080808080808
    All evaluations completed!
    
    completed_files: 0
    incomplete_files: 396
    completed_files: 396
    incomplete_files: 0
    All evaluation completed!
    Average score for contextual understanding: 4.818181818181818
    All evaluations completed!
    
    original_eval_bal
    /home/tony/minigptv2/evaluations/original_eval_bal.json 396
    completed_files: 0
    incomplete_files: 396
    completed_files: 396
    incomplete_files: 0
    All evaluation completed!
    Average score for correctness: 4.41919191919192
    completed_files: 0
    incomplete_files: 396
    completed_files: 396
    incomplete_files: 0
    All evaluation completed!
    Average score for detailed orientation: 4.601010101010101
    completed_files: 0
    incomplete_files: 396
    completed_files: 396
    incomplete_files: 0
    All evaluation completed!
    Average score for contextual understanding: 4.598484848484849
    All evaluations completed!
    
    /home/tony/minigptv2/evaluations/original_eval_handpicked.json 85
    completed_files: 0
    incomplete_files: 85
    completed_files: 85
    incomplete_files: 0
    All evaluation completed!
    Average score for correctness: 4.188235294117647
    completed_files: 0d
    incomplete_files: 85
    completed_files: 85
    incomplete_files: 0
    All evaluation completed!
    Average score for detailed orientation: 4.411764705882353
    completed_files: 0
    incomplete_files: 85
    completed_files: 85
    incomplete_files: 0
    All evaluation completed!
    Average score for contextual understanding: 4.305882352941176
    All evaluations completed!
    
    orginal_eval_daisee
    /home/tony/minigptv2/evaluations/original_eval_daisee.json 1104
    completed_files: 0
    incomplete_files: 1104
    completed_files: 1104
    incomplete_files: 0
    All evaluation completed!
    Average score for correctness: 4.4356884057971016
    completed_files: 0
    incomplete_files: 1104
    completed_files: 1104
    incomplete_files: 0
    All evaluation completed!
    Average score for detailed orientation: 4.5978260869565215
    completed_files: 0
    incomplete_files: 1104
    completed_files: 1104
    incomplete_files: 0
    All evaluation completed!
    Average score for contextual understanding: 4.576086956521739
    completed_files: 0
    incomplete_files: 1104
    completed_files: 1104
    incomplete_files: 0
    All evaluation completed!
    Average score for consistency: 3.6784420289855073
    
    /home/tony/minigptv2/evaluations/dpo_eval_daisee.json 1104
    completed_files: 0
    incomplete_files: 1104
    completed_files: 1104
    incomplete_files: 0
    All evaluation completed!
    Average score for correctness: 4.471014492753623
    completed_files: 0
    incomplete_files: 1104
    completed_files: 1104
    incomplete_files: 0
    All evaluation completed!
    Average score for detailed orientation: 4.628623188405797
    completed_files: 0
    incomplete_files: 1104
    completed_files: 1104
    incomplete_files: 0
    All evaluation completed!
    Average score for contextual understanding: 4.594202898550725
    completed_files: 0
    incomplete_files: 1104
    Error processing file '1110031038_frame_0007_0': Gateway timeout. {"error":{"code":524,"message":"Gateway timeout.","param":null,"type":"cf_gateway_timeout"}} 524 {'error': {'code': 524, 'message': 'Gateway timeout.', 'param': None, 'type': 'cf_gateway_timeout'}} {'Date': 'Sat, 13 Jul 2024 23:16:57 GMT', 'Content-Type': 'application/json', 'Content-Length': '92', 'Connection': 'keep-alive', 'Strict-Transport-Security': 'max-age=15552000; includeSubDomains; preload', 'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'SAMEORIGIN', 'Referrer-Policy': 'same-origin', 'Cache-Control': 'private, max-age=0, no-store, no-cache, must-revalidate, post-check=0, pre-check=0', 'Expires': 'Thu, 01 Jan 1970 00:00:01 GMT', 'Server': 'cloudflare', 'CF-RAY': '8a2cf9630bb619f7-EWR', 'alt-svc': 'h3=":443"; ma=86400'}
    completed_files: 1103
    incomplete_files: 1
    completed_files: 1104
    incomplete_files: 0
    All evaluation completed!
    Average score for consistency: 3.6385869565217392
    All evaluations completed!
    
    
    /home/tony/minigptv2/evaluations/dpo_eval_raw.json 17129
    completed_files: 0
    incomplete_files: 17129
    Error processing file '166161_24_0': Gateway timeout. {"error":{"code":524,"message":"Gateway timeout.","param":null,"type":"cf_gateway_timeout"}} 524 {'error': {'code': 524, 'message': 'Gateway timeout.', 'param': None, 'type': 'cf_gateway_timeout'}} {'Date': 'Sun, 14 Jul 2024 14:37:31 GMT', 'Content-Type': 'application/json', 'Content-Length': '92', 'Connection': 'keep-alive', 'Strict-Transport-Security': 'max-age=15552000; includeSubDomains; preload', 'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'SAMEORIGIN', 'Referrer-Policy': 'same-origin', 'Cache-Control': 'private, max-age=0, no-store, no-cache, must-revalidate, post-check=0, pre-check=0', 'Expires': 'Thu, 01 Jan 1970 00:00:01 GMT', 'Server': 'cloudflare', 'CF-RAY': '8a323de14db28c0f-EWR', 'alt-svc': 'h3=":443"; ma=86400'}
    completed_files: 17128
    incomplete_files: 1
    completed_files: 17129
    incomplete_files: 0
    All evaluation completed!
    Average score for correctness: 4.770739681242338
    completed_files: 0
    incomplete_files: 17129
    Error processing file '164325_041_0': invalid syntax (<unknown>, line 1)
    Error processing file '164903_009_0': Gateway timeout. {"error":{"code":524,"message":"Gateway timeout.","param":null,"type":"cf_gateway_timeout"}} 524 {'error': {'code': 524, 'message': 'Gateway timeout.', 'param': None, 'type': 'cf_gateway_timeout'}} {'Date': 'Sun, 14 Jul 2024 14:46:30 GMT', 'Content-Type': 'application/json', 'Content-Length': '92', 'Connection': 'keep-alive', 'Strict-Transport-Security': 'max-age=15552000; includeSubDomains; preload', 'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'SAMEORIGIN', 'Referrer-Policy': 'same-origin', 'Cache-Control': 'private, max-age=0, no-store, no-cache, must-revalidate, post-check=0, pre-check=0', 'Expires': 'Thu, 01 Jan 1970 00:00:01 GMT', 'Server': 'cloudflare', 'CF-RAY': '8a324b060be57d00-EWR', 'alt-svc': 'h3=":443"; ma=86400'}
    Error processing file '181432_2_0': invalid syntax (<unknown>, line 1)
    completed_files: 17126
    incomplete_files: 3
    completed_files: 17129
    incomplete_files: 0
    All evaluation completed!
    Average score for consistency: 3.697997548017981
    All evaluations completed!
    
    /home/tony/minigptv2/evaluations/sed_eval_raw.json 17129
    completed_files: 0
    incomplete_files: 17129
    Error processing file '164317_062_0': invalid syntax (<unknown>, line 1)
    completed_files: 17128
    incomplete_files: 1
    completed_files: 17129
    incomplete_files: 0
    All evaluation completed!
    Average score for correctness: 4.769805592854224
    completed_files: 0
    incomplete_files: 17129
    Error processing file '166162_25_0': invalid syntax (<unknown>, line 1)
    Error processing file '164734_060_0': invalid syntax (<unknown>, line 1)
    Error processing file '165031_003_0': invalid syntax (<unknown>, line 1)
    Error processing file '164327_003_0': Gateway timeout. {"error":{"code":524,"message":"Gateway timeout.","param":null,"type":"cf_gateway_timeout"}} 524 {'error': {'code': 524, 'message': 'Gateway timeout.', 'param': None, 'type': 'cf_gateway_timeout'}} {'Date': 'Sun, 14 Jul 2024 16:23:52 GMT', 'Content-Type': 'application/json', 'Content-Length': '92', 'Connection': 'keep-alive', 'Strict-Transport-Security': 'max-age=15552000; includeSubDomains; preload', 'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'SAMEORIGIN', 'Referrer-Policy': 'same-origin', 'Cache-Control': 'private, max-age=0, no-store, no-cache, must-revalidate, post-check=0, pre-check=0', 'Expires': 'Thu, 01 Jan 1970 00:00:01 GMT', 'Server': 'cloudflare', 'CF-RAY': '8a32d9aab83219df-EWR', 'alt-svc': 'h3=":443"; ma=86400'}
    completed_files: 17125
    incomplete_files: 4
    completed_files: 17129
    incomplete_files: 0
    All evaluation completed!
    Average score for consistency: 3.2680833673886394
    All evaluations completed!
    
    
    /home/tony/minigptv2/evaluations/original_eval_raw.json 17129
    completed_files: 0
    incomplete_files: 17129
    Error processing file '166160_89_0': Gateway timeout. {"error":{"code":524,"message":"Gateway timeout.","param":null,"type":"cf_gateway_timeout"}} 524 {'error': {'code': 524, 'message': 'Gateway timeout.', 'param': None, 'type': 'cf_gateway_timeout'}} {'Date': 'Sun, 14 Jul 2024 18:29:21 GMT', 'Content-Type': 'application/json', 'Content-Length': '92', 'Connection': 'keep-alive', 'Strict-Transport-Security': 'max-age=15552000; includeSubDomains; preload', 'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'SAMEORIGIN', 'Referrer-Policy': 'same-origin', 'Cache-Control': 'private, max-age=0, no-store, no-cache, must-revalidate, post-check=0, pre-check=0', 'Expires': 'Thu, 01 Jan 1970 00:00:01 GMT', 'Server': 'cloudflare', 'CF-RAY': '8a339177caf20ca8-EWR', 'alt-svc': 'h3=":443"; ma=86400'}
    completed_files: 17128
    incomplete_files: 1
    completed_files: 17129
    incomplete_files: 0
    All evaluation completed!
    Average score for correctness: 4.545098954988616
    completed_files: 0
    incomplete_files: 17129
    Error processing file '166334_060_0': invalid syntax (<unknown>, line 1)
    Error processing file '181449_11_0': Gateway timeout. {"error":{"code":524,"message":"Gateway timeout.","param":null,"type":"cf_gateway_timeout"}} 524 {'error': {'code': 524, 'message': 'Gateway timeout.', 'param': None, 'type': 'cf_gateway_timeout'}} {'Date': 'Sun, 14 Jul 2024 18:39:04 GMT', 'Content-Type': 'application/json', 'Content-Length': '92', 'Connection': 'keep-alive', 'Strict-Transport-Security': 'max-age=15552000; includeSubDomains; preload', 'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'SAMEORIGIN', 'Referrer-Policy': 'same-origin', 'Cache-Control': 'private, max-age=0, no-store, no-cache, must-revalidate, post-check=0, pre-check=0', 'Expires': 'Thu, 01 Jan 1970 00:00:01 GMT', 'Server': 'cloudflare', 'CF-RAY': '8a339fb52d6b19cf-EWR', 'alt-svc': 'h3=":443"; ma=86400'}
    Error processing file '164451_001_0': invalid syntax (<unknown>, line 1)
    completed_files: 17126
    incomplete_files: 3
    completed_files: 17129
    incomplete_files: 0
    All evaluation completed!
    Average score for consistency: 3.477435927374628
    All evaluations completed!
    '''
    main()

