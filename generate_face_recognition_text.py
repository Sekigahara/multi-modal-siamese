import os
import tqdm

from module.util_llm import *
from module.inference_llm import *
from module.load_data import *

os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

def get_url_string(fname_list_1:str, fname_list_2:str, image_base_dir:str='dataset/cifar10_randomed/images'):
    url_1_list, url_2_list = [], []
    for idx, fname in tqdm.tqdm(enumerate(fname_list_1)):
        url_1 = pixtral_preprocessing(
            os.path.join(image_base_dir, fname_list_1[idx])
        )
        
        url_2 = pixtral_preprocessing(
            os.path.join(image_base_dir, fname_list_2[idx])
        )
        
        url_1_list.append(url_1)
        url_2_list.append(url_2)
        
    return url_1_list, url_2_list

def main():
    model, sampling_params = load_model(model_name='mistralai/Pixtral-12B-2409')
    
    prompt = 'Describe the person you saw from the given images especially from hair, face, and nose. Tell only the differences or similarity without any chat'
    
    dataset_main_path = 'formatted_dataset/face_recognition/'

    pair_1_fname_train, pair_2_fname_train, label_pairing_train, label_anchor_train, label_non_anchor_train = load_npz(dataset_main_path+"train.npz")
    pair_1_fname_test, pair_2_fname_test, label_pairing_test, label_anchor_test, label_non_anchor_test = load_npz(dataset_main_path+"test.npz")
    
    pair_1_url_train, pair_2_url_train = get_url_string(
        fname_list_1=pair_1_fname_train,
        fname_list_2=pair_2_fname_train
    )

    pair_1_url_test, pair_2_url_test = get_url_string(
        fname_list_1=pair_1_fname_test,
        fname_list_2=pair_2_fname_test
    )
    
    pair_text_train = pixtral_inference(
        llm=model,
        sampling_params=sampling_params,
        image_url_1=pair_1_url_train,
        image_url_2=pair_2_url_train,
        prompt=prompt
    )

    # Save into json based on fname
    export_generated_text_json(
        generated_text=pair_text_train,
        fname_list_pair_1=pair_1_fname_train, 
        fname_list_pair_2=pair_2_fname_train,
        target_dir='formatted_dataset/face_recognition/text'
    )
    
    pair_text_test = pixtral_inference(
        llm=model,
        sampling_params=sampling_params,
        image_url_1=pair_1_url_test,
        image_url_2=pair_2_url_test,
        prompt=prompt
    )

    export_generated_text_json(
        generated_text=pair_text_test,
        fname_list_pair_1=pair_1_fname_test,
        fname_list_pair_2=pair_2_fname_test,
        target_dir='formatted_dataset/face_recognition/text'
    )

if __name__ == "__main__":
    main()