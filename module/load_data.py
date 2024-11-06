import os
import cv2
import json
import numpy as np

def load_npz(path:str):
    loaded_data = np.load(path)
    
    pair_1, pair_2 = loaded_data['pair_1'], loaded_data['pair_2']
    label_pairing = loaded_data['label_pairing']
    label_anchor, label_non_anchor = loaded_data['label_anchor'], loaded_data['label_non_anchor']
    
    return pair_1, pair_2, label_pairing, label_anchor, label_non_anchor

def load_image_pair(pair_1, pair_2):
    image_pair_1 = cv2.imread(pair_1)
    image_pair_2 = cv2.imread(pair_2)
    
    return image_pair_1, image_pair_2

def export_generated_text_json(
    generated_texts:list, fname_list_pair_1:list, fname_list_pair_2:list, target_dir:str='formatted_dataset/'
):
    for idx, text in enumerate(generated_texts):
        json_scheme = {
            "pair_1" : fname_list_pair_1[idx],
            "pair_2" : fname_list_pair_2[idx],
            "text" : text
        }
        
        # Construct filename
        saving_path = os.path.join(target_dir, fname_list_pair_1[idx].replace("_pair_1", ""))
        with open(saving_path, 'w') as f:
            json.dump(json_scheme, f, indent=4)
        