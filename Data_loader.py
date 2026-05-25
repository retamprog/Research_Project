import pandas as pd

# just loading the ASVspoof 2019 cm protocols for the training dataset

def protocol_loader(path):
    data =[]
    with open(path,'r') as f:
        for line in f:
            parts = line.strip().split()
            
            # speaker_id,Audio_file_name,-,Systemid,key
            speaker_id=parts[0]
            audio_file_id=parts[1]
            attack_id=parts[3]
            label_str=parts[4]
            label=0 if label_str=='bonafide' else 1

            data.append({"speaker_id":speaker_id,
                          "audio_id":audio_file_id,
                          "attack_id":attack_id,
                          "label_str":label_str,
                          "label":label})
    
    return pd.DataFrame(data)






