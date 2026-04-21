import os
import pandas as pd
import numpy as np
from pathlib import Path
import glob

raw_data_folder = r"archive\raw_data" 
new_data_folder = r"data\bme"

gesture_labels= {
    'unknown':  0,
    'rest':     1,
    'fist':     2,
    'flex':     3,
    'extend':   4,
    'radial':   5,
    'ulnar':    6,
    'pro':      7,
    'sup':      8,
    '2f':       9,
    '3f':       10,
}

def main(write_filtered=True):
    failed_writes = []
    folder_list = glob.glob(os.path.join(raw_data_folder, "*/"))
    for folder in folder_list:
        new_folder = os.path.join(new_data_folder,Path(folder).name)
        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)
        file_list = glob.glob(os.path.join(folder, "*.csv"))
        for file in file_list:
            filename = Path(file).name            
            print(f"------- Processing {filename} -------")
            data_info = filename.split('_')
            gesture = data_info[1].lower()
            if gesture == "rest.csv":
                gesture = "rest"
                interval = 30
            else:
                interval = int(data_info[2][0])
            label = gesture_labels[gesture]

            df = pd.read_csv(file, sep='\t')
            df = df.drop(columns=["GyroX","GyroY","GyroZ",
                             "AccX","AccY","AccZ",
                             "PPG1","PPG2","rawPPG1","rawPPG2","rawPPG3",
                             "Hr","Hrv","Battery",
                             "Trigger","PhysicalTrigger","AutoTrigger"])
            if write_filtered:
                df = df.drop(columns=["Channel1","Channel2","Channel3","Channel4",
                                    "Channel5","Channel6","Channel7","Channel8"])
            else:
                df = df.drop(columns=["FilteredChannel1","FilteredChannel2","FilteredChannel3","FilteredChannel4",
                                    "FilteredChannel5","FilteredChannel6","FilteredChannel7","FilteredChannel8"])
            
            print(f"Gesture :    {gesture}")
            print(f"Label   :    {label}")
            print(f"Interval:    {interval}")
            df = annotate_emg(df, label, interval) 
            new_file = os.path.join(new_folder, filename)
            try:
                df.to_csv(new_file, index=False)
                print(f"Successfully wrote {new_file}")
            except PermissionError as e:
                print(f"Permission denied writing {new_file}: {e}")
                failed_writes.append(filename)
            except FileNotFoundError as e:
                print(f"Directory not found for {new_file}: {e}")
                failed_writes.append(filename)
            except OSError as e:
                print(f"OS error writing {new_file}: {e}")
                failed_writes.append(filename)
            except Exception as e:
                print(f"Unexpected error writing {new_file}: {type(e).__name__}: {e}")
                failed_writes.append(filename)
            # if gesture == 'rest':


def annotate_emg(df, label, interval=3):
    n = len(df)
    if label==1:
        df['label'] = [1]*n
        return df
    else:
        rep_cnt = interval*500
        reps = (n//(rep_cnt*2))+1
        df['label'] = np.tile(np.repeat([0,label], rep_cnt), reps=reps)[:n]
        return df

if __name__ == "__main__":
    main(write_filtered=True)  
            