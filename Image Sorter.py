import face_recognition
import numpy as np
import pickle
from PIL import Image as im
import os
import shutil

pname = "Your_Name"

print(f"[INFO] loading encodings for {pname}...")

with open(f"{pname}.p", "rb") as f:
    known_face_encodings = pickle.loads(f.read())
f.close()

shift_list = []

def process_frame(frame, index):
    face_locations = []
    face_encodings = []
    global shift_list
    
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations, model='large')
    
    for face_encoding in face_encodings:
        
        face_distances = list(face_recognition.face_distance(known_face_encodings, face_encoding))
        
        max = np.max(face_distances)
        avg = np.average(face_distances)
        
        face_distances.sort()
        
        if ((face_distances[0]+face_distances[1]+face_distances[2]+face_distances[3])/4 < 0.425 and max < 0.95 and avg < 0.5):
            shift_list.append(index)

dir = 'Your_Directory'

def tree(path):
    addr = []     
    
    for root, dirs, files in os.walk(path):
        if files != []:
            for img in files:
                if '.jpg' in img or '.jpeg' in img or '.png' in img:      
                    addr.append(os.path.join(root,img)) 
    
    return addr

print('[INFO] "Walking" the file structure')

img_list = tree(dir)

if len(img_list) != 0:

    print(f'[INFO] found {len(img_list)} images')

    for index, image in enumerate((img_list)):
        
        print(f'[INFO] Scanning Image {index+1}/{len(img_list)}')
        frame = np.asarray(im.open(image))
        
        try:
            process_frame(frame,index)
            
        except Exception as e:
            
            print(f'[INFO] Skipping Image {index + 1} check Logs')
            
            with open('Logs.txt','a') as f:
                f.write(f"Can't Process {image} - {e}\n")
            f.close()
    
    if len(shift_list) != 0:
    
        print(f"[INFO] Scanning Complete, found {len(shift_list)} matches")

        save_dir = f'D:\\Desktop\\{pname}'

        print(f'[INFO] Moving images to "{save_dir}"')

        for id, src in enumerate(shift_list):
            
            print(f'[INFO] Moving image {id + 1}/{len(shift_list)}')
            
            try:
                if os.path.exists(save_dir):
                    pass
                else:
                    os.mkdir(save_dir)
                shutil.move(img_list[src],save_dir)
            
            except Exception as e:
            
                print(f"[INFO] Can't  move Image {id + 1} check Logs")
            
                with open('Logs.txt','a') as f:
                    f.write(f"[INFO] Can't move {img_list[src]} - {e}\n")
                f.close()                
    
    else:
        print("[INFO] Scanning Complete, 0 matches found")

else:
    print('[INFO] No Images found')

print('[INFO] sorting Completed')
