
import requests
from bs4 import BeautifulSoup
import os
import glob
import os
import cv2
from PIL import Image


for lll in range(2024,2025,1):

    list_month = [
        "01_Jan",
        "02_Feb"
        
        # "03_Mar",
        # "04_Apr",
        # "05_May",
        # "06_Jun",
        # "07_Jul"
        
    ]

    os.makedirs(str(lll))
    for month in list_month:
        os.makedirs(str(lll)+'/'+month)
        url="https://noaadata.apps.nsidc.org/NOAA/G02135/south/daily/images/"+str(lll)+'/'+ month +'/'
        archive_url = url
        r = requests.get(archive_url)
        data = BeautifulSoup(r.text, "html.parser")
        for l in data.find_all("a")[1:]:
            	
            r = requests.get(archive_url + l["href"])
            	
            file_path = os.path.join(str(lll)+'/'+month+'/', l["href"])
            
            
               
            if (file_path[-14:] == '_conc_v3.0.png'):
                
        
                with open(file_path, "wb") as f:
                    #print(r.content)
                    f.write(r.content)
                    
               
                
                
        
