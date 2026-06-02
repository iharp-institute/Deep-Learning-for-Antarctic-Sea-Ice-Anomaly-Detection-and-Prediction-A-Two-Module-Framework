import glob
import os
import cv2
from PIL import Image
# Path to the directory or subdirectory


lastyear = 2025
for year in range(2024,lastyear,1):
    path = str(year)+"/**/*"

    # Using glob to get all files recursively
    all_files = [file for file in glob.glob(path, recursive=True) if not os.path.isdir(file)]

    # Print the list of files
    for file in all_files:
        print(file)

        monthfromfile = file[0:12]
        
        

        
        save_fileame = file[:-14]
        save_fileame = save_fileame[14:]
        print(save_fileame)
        
        
        
        image = cv2.imread(file)  # Replace with the path to your image

    # Define the crop region using the provided values
        width = 318
        height = 334
        offset_x = 11
        offset_y = 87

    # Crop the image using the specified region
        cropped_image = image[offset_y:offset_y + height, offset_x:offset_x + width]
        cropped_image = cropped_image[1:-1, 1:-1]
        cv2.imwrite(monthfromfile+save_fileame+'.png', cropped_image)

        os.remove(file)



    print(str(year)+"  Conversion completed!")
