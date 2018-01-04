'''

This code has three parts:
    1. Frame extractor
    2. URL downloader (youtube API)
    3. Video downloader

'''



# 1. Frame extractor 
'''
1. Input the path of the folder: "Class_x_classname" eg: Class_3_RPG
        "os.chdir("..."
2. Define the frame rate
        "frameRate = 25"
3. Set the frame resolution (otherwise video resolution)
        "frame = cv2.resize(frame, (224, 224)..."
4. Run
'''


import pandas as pd
import pytube
import cv2
import math
import os

os.chdir("C:/Users/tatar/Desktop/DL Project/Data_set_2/Class_3_RPG") # path of the folder with video folders
class_folder_path = os.getcwd().replace("\\", "/")
class_folder_strucutre = os.listdir(class_folder_path) 

for folder in class_folder_strucutre:
    video_folder = str(class_folder_path)+ "/" + str(folder) # path of the folder with videos
    filelist_video_folder = os.listdir(video_folder) # list of the folders with videos
    plik=0 # for naming
    video_folder_name = str(folder) + '_frames'
    os.makedirs(video_folder_name) #creates frame folder
     
    for video in filelist_video_folder:
        try:
            
            plik = plik+1 #just for naming
            print("plik:", plik)
            videoFile = video_folder + "/"+ str(video) #path to find videos in video folder
            videoFile2 = video_folder_name + "/"+ str(video) # path to save frames in frame folder
            FolderName = str(videoFile2) # name of the frame subfolder
            os.makedirs(FolderName)
            
            imagesFolder = FolderName # where to save frames
            cap = cv2.VideoCapture(videoFile)
            frameRate = 25 # frame rate
            while(cap.isOpened()):
                frameId = cap.get(1) #current frame number
                ret, frame = cap.read()
                frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
                #frame = cv2.resize(frame, (480, 270)) 
                if (ret != True):
                    break
                if (frameId % math.floor(frameRate) == 0):
                    filename = imagesFolder + "/frame_" + str(folder) + "_" + str(int(frameId)) + ".jpg"
                    cv2.imwrite(video_folder_name, frame)
            cap.release()
            print("6")
        except:
            pass







# 2. URL downloader (youtube API)
'''
1. Set DEVELOPER_KEY to the API key value from the APIs & auth > Registered apps
tab of https://cloud.google.com/console
2. Please ensure that you have enabled the YouTube Data API for your project.         
3. In "search_response" argument maxResults defines the number of URL per search query.
4. Execute youtube_search(" ") once per search query. All URLs saved in one Excel table anyway.

'''
from apiclient.discovery import build
# from apiclient.errors import HttpError
# from oauth2client.tools import argparser # removed by Dongho
import argparse
import csv
import unidecode


DEVELOPER_KEY = "AIzaSyB7WefKy4DRAB0az9mbBiK2LQrAVfOgNTI" #personal developer key
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
CLASS_NAME = "rpg"         #used for excel table name
#option = excel_table

def youtube_search(search_query):
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    # Call the search.list method to retrieve results matching the specified
    # query term.
    search_response = youtube.search().list(q=search_query, order="viewCount", type="video", part="id,snippet", maxResults=10, videoDuration="medium").execute()
    
    videos = []
    channels = []
    playlists = []
    # create a CSV output for video list    
    csvFile = open(str(CLASS_NAME) + '.csv', 'a')
    csvWriter = csv.writer(csvFile)
    # define the table content
    csvWriter.writerow(["title", "videoId", "viewCount", "likeCount", "dislikeCount", "commentCount", "favoriteCount", ""])

    # Add each result to the appropriate list, and then display the lists of
    # matching videos, channels, and playlists.
    for search_result in search_response.get("items", []):
        if search_result["id"]["kind"] == "youtube#video":
            # videos.append("%s (%s)" % (search_result["snippet"]["title"],search_result["id"]["videoId"]))
            title = search_result["snippet"]["title"]
            title = unidecode.unidecode(title)  # Dongho 08/10/16
            videoId = search_result["id"]["videoId"]
            video_response = youtube.videos().list(id=videoId, part="statistics").execute()
            for video_result in video_response.get("items", []):
                viewCount = video_result["statistics"]["viewCount"]
                if 'likeCount' not in video_result["statistics"]:
                    likeCount = 0
                else:
                    likeCount = video_result["statistics"]["likeCount"]
                if 'dislikeCount' not in video_result["statistics"]:
                    dislikeCount = 0
                else:
                    dislikeCount = video_result["statistics"]["dislikeCount"]
                if 'commentCount' not in video_result["statistics"]:
                    commentCount = 0
                else:
                    commentCount = video_result["statistics"]["commentCount"]
                if 'favoriteCount' not in video_result["statistics"]:
                    favoriteCount = 0
                else:
                    favoriteCount = video_result["statistics"]["favoriteCount"]

            csvWriter.writerow([title, videoId, viewCount, likeCount, dislikeCount, commentCount, favoriteCount])

    csvFile.close()

youtube_search("Elder Scrolls gameplay")
youtube_search("Witcher 3 gameplay")
youtube_search("Fallout gameplay")
youtube_search("World of Warcraft gameplay")
youtube_search("New Vegas gameplay")
youtube_search("Diablo III gameplay")
youtube_search("Shadow of mordor gameplay")
youtube_search("Dark Souls gameplay")
youtube_search("Dragon Age gameplay")
youtube_search("mass effect gameplay")








# 3. Video downloader     
'''
1. Insert link to the Excel table "ytlist=pd.read_csv(..."
2. Define the range of rows in the excel table "ytlist=ytlist.iloc[0:228,:]"
3. Define the target folder for videos "vids[1].download(r..."
'''

ytlist=pd.read_csv('C:/Users/tatar/Desktop/DL Project/rpg1.csv',header= None)
ytlist=ytlist.iloc[0:228,:]
xname=0
for i in ytlist[2]:
    try:
        xname+=1
        print("1")
        yt = pytube.YouTube(i)
        print("2")
        #yt.set_filename('Dancing')
        print("2.5")
        vids= yt.streams.all()
        print("3")
        name="file" + str(xname)
        #vids[1].download(r"C:/Users/tatar/Desktop/DL Project/ytdown", filename=name)
        vids[1].download(r"C:/Users/tatar/Desktop/DL Project/Data_set_2/Class_3_RPG")
        print("4")
        #yt.register_on_complete_callback(frames())
        print("5")
        '''get frames'''

    except:
        pass 
  
