from flask_socketio import SocketIO
from flask import render_template
from mtcnn import MTCNN
import os
import cv2
from flask import Flask, request
import numpy as np
from werkzeug.utils import secure_filename
import test
import base64


app = Flask(__name__)
UPLOAD_FOLDER = './static/upload/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
HEADER = ["Emozione rilevata", "Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness"]
FRAME_ANALYZED_PER_SECOND = 1
CURRENT_VIDEO_PATH = ""

frame_stats = []
detector = MTCNN() 

app.config['SECRET_KEY'] = 'secret!'
sio = SocketIO(app, async_mode='threading')



#Detect del viso e eventuale cropping. Se nessuna faccia viene trovata,
#retrun False
#TODO: Sistemare la percentuale del cropping
def crop_image(img):
    global UPLOAD_FOLDER
    global detector
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height = int((grey.shape[0] * 10) / 100)
    width = int((grey.shape[1] * 10) / 100)
    data=detector.detect_faces(grey)
    biggest=0
    if data != []:
        for faces in data:
            #Aggiungo "Margine" per cropping della foto
            box=faces['box']
            faces['box'][0] = faces['box'][0] - height
            faces['box'][1] = faces['box'][1] - width   
            faces['box'][2] = faces['box'][2] + height * 2
            faces['box'][3] = faces['box'][3] + width * 2            
            #Calcolo Area immagine
            area = (box[3] * box[2])*2
            if area>biggest:
                biggest=area
                bbox=box 
        bbox[0]= 0 if bbox[0]<0 else bbox[0]
        bbox[1]= 0 if bbox[1]<0 else bbox[1]
        crop = grey[bbox[1]: bbox[1]+bbox[3],bbox[0]: bbox[0]+ bbox[2]]
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR) 
        cv2.imwrite(UPLOAD_FOLDER + "cropped.jpg", rgb_crop)
        return True
    else:
        return False



#Route di caricamento delle foto o video da analizzare. Effettua un salvataggio
#in locale della foto o video
@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    global HEADER
    global frame_stats
    global CURRENT_VIDEO_PATH
    if request.method == 'POST':
        file = request.files['file']
        if file.filename.endswith('.mp4'):
            #Se il file è un video
            filename = secure_filename(file.filename)
            videopath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(videopath)
            CURRENT_VIDEO_PATH = str(videopath)
            return render_template('video.html', videopath = CURRENT_VIDEO_PATH[1:], FAS = FRAME_ANALYZED_PER_SECOND)
        elif file.filename.endswith('.jpg') or file.filename.endswith('.png' ) or file.filename.endswith('.jpeg'):
            #Se il file è un'immagine
            filename = secure_filename(file.filename)
            imgpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(imgpath)
            img = cv2.imread(imgpath)
            if crop_image(img):
                frame_stats = test.startanalysis(UPLOAD_FOLDER + "cropped.jpg").split("|")
            else:
                return render_template('uploader.html', flag = 1)
            return render_template('results.html', results = frame_stats, filename = filename, header = HEADER, index = range(len(frame_stats)-1))
    return render_template('uploader.html', flag = 0)

#Pagina principale del sito
@app.route('/')
def hello():
    return render_template('home.html')


#Restituisce la pagina html che ospita l'analisi delle emozioni
#utilizzando la webcam
@app.route('/webcam')
def webcam():
    return render_template('webcam.html', FAS = FRAME_ANALYZED_PER_SECOND)


#Aggiornamento delle preferenze nella sezione "Impostazioni"
@app.route('/settings', methods=['GET', 'POST'] )
def settings():
    return render_template('settings.html', FAS = FRAME_ANALYZED_PER_SECOND)
   

#Aggiornamento delle preferenze nella sezione "Impostazioni"
#Utilizzato in settigns.js con una chiamata Ajax
@app.route('/settings-RU', methods=['GET', 'POST'] )
def settings_RU():
    global FRAME_ANALYZED_PER_SECOND
    if request.method == "GET":
        settings = FRAME_ANALYZED_PER_SECOND
        return settings
    elif request.method == "POST":
        json = request.get_json()
        FRAME_ANALYZED_PER_SECOND = json['fps']
        return "True"

@sio.event
def connect():
    print('Client connesso')

@sio.event
def disconnect():
    print('Client disconnesso')

@sio.on('frame')
def frame(encoded_frame):
    global frame_stats
    bin_data = base64.b64decode(encoded_frame)
    image = np.asarray(bytearray(bin_data), dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    if crop_image(img):
        frame_stats = test.startanalysis(UPLOAD_FOLDER + "cropped.jpg")
    sio.emit("results", frame_stats)
