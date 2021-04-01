from __future__ import division
from flask import Flask, render_template,request
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import zipfile
from flask import send_file
from flask import send_from_directory
import cv2
import glob
import pydicom
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageChops
from PIL import Image
import numpy as np
import csv

import pandas as pd
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers


UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/uploads/'
DOWNLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/downloads/'
CROP_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/crops/'    
TEST_FOLDER = os.path.dirname(os.path.abspath(__file__)) + '/tests/'
RESULT_FOLDER=os.path.dirname(os.path.abspath(__file__)) + '/results_imgs'
ADB_FOLDER=os.path.dirname(os.path.abspath(__file__)) + '/ADB_result'
EPD_FOLDER=os.path.dirname(os.path.abspath(__file__)) + '/EPD_result' 
SPD_FOLDER=os.path.dirname(os.path.abspath(__file__)) + '/SPD_result' 
SS_FOLDER=os.path.dirname(os.path.abspath(__file__)) + '/SS_result' 
GATHER_FOLDER=os.path.dirname(os.path.abspath(__file__)) + '/gather'  
ALLOWED_EXTENSIONS = {'ima','jpg'}


app=Flask(__name__)
api = Api(app)


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['CROP_FOLDER'] = CROP_FOLDER
app.config['TEST_FOLDER']=TEST_FOLDER
app.config['RESULT_FOLDER']=RESULT_FOLDER
app.config['ADB_FOLDER']=ADB_FOLDER
app.config['EPD_FOLDER']=EPD_FOLDER
app.config['SPD_FOLDER']=SPD_FOLDER
app.config['SS_FOLDER']=SS_FOLDER
app.config['GATHER_FOLDER']=GATHER_FOLDER

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


#Only_Convert
@app.route('/D2J', methods=['GET', 'POST'])
def D2J():
    for root,dirnames, filenames in os.walk('uploads/'):
            for filename in filenames:
                filename=filename
                delete_uploads(os.path.join(app.config['UPLOAD_FOLDER'],filename), filename)
    for root,dirnames, filenames in os.walk('downloads/'):
            for filename in filenames:
                filename=filename
                delete_downloads(os.path.join(app.config['DOWNLOAD_FOLDER'],filename), filename)
    if request.method == 'POST':

        if 'files[]' not in request.files:
            print('No file attached in request')
            return redirect(request.url)

        files = request.files.getlist('files[]')

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                file.save(os.path.join(app.config['GATHER_FOLDER'], filename))
                process_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename)
        
        for root,dirnames, filenames in os.walk('downloads/'):
            for filename in filenames:
                filename=filename
                save_(os.path.join(app.config['DOWNLOAD_FOLDER'],filename), filename)

        return redirect(url_for('uploaded_file'))
        return render_template("convert.html")

@app.route('/downloads')
def uploaded_file():
    zipf = zipfile.ZipFile('DicomToJpeg.zip','w', zipfile.ZIP_DEFLATED)
    for root,dirs, files in os.walk('downloads/'):
        for file in files:
            zipf.write('downloads/'+file)
    zipf.close()
    for root,dirnames, filenames in os.walk('downloads/'):
            for filename in filenames:
                filename=filename
                delete_downloads(os.path.join(app.config['DOWNLOAD_FOLDER'],filename), filename)
    return send_file('DicomToJpeg.zip',
            mimetype = 'zip',
            attachment_filename= 'DicomToJpeg.zip',
            as_attachment = True)
    return send_from_directory(app.config['DOWNLOAD_FOLDER'],
                               filename=filename + '.jpg', as_attachment=True)
#Only_Convert

@app.route('/result_downloads', methods=['GET', 'POST'])
def result_file():
    zipf = zipfile.ZipFile('Result.zip','w', zipfile.ZIP_DEFLATED)
    for root,dirs, files in os.walk('results_imgs/'):
        for file in files:
            zipf.write('results_imgs/'+file)
    zipf.close()
    return send_file('Result.zip',
            mimetype = 'zip',
            attachment_filename= 'Result.zip',
            as_attachment = True)
    return send_from_directory(app.config['RESULT_FOLDER'],
                               filename=filename + '.jpg', as_attachment=True)














@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response




#testing phase
@app.route('/t', methods=['GET', 'POST'])
def t():
    for root,dirnames, filenames in os.walk('uploads/'):
            for filename in filenames:
                filename=filename
                delete_uploads(os.path.join(app.config['UPLOAD_FOLDER'],filename), filename)
    for root,dirnames, filenames in os.walk('downloads/'):
            for filename in filenames:
                filename=filename
                delete_downloads(os.path.join(app.config['DOWNLOAD_FOLDER'],filename), filename)
    for root,dirnames, filenames in os.walk('crops/'):
            for filename in filenames:
                filename=filename
                delete_crops(os.path.join(app.config['CROP_FOLDER'],filename), filename)
    for root,dirnames, filenames in os.walk('ADB_result/'):
            for filename in filenames:
                filename=filename
                delete_ADB(os.path.join(app.config['ADB_FOLDER'],filename), filename)
    for root,dirnames, filenames in os.walk('EPD_result/'):
            for filename in filenames:
                filename=filename
                delete_EPD(os.path.join(app.config['EPD_FOLDER'],filename), filename)
    for root,dirnames, filenames in os.walk('SPD_result/'):
            for filename in filenames:
                filename=filename
                delete_SPD(os.path.join(app.config['SPD_FOLDER'],filename), filename)
    for root,dirnames, filenames in os.walk('SS_result/'):
            for filename in filenames:
                filename=filename
                delete_SS(os.path.join(app.config['SS_FOLDER'],filename), filename)
    for root,dirnames, filenames in os.walk('results_imgs/'):
            for filename in filenames:
                filename=filename
                delete_results(os.path.join(app.config['RESULT_FOLDER'],filename), filename)


    if request.method == 'POST':

        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)

        file = request.files['file']

        #for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save(os.path.join(app.config['GATHER_FOLDER'], filename))
            process_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename)

        for root,dirnames, filenames in os.walk('downloads/'):
            for filename in filenames:
                filename=filename
                save_(os.path.join(app.config['DOWNLOAD_FOLDER'],filename), filename)

        return redirect('/model')
        return render_template("test.html")


def process_file(path, filename):
    img = pydicom.read_file(open(path,'rb'))
    matplotlib.image.imsave(app.config['DOWNLOAD_FOLDER'] + filename + '.jpg' ,img.pixel_array,cmap=plt.cm.bone)

#Dicom to jpg end 

#Auto Crop Normalize
def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 1.0, -100)
    #Bounding box given as a 4-tuple defining the left, upper, right, and lower pixel coordinates.
    #If the image is completely empty, this method returns None.
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def save_(path, filename):
    im = Image.open(path,mode='r')
    img = im.crop((65,1,320,384))
    new_img = trim(img)
    norm = (new_img - np.min(new_img)) / (np.max(new_img) - np.min(new_img))
    matplotlib.image.imsave(app.config['CROP_FOLDER'] + filename ,norm,cmap=plt.cm.bone)

    for root,dirnames, filenames in os.walk('crops/'):
            for f in filenames:
                global req
                req=f

#Auto Crop Normalize
def delete_uploads(path, filename):
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
def delete_downloads(path, filename):
    os.remove(os.path.join(app.config['DOWNLOAD_FOLDER'], filename))
def delete_crops(path, filename):
    os.remove(os.path.join(app.config['CROP_FOLDER'], filename))
def delete_ADB(path, filename):
    os.remove(os.path.join(app.config['ADB_FOLDER'], filename))
def delete_EPD(path, filename):
    os.remove(os.path.join(app.config['EPD_FOLDER'], filename))
def delete_SPD(path, filename):
    os.remove(os.path.join(app.config['SPD_FOLDER'], filename))
def delete_SS(path, filename):
    os.remove(os.path.join(app.config['SS_FOLDER'], filename))
def delete_results(path, filename):
    os.remove(os.path.join(app.config['RESULT_FOLDER'], filename))



@app.route('/reportt',methods=['GET', 'POST'])
def reportt():
    #retrieve(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename)
    for root,dirnames, filenames in os.walk('uploads/'):
            for filename in filenames:
                filename=filename
                path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
                img=pydicom.read_file(open(path,'rb'))
                print("  PATIENT'S DATA")
                print('ID           =',102)
                print('Size         =',img.PatientSize)
                print('Age          =',img.PatientAge.replace('0',''))
                print('Sex          =',img.PatientSex.replace('M','MALE'))
                print('Weight       =',str(img.PatientWeight) + ' KG')
                print('Examined Part=',img.BodyPartExamined.replace('L','LUMBAR'))
                I=filename.replace('T2_TSE_SAG__','').replace('.ima','').replace('_008','').replace('_007','').replace('_009','')
                PS=img.PatientSize
                A=img.PatientAge.replace('0','')
                S=img.PatientSex.replace('M','MALE')
                W=str(img.PatientWeight) + ' KG'
                EP=img.BodyPartExamined.replace('L','LUMBAR')
                for root,dirnames, filenames in os.walk('results_imgs/'):
                        for f in filenames:
                            file=f
                #retrieve(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename)
                #img=img.PatientSex
    #img = pydicom.read_file(open(os.path.join(app.config['UPLOAD_FOLDER'], filename), filename))
    return render_template("report.html",PS=PS,A=A,S=S,W=W,EP=EP,I=I,user_image = file)

@app.route('/results_imgs/<path:filename>')
def show_file(filename):
    for root,dirnames, filenames in os.walk('results_imgs/'):
                        for f in filenames:
                            file=f
    return send_from_directory(RESULT_FOLDER, filename, as_attachment=True)



#def retrieve(path,filename):
 #   img = pydicom.read_file(open(path,'rb'))
  #  display_info(img)
   # PS=img.PatientSize
    #return PS 
#def display_info(img): #tere kaam ki
 #   print("  PATIENT'S DATA")
  #  print('ID           =',102)
   # print('Size         =',img.PatientSize)
    #print('Age          =',img.PatientAge.replace('0',''))
    #print('Sex          =',img.PatientSex.replace('M','MALE'))
    #print('Weight       =',str(img.PatientWeight) + ' KG')
    #print('Examined Part=',img.BodyPartExamined.replace('L','LUMBAR'))



#test_frcnn
@app.route('/model')
def model():

    sys.setrecursionlimit(40000)

    config_output_filename = "config.pickle"

    with open(config_output_filename, 'rb') as f_in:
        C = pickle.load(f_in)
        K.clear_session()

    if C.network == 'resnet50':
        import keras_frcnn.resnet as nn
    elif C.network == 'vgg':
        import keras_frcnn.vgg as nn

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False

    img_path = "crops"

    def format_img_size(img, C):
        """ formats the image size based on config """
        img_min_side = float(C.im_size)
        (height,width,_) = img.shape
            
        if width <= height:
            ratio = img_min_side/width
            new_height = int(ratio * height)
            new_width = int(img_min_side)
        else:
            ratio = img_min_side/height
            new_width = int(ratio * width)
            new_height = int(img_min_side)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return img, ratio   

    def format_img_channels(img, C):
        """ formats the image channels based on config """
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.float32)
        img[:, :, 0] -= C.img_channel_mean[0]
        img[:, :, 1] -= C.img_channel_mean[1]
        img[:, :, 2] -= C.img_channel_mean[2]
        img /= C.img_scaling_factor
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def format_img(img, C):
        """ formats an image for model prediction based on config """
        img, ratio = format_img_size(img, C)
        img = format_img_channels(img, C)
        return img, ratio

    # Method to transform the coordinates of the bounding box to its original size
    def get_real_coordinates(ratio, x1, y1, x2, y2):

        real_x1 = int(round(x1 // ratio))
        real_y1 = int(round(y1 // ratio))
        real_x2 = int(round(x2 // ratio))
        real_y2 = int(round(y2 // ratio))

        return (real_x1, real_y1, real_x2 ,real_y2)

    #class_mapping = C.class_mapping

    #if 'bg' not in class_mapping:
     #   class_mapping['bg'] = len(class_mapping)

    #class_mapping = {v: k for k, v in class_mapping.items()}
    #print(class_mapping)
    #class_to_color = {class_mapping[v]: np.random.randint(0,255,3) for v in class_mapping}

    C.num_rois = 32

    if C.network == 'resnet50':
        num_features = 1024
    elif C.network == 'vgg':
        num_features = 512

    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
        input_shape_features = (num_features, None, None)
    else:
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, num_features)


    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    #classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)
    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=3, trainable=True)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    disease_bbx_list = []
    picture = None
    ## MULTIPLE MODELS
    cls_map_SPD = {'SPD': 0, 'OK': 1, 'bg': 2}
    cls_map_EPD = {'OK': 0, 'EPD': 1, 'bg': 2}
    cls_map_SS = {'N': 0, 'SS': 1, 'bg': 2}
    cls_map_ADB = {'N': 0, 'ADB': 1, 'bg': 2}

    sve_loc_list = ['./SPD_result/','./SS_result/','./EPD_result/','./ADB_result/']


    wght_loc_list = ['./Weights/SS/model_frcnn.hdf5','./Weights/SS/model_frcnn.hdf5','./Weights/SS/model_frcnn.hdf5','./Weights/SS/model_frcnn.hdf5']
    for w_path, sv_loc in zip(wght_loc_list,sve_loc_list):

        if 'SPD' in sv_loc:
            class_mapping = cls_map_SPD
        elif 'EPD' in sv_loc:
            class_mapping = cls_map_EPD
        elif 'SS' in sv_loc:
            class_mapping = cls_map_SS
        elif 'ADB' in sv_loc:
            class_mapping = cls_map_ADB
        else:
            print("Classes not found in config.pickle")
            break

        if 'bg' not in class_mapping:
            class_mapping['bg'] = len(class_mapping)

        class_mapping = {v: k for k, v in class_mapping.items()}
        print()
        print(class_mapping)
        #class_to_color = {class_mapping[v]: np.random.randint(0,255,3) for v in class_mapping}

        #print('Loading weights from {}'.format(C.model_path))
        print('Loading weights from {}'.format(w_path))
        print()

        #model_rpn.load_weights(C.model_path, by_name=True)
        model_rpn.load_weights(w_path, by_name=True)

        #model_classifier.load_weights(C.model_path, by_name=True)
        model_classifier.load_weights(w_path, by_name=True)


        model_rpn.compile(optimizer='sgd', loss='mse')
        model_classifier.compile(optimizer='sgd', loss='mse')

        all_imgs = []

        classes = {}

        bbox_threshold = 0.8

        visualise = True
        img_name_list = []
        lists = []
        thic = 2
        box_color = (0, 0, 255)
        label_color = (255, 255, 255)

        for idx, img_name in enumerate(sorted(os.listdir(img_path))):
            if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            print(img_name)
            img_name_list.append(img_name)
            #print(img_name_list)   
            st = time.time()
            filepath = os.path.join(img_path,img_name)

            img = cv2.imread(filepath)

            X, ratio = format_img(img, C)

            if K.image_dim_ordering() == 'tf':
                X = np.transpose(X, (0, 2, 3, 1))

            # get the feature maps and output from the RPN
            [Y1, Y2, F] = model_rpn.predict(X)
            

            R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

            # convert from (x1,y1,x2,y2) to (x,y,w,h)
            R[:, 2] -= R[:, 0]
            R[:, 3] -= R[:, 1]

            # apply the spatial pyramid pooling to the proposed regions
            bboxes = {}
            probs = {}

            for jk in range(R.shape[0]//C.num_rois + 1):
                ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
                if ROIs.shape[1] == 0:
                    break

                if jk == R.shape[0]//C.num_rois:
                    #pad R
                    curr_shape = ROIs.shape
                    target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
                    ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                    ROIs_padded[:, :curr_shape[1], :] = ROIs
                    ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                    ROIs = ROIs_padded

                [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

                for ii in range(P_cls.shape[1]):

                    if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                        continue

                    cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                    if cls_name not in bboxes:
                        bboxes[cls_name] = []
                        probs[cls_name] = []

                    (x, y, w, h) = ROIs[0, ii, :]

                    cls_num = np.argmax(P_cls[0, ii, :])
                    try:
                        (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                        tx /= C.classifier_regr_std[0]
                        ty /= C.classifier_regr_std[1]
                        tw /= C.classifier_regr_std[2]
                        th /= C.classifier_regr_std[3]
                        x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                    except:
                        pass
                    bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                    probs[cls_name].append(np.max(P_cls[0, ii, :]))

            all_dets = []
            
            #i = 0

            for key in bboxes:
                bbox = np.array(bboxes[key])

                new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
                for jk in range(new_boxes.shape[0]):
                    (x1, y1, x2, y2) = new_boxes[jk,:]

                    (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
                    #bbx_df = pd.DataFrame((real_x1, real_y1, real_x2, real_y2))
                

                    #print("X1 ",real_x1)
                    #print("Y1 ",real_y1)
                    #print("X2 ",real_x2)
                    #print("Y2 ",real_y2)

                    ##cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),2)

                    textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
                    ##all_dets.append((key,100*new_probs[jk]))

                    lists.append([real_x1,real_x2,real_y1,real_y2,img_name,idx,key,filepath,textLabel])
                    if (key == 'OK') or (key == 'N') or (key == '0'):
                        continue
                    else:
                        disease_bbx_list.append([real_x1,real_x2,real_y1,real_y2,key,img_name,filepath,textLabel])
                        img = cv2.rectangle(img,(real_x1,real_y1),(real_x2,real_y2),box_color,thic)
                        img = cv2.putText(img, textLabel, (real_x1,real_y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, label_color)

                    ##(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,0.4,1)
                    ##textOrg = (real_x1, real_y1-0)

                    #cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), color = None)
                    ##cv2.rectangle(img, (textOrg[0] - 3,textOrg[1]+baseLine - 3), (textOrg[0]+retval[0], textOrg[1]-retval[1]), (255,255,255), -1)
                    ##cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 255))

            print('Elapsed time = {}'.format(time.time() - st))
            print(all_dets)
            #cv2.imshow('img', img)
            #cv2.waitKey(0)
            cv2.imwrite(sv_loc+img_name,img)
            #cv2.imwrite('{}{}'.format(sv_loc,img_name),img)
        bbx_df = pd.DataFrame(lists,columns = ('x1','x2','y1','y2','img_name','index','class','path','label'))
        bbx_df.to_csv(sv_loc+'bbx_df.csv', index=None, sep=',')
    disease_df = pd.DataFrame(disease_bbx_list,columns = ('x1','x2','y1','y2','class','img_name','path','label'))
    disease_df.to_csv('final_bbx.csv', index=None, sep=',')
    thic = 2
    box_color_SPD = (102, 102, 255)
    box_color_ADB = (102, 255, 255)
    box_color_SS = (102, 255, 102)
    box_color_EPD = (255, 255, 102)
    label_color = (255, 255, 255)
    for i,j in zip(disease_df.img_name.unique(),disease_df.path.unique()):
        img = cv2.imread(j)
        cv2.rectangle(img,(4,4),(6,10),box_color_SPD,2)
        cv2.rectangle(img,(4,17),(6,23),box_color_ADB,2)
        cv2.rectangle(img,(4,29),(6,35),box_color_SS,2)
        cv2.rectangle(img,(4,41),(6,47),box_color_EPD,2)
        cv2.putText(img, "SPD", (10,11), cv2.FONT_HERSHEY_DUPLEX, 0.4, box_color_SPD)
        cv2.putText(img, "ADB", (10,24), cv2.FONT_HERSHEY_DUPLEX, 0.4, box_color_ADB)
        cv2.putText(img, "SS", (10,36), cv2.FONT_HERSHEY_DUPLEX, 0.4, box_color_SS)
        cv2.putText(img, "EPD", (10,48), cv2.FONT_HERSHEY_DUPLEX, 0.4, box_color_EPD)
        for _,row in disease_df[disease_df.img_name == i].iterrows():
            if row['class'] == "SPD":
                img = cv2.rectangle(img,(row.x1,row.y1),(row.x2,row.y2),box_color_SPD,thic)
                #img = cv2.putText(img, row['class'], (row.x1,row.y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, label_color)
            elif row['class'] == "ADB":
                img = cv2.rectangle(img,(row.x1,row.y1),(row.x2,row.y2),box_color_ADB,thic)
                #img = cv2.putText(img, row['class'], (row.x1,row.y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, label_color)
            elif row['class'] == "SS":
                img = cv2.rectangle(img,(row.x1,row.y1),(row.x2,row.y2),box_color_SS,thic)
                #img = cv2.putText(img, row['class'], (row.x1,row.y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, label_color)
            elif row['class'] == "EPD":
                img = cv2.rectangle(img,(row.x1,row.y1),(row.x2,row.y2),box_color_EPD,thic)
                #img = cv2.putText(img, row['class'], (row.x1,row.y1), cv2.FONT_HERSHEY_DUPLEX, 0.5, label_color)
        cv2.imwrite('results_imgs/{}'.format(i),img)
    for root,dirnames, filenames in os.walk('results_imgs/'):
            for filename in filenames:
                filename=filename
    return render_template("result.html", user_image = filename)
    #zipf = zipfile.ZipFile('RESULTS.zip','w', zipfile.ZIP_DEFLATED)
    #for root,dirs, files in os.walk('results_imgs/'):
     #   for file in files:
      #      zipf.write('results_imgs/'+file)
    #zipf.close()
    #return send_file('RESULTS.zip',
     #       mimetype = 'zip',
      #      attachment_filename= 'RESULTS.zip',
       #     as_attachment = True)
    #return send_from_directory(app.config['RESULT_FOLDER'],
     #                          filename=filename + '.jpg', as_attachment=True)
    #lIndex=req.rfind(".")
    #global res
    #global domain
    #domain=req[lIndex::]
    #print(domain)
    #res="0"+res[lIndex::]
    #print(res)
    K.clear_session()

    #return redirect('/uload')
#test_frcnn
#@app.route("/uload")
#def uload():
#    src="C:\\my_flask\\results_imgs\\"
#    dst="C:\\my_flask\\tests\\"
#    f=res
#    shutil.copy(path.join(src,f),dst)
#display







   



@app.route('/')
def index():
    title="Spinal Disease Diagnosing"
    return render_template("index.html",title=title)
@app.route('/about')
def about():
    title="About"
    return render_template("about.html",title=title)

@app.route('/signup')
def signup():
    title="Signup"
    return render_template("signup.html",title=title)   

@app.route('/login')
def login():
    title="Login"
    return render_template("login.html",title=title)

@app.route('/test')
def test():
    title="Test"
    return render_template("test.html",title=title)
@app.route('/result')
def result():
    title="Result"
    return render_template("result.html",title=title)
@app.route('/convert')
def convert():
    title="Convert"
    return render_template("convert.html",title=title)
@app.route('/report')
def report():
    title="Report"
    return render_template("report.html",title=title)

@app.route('/contact')
def contact():
    title="Contact"
    return render_template("contact.html",title=title)  

@app.route('/signups', methods=["POST"])
def signups():
    first_name=request.form.get("first_name")
    last_name=request.form.get("last_name")
    email=request.form.get("email")
    password=request.form.get("password")
    contact=request.form.get("contact")
    gender=request.form.get("gender")

    if not first_name or not last_name or not email or not password or not contact or not gender:
        error_statement ="All form fields required!"
        return render_template("signup.html",
            error_statement=error_statement,
            first_name=first_name,
            last_name=last_name,
            email=email,
            password=password,
            contact=contact,
            gender=gender)
    return render_template("index.html")


@app.route('/comment', methods=["POST"])
def comment():
    if request.method == 'POST':
       name=request.form.get("name")
       email=request.form.get("email")
       subject=request.form.get("subject")
       message=request.form.get("message")
       fieldnames = ['name', 'email','subject','message']
       with open('comments.csv','a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([name, email,subject,message])
            return render_template("contact.html")

if __name__ == '__main__':
    app.run(debug=True)
#   if not first or not second:
#       error_statement="All inputs required!"
#       return render_template("index.html",
#           error_statement=error_statement,
#           first=first,
#           second=second
#           )
# This program adds two numbers provided by the user
 
# Store input numbers
#   num1 = first
#   num2 = second
     
    # Add two numbers
#   sum = float(num1) + float(num2)
     
    # Display the sum
#   data=('{0} + {1} = {2}'.format(num1, num2, int(sum)))
#   return render_template("index.html",data=data)      










