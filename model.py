import  cv2
import os
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file
import time
import numpy as np
class Detector: 
    def __init__(self):
        np.random.seed(50)

    def read_output_file(self, Path): 
        file=open(Path, 'r')   
        self.output_types = file.read()
        self.output_types=self.output_types.splitlines()
        file.close()
        self.colorss = np.random.uniform(low=0, high=255, size=(len(self.output_types), 3))
    
    def downloadModel(self, modelURL):
        fileName =os.path.basename(modelURL)
        self.modelName=fileName[:fileName.index('.')]
        self.cacheDir="./pretrained_models"
        os.makedirs(self.cacheDir,exist_ok=True)
        get_file(fname=fileName, origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)


    def loadModel(self,model_name):
        self.modelName=model_name
        self.cacheDir="./pretrained_models"
        print("Loading model ",self.modelName,"right now")
        # tf.keras.backend.clear_session()
        self.model=tf.saved_model.load(os.path.join(self.cacheDir,"checkpoints",self.modelName,'saved_model'))
        print("loaded",self.modelName,"successfully")


    def Predict(self,image_path,train_type):
        imagee=cv2.imread(image_path)
        bbox_Image = self.create_Bounding_Box(imagee,train_type)
        # cv2.imwrite(self.modelName + ".png", bbox_Image) 
        bbox_Image = cv2.resize(bbox_Image, (800, 600)) 
        cv2.imshow("Result",bbox_Image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def create_Bounding_Box(self, image,train_type):
        Tensor_conversion = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB) 
        Tensor_conversion = tf.convert_to_tensor(Tensor_conversion, dtype=tf.uint8) 
        Tensor_conversion = Tensor_conversion[tf.newaxis,...]
        predictions = self.model(Tensor_conversion)
        Annotation_boxes = predictions['detection_boxes'][0].numpy(); 
        classIndexes = predictions['detection_classes'][0].numpy().astype(np.int32)
        relative_confidence = predictions['detection_scores'][0].numpy()
        image_height, image_width, image_c = image.shape
        if train_type==1:
            bboxIdx=tf.image.non_max_suppression(Annotation_boxes,relative_confidence,max_output_size=50,
            iou_threshold=0.5,score_threshold=0.5)
        else:
            bboxIdx=tf.image.non_max_suppression(Annotation_boxes,relative_confidence,max_output_size=50,
            iou_threshold=0.4,score_threshold=0.4)
        
        
        if len(bboxIdx) != 0: 
            for i in bboxIdx:
                bbox = tuple(Annotation_boxes[i].tolist()) 
                classConfidence = round(100*relative_confidence[i]) 
                classIndex = classIndexes[i]

                classLabelText = self.output_types[classIndex] 
                classColor = self.colorss[classIndex]
                displayText = '{}: {}%'.format(classLabelText, classConfidence)
                ymin, xmin, ymax, xmax = bbox
                xmin, xmax, ymin, ymax = (xmin * image_width, xmax * image_width, ymin * image_height, ymax * image_height)
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=2),
                cv2.putText(image,displayText,(xmin,ymin-10),cv2.FONT_HERSHEY_PLAIN,3,classColor,3)       
                lineWidth=min(int((xmax-xmin)*0.2),int((ymax-ymin)*0.2))
                cv2.line(image,(xmin,ymin),(xmin+lineWidth,ymin),classColor,thickness=5)
                cv2.line(image,(xmin,ymin),(xmin,ymin+lineWidth),classColor,thickness=5)
                cv2.line(image,(xmax,ymin),(xmax-lineWidth,ymin),classColor,thickness=5)
                cv2.line(image,(xmax,ymin),(xmax,ymin+lineWidth),classColor,thickness=5)

                cv2.line(image,(xmin,ymax),(xmin+lineWidth,ymax),classColor,thickness=5)
                cv2.line(image,(xmin,ymax),(xmin,ymax-lineWidth),classColor,thickness=5)
                cv2.line(image,(xmax,ymax),(xmax-lineWidth,ymax),classColor,thickness=5)
                cv2.line(image,(xmax,ymax),(xmax,ymax-lineWidth),classColor,thickness=5)
        return image

    def predictVideo(self,path,train_type):
        cap=cv2.VideoCapture(path)
        if (cap.isOpened()==False):
            print("Error opening video :")
            return   
        (success,image)=cap.read()
        starttime=0
        while success:
            currenttime=time.time()
            fps=currenttime=1/(currenttime-starttime)
            starttime=currenttime
            bboximage=self.create_Bounding_Box(image,train_type)
            cv2.putText(bboximage,"FPS :"+str(int(fps)),(20,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
            bboximage = cv2.resize(bboximage, (800, 600)) 
            # resize_img = cv2.resize(bboximage,(int(cap.get(4)),int(cap.get(3))))
            cv2.imshow("Result",bboximage)
            key=cv2.waitKey(1) #& 0xFF
            if key==ord("a"):
                break
            (success,image)=cap.read()
        cv2.destroyAllWindows()