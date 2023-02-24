from time import sleep
from tkinter import *
from turtle import back
from PIL import Image,ImageTk
from tkinter import filedialog

from tomli import load
from model import *

def webcam_predict(Detector,connect,connect1,connect2,train_type):
    connect.config(state=DISABLED)
    connect1.config(state=DISABLED)
    connect2.config(state=DISABLED)
    Detector.predictVideo(0,train_type)
    connect.config(state=ACTIVE)
    connect1.config(state=ACTIVE)
    connect2.config(state=ACTIVE)

def video_predict(Detector,connect,connect1,connect2,train_type):
    global filename
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("Text files",
                                                        "*.txt*"),
                                                       ("all files",
                                                        "*.*")))
    connect.config(state=DISABLED)
    connect1.config(state=DISABLED)
    connect2.config(state=DISABLED)
    Detector.predictVideo(filename,train_type)
    connect.config(state=ACTIVE)
    connect1.config(state=ACTIVE)
    connect2.config(state=ACTIVE)

filename=''
def picture_predict(detector,connect,connect1,connect2,train_type):
    global filename
    filename = filedialog.askopenfilename(initialdir = "/",
                                          title = "Select a File",
                                          filetypes = (("Text files",
                                                        "*.txt*"),
                                                       ("all files",
                                                        "*.*")))
    filename2=filename
    connect.config(state=DISABLED)
    connect1.config(state=DISABLED)
    connect2.config(state=DISABLED)
    detector.Predict(filename,train_type)
    connect.config(state=ACTIVE)
    connect1.config(state=ACTIVE)
    connect2.config(state=ACTIVE)

def back(window,l,detector,connect,connect1,connect2,backbutton,cur):
    del(detector)
    connect1.destroy()
    connect2.destroy()
    backbutton.destroy()
    cur.destroy()
    model_type(window,l,connect)


def options_page(window,l,con,detector,model_name,train_type,cur_model_name):
    con.destroy()
    l.config(text='OPTIONS',font=("Courier", 70), foreground="green",background='Black')
    l.place(x=200,y=50)
    connect=Button(text='Browse a image for detection',font=('Arial',20,'bold'),relief=RAISED,bd=10,bg='black',fg='white',activebackground='black',activeforeground='grey',state=ACTIVE)
    connect.place(relx = 0.76,rely=0.4, x =-2, y = 2, anchor = NE)
    connect1=Button(window,text='Browse a video for detection',font=('Arial',20,'bold'),relief=RAISED,bd=10,bg='black',fg='white',activebackground='black',activeforeground='grey',state=ACTIVE)
    connect1.place(relx = 0.76,rely=0.6, x =-2, y = 2, anchor = NE)
    connect2=Button(window,text='Open webcam for detection',font=('Arial',20,'bold'),relief=RAISED,bd=10,bg='black',fg='white',activebackground='black',activeforeground='grey',state=ACTIVE)
    connect2.place(relx = 0.745,rely=0.8, x =-2, y = 2, anchor = NE)
    connect1.config(command= lambda: [video_predict(detector,connect,connect1,connect2,train_type)])
    connect2.config(command= lambda: [webcam_predict(detector,connect,connect1,connect2,train_type)])
    connect.config(command= lambda: [picture_predict(detector,connect,connect1,connect2,train_type)])
    t="Current Model:\n"+cur_model_name
    current_model_name=Label(text=t,font=('Arial',12,'bold'))
    current_model_name.place(x=590,y=5)
    backbutton=Button(window,text='change model',font=('Arial',16,'bold'),relief=RAISED,bd=10,bg='black',fg='white',activebackground='black',activeforeground='grey',state=ACTIVE)
    backbutton.config(command=lambda:[back(window,l,detector,connect,connect1,connect2,backbutton,current_model_name)])
    backbutton.place(x=5,y=5)

def load_model(window,l,button1,button2,model_name,train_type):
    if train_type==1 and model_name==1:
        namee='pretrained_classes'
        modelname='pretrained_efficientNet'
    elif train_type==1 and model_name==2:
        namee='pretrained_classes'
        modelname='pretrained_mobileNet'

    elif train_type==2 and model_name==1:
        namee='customtrained_classes'
        modelname='customtrained_efficientNet'

    elif train_type==2 and model_name==2:
        namee='customtrained_classes'
        modelname='customtrained_mobileNet'
    button1.destroy()
    button2.destroy()
    l.config(text="MODEL LOADED \nSUCCESSFULLY",font=("Nueva Std Cond", 50), foreground="green",background='Black')
    l.place(x=130,y=100)
    detector=Detector()
    detector.read_output_file(namee)
    detector.loadModel(modelname)
    connect=Button(window,command= lambda: [options_page(window,l,connect,detector,model_name,train_type,modelname)],text='Proceed',font=('Arial',25,'bold'),relief=RAISED,bd=10,bg='black',fg='white',activebackground='black',activeforeground='grey',state=ACTIVE)
    connect.place(relx = 0.625,rely=0.7, x =-2, y = 2, anchor = NE)

    

def model_selection(window,l,button1,button2,answer):
    button1.destroy()
    button2.destroy()
    l.configure(text="Select model type",font=("Nueva Std Cond", 40))
    l.place(x=200,y=100)
    button3=Button(window,command= lambda: [load_model(window,l,button3,button4,answer,1)],text='pre-trained',font=('Arial',25,'bold'),relief=RAISED,bd=10,bg='black',fg='white',activebackground='black',activeforeground='grey',state=ACTIVE)
    button3.place(relx = 0.4,rely=0.7, x =0, y = 2, anchor = NE)
    button4=Button(window,command= lambda: [load_model(window,l,button3,button4,answer,2)],text='custom-trained',font=('Arial',25,'bold'),relief=RAISED,bd=10,bg='black',fg='white',activebackground='black',activeforeground='grey',state=ACTIVE)
    button4.place(relx = 0.925,rely=0.7, x =0, y = 2, anchor = NE)



def model_type(window,l,connect):
    connect.destroy()
    l.configure(text="Select model ",font=("Nueva Std Cond", 60))
    l.place(x=150,y=100)
    button1=Button(window,command= lambda: [model_selection(window,l,button1,button2,1)],text='Efficient-D0',font=('Arial',25,'bold'),relief=RAISED,bd=10,bg='black',fg='white',activebackground='black',activeforeground='grey',state=ACTIVE)
    button1.place(relx = 0.4,rely=0.7, x =0, y = 2, anchor = NE)
    button2=Button(window,command= lambda: [model_selection(window,l,button1,button2,2)],text='MobileNet-D0',font=('Arial',25,'bold'),relief=RAISED,bd=10,bg='black',fg='white',activebackground='black',activeforeground='grey',state=ACTIVE)
    button2.place(relx = 0.925,rely=0.7, x =0, y = 2, anchor = NE)

def main_window():   
    window = Tk()
    window.geometry("800x500")
    window.configure(bg = "#ffffff")
    img= ImageTk.PhotoImage(Image.open("Background.jpg"))
    mylabel=Label(image=img)    
    mylabel.pack()
    l = Label(window, text="OBJECTIFIER")
    l.config(font=("Nueva Std Cond", 60), foreground="green",background='Black')
    l.place(x=150,y=100)
    # detector.downloadModel('http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz')
    connect=Button(window,command= lambda: [model_type(window,l,connect)],text='Proceed',font=('Arial',25,'bold'),relief=RAISED,bd=10,bg='black',fg='white',activebackground='black',activeforeground='grey',state=ACTIVE)
    connect.place(relx = 0.625,rely=0.7, x =-2, y = 2, anchor = NE)
    window.mainloop()



main_window()