import tkinter as tk
import numpy as np
import customtkinter
import os
from PIL import Image
from PIL import Image, ImageTk
from tkinter import PhotoImage, filedialog
from sklearn import datasets
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import time
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import sys


class App(customtkinter.CTk):
    def __init__(self):
        global dataset_var,data_selection,data
        super().__init__()
        self.title("NCML")
        self.geometry("1000x600")

        # set grid layout 1x2
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # load images with light and dark mode image
        image_path =os.getcwd()
        self.logo_image = customtkinter.CTkImage(Image.open("prev/CustomTkinter_logo_single.png"), size=(26, 26))
        self.large_test_image = customtkinter.CTkImage(Image.open("prev/CustomTkinter_logo_single.png"), size=(500, 150))
        self.image_icon_image = customtkinter.CTkImage(Image.open("prev/CustomTkinter_logo_single.png"), size=(20, 20))
        self.home_image = customtkinter.CTkImage(light_image=Image.open("prev/CustomTkinter_logo_single.png"),
                                                 dark_image=Image.open("prev/CustomTkinter_logo_single.png"), size=(20, 20))
        self.chat_image = customtkinter.CTkImage(light_image=Image.open("prev/CustomTkinter_logo_single.png"),
                                                 dark_image=Image.open("prev/CustomTkinter_logo_single.png"), size=(20, 20))
        self.add_user_image = customtkinter.CTkImage(light_image=Image.open("prev/CustomTkinter_logo_single.png"),
                                                     dark_image=Image.open("prev/CustomTkinter_logo_single.png"), size=(20, 20))

        # create navigation frame
        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(7, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="No Code Machine Learning", image=self.logo_image,
                                                             compound="left", font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=20, pady=20)

        self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Data import",
                                                   fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                   image=self.home_image, anchor="w", command=self.home_button_event)
        self.home_button.grid(row=1, column=0, sticky="ew")

        self.frame_2_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Data visualization",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      image=self.chat_image, anchor="w", command=self.frame_2_button_event)
        self.frame_2_button.grid(row=2, column=0, sticky="ew")

        self.frame_3_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Framework selection",
                                                      fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                                                      image=self.add_user_image, anchor="w", command=self.frame_3_button_event)
        self.frame_3_button.grid(row=3, column=0, sticky="ew")
        self.frame_4_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                      border_spacing=10, text="Model selection",
                                                      fg_color="transparent", text_color=("gray10", "gray90"),
                                                      hover_color=("gray70", "gray30"),
                                                      image=self.add_user_image, anchor="w",
                                                      command=self.frame_4_button_event)
        self.frame_4_button.grid(row=4, column=0, sticky="ew")
        self.frame_5_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                      border_spacing=10, text="Train and test",
                                                      fg_color="transparent", text_color=("gray10", "gray90"),
                                                      hover_color=("gray70", "gray30"),
                                                      image=self.add_user_image, anchor="w",
                                                      command=self.frame_5_button_event)
        self.frame_5_button.grid(row=5, column=0, sticky="ew")
        self.frame_6_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40,
                                                      border_spacing=10, text="Tuning",
                                                      fg_color="transparent", text_color=("gray10", "gray90"),
                                                      hover_color=("gray70", "gray30"),
                                                      image=self.add_user_image, anchor="w",
                                                      command=self.frame_6_button_event)
        self.frame_6_button.grid(row=6, column=0, sticky="ew")

        self.appearance_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame, values=["Light", "Dark", "System"],
                                                                command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=7, column=0, padx=20, pady=20, sticky="s")


        # create home frame
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=1)
        image0 = Image.open("img/local.png").resize((200, 200))
        self.home_frame.local_label = tk.Label(self.home_frame, text="Select from Local")
        self.home_frame.local_button = customtkinter.CTkButton(self.home_frame,text='Select data', command=lambda:[data_from_local(self.home_frame)])
        self.home_frame.local_label.grid()
        self.home_frame.local_button.grid()
        self.home_frame.load_button = tk.Button(self.home_frame, text="Load", command=lambda: [self.home_frame.local_label.grid_forget(),self.home_frame.local_button.grid_forget(),
                                                                                                   self.home_frame.select_button.grid_forget(),
                                                                                                   self.home_frame.new_data_button.grid(),
                                                                                                   self.home_frame.dataset_select.grid_forget(),
                                                                                                   self.home_frame.web_label.grid_forget(),
                                                                                                   self.home_frame.dataset_select_label.grid_remove()])
        self.home_frame.load_button.grid()



        self.home_frame.web_label = tk.Label(self.home_frame, text="Select from Web")
        self.home_frame.web_label.grid()

        dataset_var = tk.StringVar(value='load_iris')
        dataset_names = ['load_iris', 'load_diabetes']
        self.home_frame.dataset_select_label = tk.Label(self.home_frame, text="Select from listed datasets.")
        self.home_frame.dataset_select_label.grid()
        self.home_frame.dataset_select = customtkinter.CTkOptionMenu(self.home_frame,variable=dataset_var, values=dataset_names)
        self.home_frame.dataset_select.grid()
        self.home_frame.select_button = tk.Button(self.home_frame, text="Select", command=lambda:[import_data2(),self.home_frame.select_button.grid_forget(),self.home_frame.new_data_button.grid(),self.home_frame.dataset_select.grid_forget(),self.home_frame.web_label.grid_forget(),self.home_frame.dataset_select_label.grid_remove(),data_vis(self.second_frame)])
        self.home_frame.select_button.grid()
        self.home_frame.new_data_button = tk.Button(self.home_frame, text="New data",
                                                    command=lambda: [self.home_frame_clean(),
                                                                     self.home_frame.web_label.grid(),
                                                                     self.home_frame.dataset_select_label.grid(),
                                                                     self.home_frame.dataset_select.grid(),
                                                                     self.home_frame.select_button.grid()])


        label = tk.Label(self.home_frame)
        label.grid()






        # create second frame
        self.second_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.second_frame.grid_columnconfigure(0, weight=1)
        self.second_frame.text_var = tk.StringVar()
        self.second_frame.text_var.set('Please select data')
        self.second_frame.intro = tk.Label(self.second_frame, textvariable=self.second_frame.text_var)
        self.second_frame.intro.grid()
        self.second_frame.data_info = tk.StringVar()
        self.second_frame.data_info.set('')
        self.second_frame.data_info_label = tk.Label(self.second_frame, textvariable=self.second_frame.data_info)
        self.second_frame.data_info_label.grid()


        # create third frame
        self.third_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.third_frame.grid_columnconfigure(0, weight=1)
        self.third_frame.label_1_var = tk.StringVar()
        self.third_frame.label_1_var.set('Please select')
        image_sp = ImageTk.PhotoImage(Image.open("img/spark.png").resize((200, 200)))
        self.third_frame.pyspark_button = customtkinter.CTkButton(self.third_frame, text="pySpark",
                                                                  command=lambda: [pyspark_func(self.third_frame),
                                                                                   self.third_frame.label_1_var.set(
                                                                                       'PySpark is selected'),
                                                                                   self.third_frame.pyspark_button.grid_forget(),
                                                                                   self.third_frame.sklearn_button.grid_forget()])


        image_sk = ImageTk.PhotoImage(Image.open("img/sklearn.png").resize((200, 200)))
        self.third_frame.sklearn_button = customtkinter.CTkButton(self.third_frame, text="sklearn",
                                                                  command=lambda: [sklearn_func(self.fourth_frame),
                                                                                   self.third_frame.label_1_var.set(
                                                                                       'Sklearn is selected'),
                                                                                   self.third_frame.pyspark_button.grid_forget(),
                                                                                   self.third_frame.sklearn_button.grid_forget()])

        # Pack radio buttons and next button
        self.third_frame.pyspark_button.grid()
        self.third_frame.sklearn_button.grid()

        self.third_frame.label_1 = customtkinter.CTkLabel(master=self.third_frame, textvariable=self.third_frame.label_1_var, justify=tk.LEFT)
        self.third_frame.label_1.grid(pady=10, padx=10)
        # create fourth frame
        self.fourth_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.fourth_frame.grid_columnconfigure(0, weight=1)



        # create fifth frame
        self.fifth_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.fifth_frame.grid_columnconfigure(0, weight=1)
        # create fifth frame
        self.sixth_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.sixth_frame.grid_columnconfigure(0, weight=1)
        self.sixth_frame.text_var = tk.StringVar()
        self.sixth_frame.text_var.set('Please select data to proceed')
        self.sixth_frame.intro = tk.Label(self.sixth_frame, textvariable=self.sixth_frame.text_var)
        self.sixth_frame.intro.grid()

        # select default frame
        self.select_frame_by_name("home")


        #display_data2()

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")
        self.frame_2_button.configure(fg_color=("gray75", "gray25") if name == "frame_2" else "transparent")
        self.frame_3_button.configure(fg_color=("gray75", "gray25") if name == "frame_3" else "transparent")
        self.frame_4_button.configure(fg_color=("gray75", "gray25") if name == "frame_4" else "transparent")
        self.frame_5_button.configure(fg_color=("gray75", "gray25") if name == "frame_5" else "transparent")
        # show selected frame
        if name == "home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "frame_2":
            self.second_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.second_frame.grid_forget()
        if name == "frame_3":
            self.third_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.third_frame.grid_remove()
        if name == "frame_4":
            self.fourth_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.fourth_frame.grid_forget()
        if name == "frame_5":
            self.fifth_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.fifth_frame.grid_forget()
        if name == "frame_6":
            self.sixth_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.sixth_frame.grid_forget()

    def home_frame_clean(self):

        self.home_frame.dataset_select_label.grid_forget()
        #self.home_frame.web_label.grid_forget()
        self.home_frame.target_label.grid_forget()
        self.home_frame.target_select.grid_forget()
        self.home_frame.feature_label.grid_forget()
        self.home_frame.feature_select.grid_forget()
        self.home_frame.submit_data.grid_forget()
        self.home_frame.data_label.grid_forget()
        self.home_frame.new_data_button.grid_forget()
        self.home_frame.grid(row=0, column=1, sticky="nsew")

    def home_button_event(self):
        self.select_frame_by_name("home")


    def frame_2_button_event(self):
        self.select_frame_by_name("frame_2")

    def frame_3_button_event(self):
        self.select_frame_by_name("frame_3")
    def frame_4_button_event(self):
        self.select_frame_by_name("frame_4")

    def frame_5_button_event(self):
        self.select_frame_by_name("frame_5")

    def frame_6_button_event(self):
        self.select_frame_by_name("frame_6")

    def change_appearance_mode_event(self, new_appearance_mode):
        customtkinter.set_appearance_mode(new_appearance_mode)

def data_vis(frame):
    global data, data_selection,app,features,target
    frame.text_var.set('Data is selected')
    frame.data_info.set(data.head().to_string())


def pyspark_func(frame):
    print('hell')


def sklearn_func(frame):
    models_param = {}
    model_var = tk.StringVar()
    model_var.trace("w", lambda *args: get_parameters(model_var.get()))
    def get_parameters():
        global models_param, models_list, target, features
        a=1
        for model_name,model_func in models_list:
            parameters = model_func.get_params()
            print(parameters)
            label1 = tk.Label(frame, text=f"Parameters for {model_name}")
            label1.grid(row=0, column=a)

            b=2
            for key, value in parameters.items():

                label = tk.Label(frame, text=f"{key} :default is {value}")
                label.grid(row=b,column=a)
                globals()[f"value{key}"] = tk.StringVar(frame, value=value)
                globals()[f"entry{key}"] = customtkinter.CTkEntry(frame, textvariable=globals()[f"value{key}"]) #, command=model_func.set_params(classifier__C=C)
                globals()[f"entry{key}"].grid(row=b+1,column=a)
                b=b+2
            a = a + 1
        set_params_button = customtkinter.CTkButton(frame, text="Set parameters",
                                                    command=lambda: [set_parameters()])
        set_params_button.grid(row=len(model_list)+3, column=0)
    def set_parameters():
        for model_name,model_func in models_list:
            parameters = model_func.get_params()
            params={}
            for key, value in parameters.items():
                key_value = globals()[f"value{key}"].get()
                if type(value) == int:
                    key_value = int(key_value)
                elif type(value) == float:
                    key_value = float(key_value)
                else:
                    key_value = value

                #value = str_to_class()
                params[key]=key_value
            print(params)
            model_func.set_params(**params)
    def on_select(var_list,model_list):
        global models_list, selection
        selection = [model_list[i] for i in range(len(var_list)) if var_list[i].get()]
        models_list = []
        for model_name in selection:
            if model_name == "Decision Tree":
                models_list.append(('Decision Tree',DecisionTreeClassifier()))
            elif model_name == "Support Vector Machine":
                models_list.append(('SVM',SVC()))
            elif model_name == "Logistic Regression":
                models_list.append(('LogReg', LogisticRegression()))
            elif model_name == "KNeighborsClassifier":
                models_list.append(('KNN', KNeighborsClassifier()))
            elif model_name == "GaussianNB":
                models_list.append(('GaussianNB', GaussianNB()))
            elif model_name == "MLPC":
                models_list.append(('MLPC', KNeighborsClassifier()))
            elif model_name == "RandomForestClassifier":
                models_list.append(('RandomForestClassifier', RandomForestClassifier()))
            elif model_name == "AdaBoostClassifier":
                models_list.append(('AdaBoostClassifier', AdaBoostClassifier()))
            elif model_name == "LinearDiscriminantAnalysis":
                models_list.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))
            elif model_name == "KNeighborsClassifier":
                models_list.append(('KNN', KNeighborsClassifier()))
        print(models_list)
        #get_parameters()

    var_list=[]

    model_list = ["Logistic Regression", "Support Vector Machine",  "Decision Tree", "KNeighborsClassifier","GaussianNB" ,"MLPC","RandomForestClassifier","AdaBoostClassifier","LinearDiscriminantAnalysis"]
    a=1
    for element in model_list:
        var = tk.IntVar()
        var_list.append(var)
        cb = customtkinter.CTkCheckBox(frame, text=element, variable=var, command=lambda: on_select(var_list,model_list))
        cb.grid(row=a,column=0)
        a=a+1
    set_button = customtkinter.CTkButton(frame, text="Get models", command=lambda: [on_train_sklearn(),get_parameters(),find_best_window()])
    set_button.grid(row=len(model_list)+1,column=0)
count_test = 0
count_train = 0
def train_clicked(): # without event because I use `command=` instead of `bind`
    global count_train

    count_train = count_train + 1

def test_clicked(): # without event because I use `command=` instead of `bind`
    global count_test

    count_test = count_test + 1
def on_train_sklearn():
    global models_param, models_list, target, features
    frame = app.fifth_frame
    on_train_window = frame
    X = features
    y = target
    print(X.shape,y.shape)
    def train_models():
        print(k_fold.get())
        global X_train, X_test, y_train, y_test, trained_models,count_train,count_test
        trained_models=[]

        for a,model in enumerate(models_list):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - app.split_ratio_var.get()))
            start_time = time.time()
            print(model[1])
            trained_models.append(model[1].fit(X_train, y_train))
            end_time = time.time()

            execution_time = round(end_time - start_time, 5)
            # frame.train_label = tk.Label(frame.canvas, text=f"{model[0]} is successfully trained in {execution_time} seconds.")
            # frame.train_label.grid(row=count_train + a,column=1)
            frame.canvas.insert(tk.END,f"{model[0]} is successfully trained in {execution_time} seconds.")

            print(f"{model[0]} is successfully trained in {execution_time} seconds.")
        count_train = count_train + a
    def test_models():
        global X_train, X_test, y_train, y_test, trained_models
        for index,model in enumerate(trained_models):
            accuracy = model.score(X_test, y_test)
            print(f"{models_list[index][0]} has an accuracy of {accuracy}.")
            # frame.test_label = tk.Label(frame.canvas,
            #                        text=f"{models_list[index][0]} has an accuracy of {accuracy}.")
            # frame.test_label.grid(row=index,column=2,sticky="e")
            frame.canvas2.insert(tk.END, f"{models_list[index][0]} has an accuracy of {accuracy}.")



    # Create input for train/test split ratio
    app.split_ratio_var = tk.DoubleVar(value=0.8)
    split_ratio_entry = tk.Entry(on_train_window, textvariable=app.split_ratio_var)
    split_ratio_entry.grid()

    # Create train button
    train_button = tk.Button(on_train_window, text="Train", command=lambda :[train_models(),train_clicked()])
    train_button.grid()

    # Create test button
    test_button = tk.Button(on_train_window, text="Test", command=lambda :[test_models(),test_clicked()])
    test_button.grid()

    k_fold = tk.DoubleVar(value=2)
    frame.entry_1 = customtkinter.CTkEntry(master=frame, textvariable=k_fold, placeholder_text="k_fold number, min 2")
    frame.checkbox_1 = customtkinter.CTkCheckBox(master=frame, text='Employ cross-validation, as integer, minimum 2',
                                                             command=frame.entry_1.grid)
    frame.checkbox_1.grid()

    tk.Label(frame,text="Training").grid(row=4, column=0)
    # Add a canvas in that frame
    frame.canvas = tk.Listbox(frame)
    frame.canvas.grid(row=5, column=0, sticky="news")

    # Link a scrollbar to the canvas
    vsb = tk.Scrollbar(frame, orient="vertical", command=frame.canvas.yview)
    vsb.grid(row=5, column=1, sticky='ns')
    frame.canvas.configure(yscrollcommand=vsb.set)
    tk.Label(frame,text="Test").grid(row=6, column=0)
    # Add a canvas in that frame
    frame.canvas2 = tk.Listbox(frame)
    frame.canvas2.grid(row=7, column=0, sticky="news")

    # Link a scrollbar to the canvas
    vsb2 = tk.Scrollbar(frame, orient="vertical", command=frame.canvas2.yview)
    vsb2.grid(row=7, column=1, sticky='ns')
    frame.canvas2.configure(yscrollcommand=vsb2.set)


    #on_train_window.mainloop()

def framework_selection(frame):
    pass

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)
def data_from_local(display_window):
    global data
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files","*.csv"),("MAT Files","*.mat"),("TXT Files","*.txt")])
    file_ext = filepath.split(".")[-1]
    if file_ext == "csv":
        data = pd.read_csv(filepath)
        data = pd.DataFrame(data)
    elif file_ext == "mat":
        data = scipy.io.loadmat(filepath)
        data = pd.DataFrame(data)
    elif file_ext == "txt":
        data = pd.read_csv(filepath, delimiter = '\t')
        data = pd.DataFrame(data)

    def store_selections(target_var, feature_vars):
        global data, target, features, data_selection, feature_names

        target = (data[target_var.get()])  # data.columns[str( )]
        features = pd.DataFrame()
        data_selection = 1
        for i in feature_vars:
            print({data.columns[i]: (data[data.columns[i]])})
            features[data.columns[i]] = np.asarray(data[data.columns[
                i]])  # features.append(features_temp, ignore_index=True) #features[] = data[data.columns[i]]
        print(features)
        feature_names = [data.columns[i] for i in feature_vars]
        target_text = tk.Label(display_window,
                               text=f"Data is selected. You can return this page to change selected features.\n Selected feature(s): {feature_names}"
                               )
        target_text.grid()

    display_window.target_var = tk.StringVar(value='target')
    display_window.feature_vars = tk.StringVar()
    display_window.target_label = tk.Label(display_window, text="Select Target Column")
    display_window.target_label.grid()
    display_window.target_select = customtkinter.CTkOptionMenu(display_window, variable=display_window.target_var,
                                                               values=list(data.columns))
    display_window.target_select.grid()
    display_window.feature_label = tk.Label(display_window, text="Select Feature Columns")
    display_window.feature_label.grid()
    display_window.feature_select = tk.Listbox(display_window, selectmode='multiple',
                                               listvariable=display_window.feature_vars)
    for col in data.columns:
        display_window.feature_select.insert(tk.END, col)
    display_window.feature_select.grid()

    display_window.data_label = tk.Label(display_window, text=data.head().to_string())
    display_window.data_label.grid()
    display_window.submit_data = customtkinter.CTkButton(display_window, text="Submit data",
                                                         command=lambda: store_selections(display_window.target_var,
                                                                                          display_window.feature_select.curselection()))
    display_window.submit_data.grid()

def import_data2():
    global data,data_selection
    dataset_name = dataset_var.get()
    data = getattr(datasets,dataset)()

    #display_data2()

def display_data2(display_window):
    global data, target, features,dataset_var,data_selection
    data = pd.DataFrame(data=np.c_[data['data'], data['target']],columns=data['feature_names'] + ['target'])
    data = data.dropna()

    def store_selections(target_var, feature_vars):
        global data, target, features,data_selection,feature_names

        target = (data[target_var.get()])#data.columns[str( )]
        features = pd.DataFrame()
        data_selection = 1
        for i in feature_vars:
            print({data.columns[i]: (data[data.columns[i]])})
            features[data.columns[i]] = np.asarray(data[data.columns[i]])# features.append(features_temp, ignore_index=True) #features[] = data[data.columns[i]]
        print(features)
        feature_names = [data.columns[i] for i in feature_vars]
        target_text = tk.Label(display_window, text=f"Data is selected. You can return this page to change selected features.\n Selected feature(s): {feature_names}"
                                                    )
        target_text.grid()
    display_window.target_var = tk.StringVar(value='target')
    display_window.feature_vars = tk.StringVar()
    display_window.target_label = tk.Label(display_window, text="Select Target Column")
    display_window.target_label.grid()
    display_window.target_select = customtkinter.CTkOptionMenu(display_window, variable=display_window.target_var, values=list(data.columns))
    display_window.target_select.grid()
    display_window.feature_label = tk.Label(display_window, text="Select Feature Columns")
    display_window.feature_label.grid()
    display_window.feature_select = tk.Listbox(display_window, selectmode='multiple', listvariable=display_window.feature_vars)
    for col in data.columns:
        display_window.feature_select.insert(tk.END, col)
    display_window.feature_select.grid()

    display_window.data_label = tk.Label(display_window, text=data.head().to_string())
    display_window.data_label.grid()
    display_window.submit_data = customtkinter.CTkButton(display_window, text="Submit data", command=lambda: store_selections(display_window.target_var,display_window.feature_select.curselection()))
    display_window.submit_data.grid()


def donothing():
    x = 0
def import_data2():
    global data
    data = getattr(datasets, dataset_var.get())()
    print(data)

    display_data2(app.home_frame)

def new_screen():
    global app, data_selection
    app.destroy()
    data_selection=0
    app= App()
    menubar(app)
    app.mainloop()

def menubar(app):
    menubar = tk.Menu(app)
    filemenu = tk.Menu(menubar, tearoff=0)
    filemenu.add_command(label="New", command=new_screen)
    filemenu.add_command(label="Open", command=donothing)
    filemenu.add_command(label="Save", command=donothing)
    filemenu.add_separator()
    filemenu.add_command(label="Exit", command=app.quit)
    menubar.add_cascade(label="File", menu=filemenu)

    helpmenu = tk.Menu(menubar, tearoff=0)
    helpmenu.add_command(label="Help Index", command=donothing)
    helpmenu.add_command(label="About...", command=donothing)
    menubar.add_cascade(label="Help", menu=helpmenu)
    app.config(menu=menubar)

def find_best_window():
    global feature_names, selection
    app.sixth_frame.text_var.set(f'Data and models are selected.\n\n Selected model(s): {selection} \n\n Selected feature(s): {feature_names} \n\n Please select what you want to tune')
    app.sixth_frame.find_best_frame = tk.LabelFrame(app.sixth_frame, text="Find Best")
    app.sixth_frame.find_best_frame.grid()  # (row=0, column=4, columnspan=2)
    app.sixth_frame.find_best_var = tk.StringVar(value="features")
    tk.Radiobutton(app.sixth_frame.find_best_frame, text="Features", variable=app.sixth_frame.find_best_var,
                   value="features").grid()
    tk.Radiobutton(app.sixth_frame.find_best_frame, text="Model", variable=app.sixth_frame.find_best_var,
                   value="model").grid()
    tk.Radiobutton(app.sixth_frame.find_best_frame, text="Features and Model", variable=app.sixth_frame.find_best_var,
                   value="features and model").grid()
    app.sixth_frame.find_best_button = customtkinter.CTkButton(app.sixth_frame, text="Find best combination",
                            command=lambda: [find_best()])
    app.sixth_frame.find_best_button.grid()

def find_best():
    print(app.sixth_frame.find_best_var.get())
    app.sixth_frame.find_best_button.grid_forget()
    app.progress_bar = customtkinter.CTkProgressBar(app.sixth_frame,mode='determinate', orientation='horizontal')
    app.progress_bar.set(0)
    app.progress_bar.grid(pady=40)
    app.sixth_frame.find_best_button.configure(state="disable")
    if app.sixth_frame.find_best_var.get() == 'model':
        find_best_model()
    if app.sixth_frame.find_best_var.get() == 'features':
        find_best_features()
    if app.sixth_frame.find_best_var.get() == "features and model":
        find_best_all()

def find_best_model():
    global models_list, target, label, data, features,data_selection,feature_names

    length_of_process = len(feature_names)
    X = features
    y = target

    def train_test_models():
        global X_train, X_test, y_train, y_test, trained_models, count_train, count_test, best_model
        trained_models = []
        best_accuracy = 0
        best_model = []
        for a, model in enumerate(models_list):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - app.split_ratio_var.get()))
            app.progress_bar.set(a*1/length_of_process)
            start_time = time.time()
            trained_models.append(model[1].fit(X_train, y_train))
            end_time = time.time()
            execution_time = round(end_time - start_time, 5)
            #frame.canvas.insert(tk.END, f"{model[0]} is successfully trained in {execution_time} seconds.")
            print(f"{model[0]} is successfully trained in {execution_time} seconds.")
            y_pred = model[1].predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{model[0]} has an accuracy of {accuracy}.")
            if accuracy >= best_accuracy:
                best_model.append([model[0], accuracy])
                print('New good performance!')
                # Print the performance of the model
                print('Model: {}'.format(model[0]))
                print('Accuracy: {}'.format(accuracy))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = model[0]
                    print(f'New best performance! :{best_model_name}')
        count_train = count_train + a
    train_test_models()
    print_best()

def find_best_features():
    pass
def find_best_all():
    pass

def print_best():
    global best_model
    app.print_best_text = tk.Label(app.sixth_frame,
                           text=f"Best performance is achieved with \n {best_model}")

    app.print_best_text.grid()
    app.progress_bar.grid_forget()
    app.sixth_frame.find_best_button.grid()
if __name__ == "__main__":
    global app, data_selection
    data_selection=0
    app = App()
    menubar(app)
    app.mainloop()
