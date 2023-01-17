import tkinter as tk
import numpy as np
import customtkinter
import os
from PIL import Image
from PIL import Image, ImageTk
from tkinter import PhotoImage, filedialog
from sklearn import datasets
import pandas as pd
import time
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


def data_from_local():
    global data
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files","*.csv"),("MAT Files","*.mat"),("TXT Files","*.txt")])
    file_ext = filepath.split(".")[-1]
    if file_ext == "csv":
        data = pd.read_csv(filepath)
    elif file_ext == "mat":
        data = scipy.io.loadmat(filepath)
    elif file_ext == "txt":
        data = pd.read_csv(filepath, delimiter = '\t')


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
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="No Code Machine LEarning", image=self.logo_image,
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

        self.appearance_mode_menu = customtkinter.CTkOptionMenu(self.navigation_frame, values=["Light", "Dark", "System"],
                                                                command=self.change_appearance_mode_event)
        self.appearance_mode_menu.grid(row=6, column=0, padx=20, pady=20, sticky="s")

        # create home frame
        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.home_frame.grid_columnconfigure(0, weight=1)
        image0 = Image.open("img/local.png").resize((200, 200))
        local_label = tk.Label(self.home_frame, text="Select from Local")
        local_button = customtkinter.CTkButton(self.home_frame,text='Load data', command=data_from_local)
        local_label.grid()
        local_button.grid()



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
        self.home_frame.new_data_button = tk.Button(self.home_frame, text="New data", command=lambda:[self.home_frame_clean(),self.home_frame.web_label.grid(),self.home_frame.dataset_select_label.grid(),
                                                                                                      self.home_frame.dataset_select.grid(),self.home_frame.select_button.grid()])

        label = tk.Label(self.home_frame)
        label.grid()




        # create second frame
        self.second_frame = customtkinter.CTkFrame(self, corner_radius=0, fg_color="transparent")
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
        self.third_frame.label_1_var = tk.StringVar()
        self.third_frame.label_1_var.set('Please select')
        image_sp = ImageTk.PhotoImage(Image.open("img/spark.png").resize((200, 200)))
        self.third_frame.pyspark_button = customtkinter.CTkButton(self.third_frame, text="pySpark", command=lambda: [pyspark_func(self.third_frame),
                                                                                               self.third_frame.label_1_var.set(
                                                                                                   'PySpark is selected'), self.third_frame.pyspark_button.grid_forget(), self.third_frame.sklearn_button.grid_forget()])

        image_sk = ImageTk.PhotoImage(Image.open("img/sklearn.png").resize((200, 200)))
        self.third_frame.sklearn_button = customtkinter.CTkButton(self.third_frame, text="sklearn", command=lambda: [sklearn_func(self.third_frame),
                                                                                               self.third_frame.label_1_var.set(
                                                                                                   'Sklearn is selected'), self.third_frame.pyspark_button.grid_forget(), self.third_frame.sklearn_button.grid_forget()])
        # Pack radio buttons and next button
        self.third_frame.pyspark_button.grid()
        self.third_frame.sklearn_button.grid()

        self.third_frame.label_1 = customtkinter.CTkLabel(master=self.third_frame, textvariable=self.third_frame.label_1_var, justify=tk.LEFT)
        self.third_frame.label_1.grid(pady=10, padx=10)

        # select default frame
        self.select_frame_by_name("home")


        #display_data2()

    def select_frame_by_name(self, name):
        # set button color for selected button
        self.home_button.configure(fg_color=("gray75", "gray25") if name == "home" else "transparent")
        self.frame_2_button.configure(fg_color=("gray75", "gray25") if name == "frame_2" else "transparent")
        self.frame_3_button.configure(fg_color=("gray75", "gray25") if name == "frame_3" else "transparent")

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
            self.third_frame.grid_forget()

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

        for model_name,model_func in models_list:
            parameters = model_func.get_params()
            print(parameters)
            for key, value in parameters.items():
                print(key,value)
                label = tk.Label(frame, text=f"{key} :default is {value}")
                label.grid()
                print(type(value))
                entry = customtkinter.CTkEntry(frame, placeholder_text=value)
                entry.grid()


    def on_select(var_list,model_list):
        global models_list
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
    for element in model_list:
        var = tk.IntVar()
        var_list.append(var)
        cb = customtkinter.CTkCheckBox(frame, text=element, variable=var, command=lambda: on_select(var_list,model_list))
        cb.grid()
    set_button = customtkinter.CTkButton(frame, text="Set models", command=lambda: [on_train_sklearn(frame)])
    set_button.grid()
def on_train_sklearn(frame):
    global models_param, models_list, target, features
    X = features
    y = target
    print(X.shape,y.shape)
    def train_models():
        global X_train, X_test, y_train, y_test, trained_models
        trained_models=[]
        for model in models_list:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1 - split_ratio_var.get()))
            start_time = time.time()
            print(model[1])
            trained_models.append(model[1].fit(X_train, y_train))
            end_time = time.time()

            execution_time = round(end_time - start_time, 5)
            frame.train_label = tk.Label(on_train_window, text=f"{model[0]} is successfully trained in {execution_time} seconds.")
            frame.train_label.grid()
            print(f"{model[0]} is successfully trained in {execution_time} seconds.")

    def test_models():
        global X_train, X_test, y_train, y_test, trained_models
        for index,model in enumerate(trained_models):
            accuracy = model.score(X_test, y_test)
            print(f"{models_list[index][0]} has an accuracy of {accuracy}.")
            frame.test_label = tk.Label(on_train_window,
                                   text=f"{models_list[index][0]} has an accuracy of {accuracy}.")
            frame.test_label.grid()


    on_train_window = frame


    # Create input for train/test split ratio
    split_ratio_var = tk.DoubleVar(value=0.8)
    split_ratio_entry = tk.Entry(on_train_window, textvariable=split_ratio_var)
    split_ratio_entry.grid()

    # Create train button
    train_button = tk.Button(on_train_window, text="Train", command=train_models)
    train_button.grid()

    # Create test button
    test_button = tk.Button(on_train_window, text="Test", command=test_models)
    test_button.grid()

    #on_train_window.mainloop()

def framework_selection(frame):
    pass



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

if __name__ == "__main__":
    global app, data_selection
    data_selection=0
    app = App()
    menubar(app)
    app.mainloop()