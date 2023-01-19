import tkinter as tk
from tkinter import ttk

import keras


from sklearn import datasets
import customtkinter
class NeuralNetworkApp(tk.Tk):
    global count
    count = 0
    def __init__(self):
        super().__init__()

        self.title("Neural Network Builder")
        self.geometry("500x500")



        self.add_layer_button = ttk.Button(self, text="Add hidden Layer", command=lambda: [self.layer_frame.grid_remove(),self.set_network_button.grid_forget(),
                                                                                           self.input_layer(),self.layer_counter(),self.add_hidden_layer(),self.output_layer()])
        self.add_layer_button.pack()
        self.layers = []
        self.layer_frame = ttk.Frame(self)
        self.layer_frame.pack()
        self.input_layer()
        self.output_layer()

        #self.add_layer("Input", "ReLU")
        #self.add_layer("Output", "Sigmoid")
    def output_layer(self,layer_type="Output", activation_function="ReLU"):

        global count
        layer_label = ttk.Label(self.layer_frame, text=f"Output Layer :")
        layer_label.grid(row=count+2, column=0)

        layer_name_label = ttk.Label(self.layer_frame, text="Layer Name: ")
        layer_name_label.grid(row=count+2, column=3)
        self.layer_name_output = ttk.Entry(self.layer_frame)
        self.layer_name_output.insert(0, f'layer_output')
        self.layer_name_output.grid(row=count+2, column=4)

        layer_neurons_label = ttk.Label(self.layer_frame, text="Number of Neurons: ")
        layer_neurons_label.grid(row=count+2, column=5)
        self.layer_neurons_output = ttk.Entry(self.layer_frame)
        self.layer_neurons_output.insert(0, 1)
        self.layer_neurons_output.grid(row=count+2, column=6)

        layer_activation_label = ttk.Label(self.layer_frame, text="Activation Function: ")
        layer_activation_label.grid(row=count+2, column=7)
        self.layer_activation_output = ttk.Combobox(self.layer_frame, values=["ReLU", "Sigmoid", "Tanh", "LeakyReLU", "Softmax"],
                                        state='readonly')
        self.layer_activation_output.set(activation_function)
        self.layer_activation_output.grid(row=count+2, column=8)
        self.set_network_button = customtkinter.CTkButton(self.layer_frame, text="Set network",
                                                          command=lambda: [set_network(), set_compiler(self.layer_frame)])
        self.set_network_button.grid()

    def input_layer(self, layer_type="Input", activation_function="ReLU"):

        layer_label = ttk.Label(self.layer_frame, text=f"Input Layer :")
        layer_label.grid(row=1, column=0)


        layer_name_label = ttk.Label(self.layer_frame, text="Layer Name: ")
        layer_name_label.grid(row=1, column=3)
        self.layer_name_input = ttk.Entry(self.layer_frame)
        self.layer_name_input.insert(0, f'layer_input')
        self.layer_name_input.grid(row=1, column=4)

        layer_neurons_label = ttk.Label(self.layer_frame, text="Number of Neurons: ")
        layer_neurons_label.grid(row=1, column=5)
        self.layer_neurons_input = ttk.Entry(self.layer_frame)
        self.layer_neurons_input.insert(0, 12)
        self.layer_neurons_input.grid(row=1, column=6)

        layer_activation_label = ttk.Label(self.layer_frame, text="Activation Function: ")
        layer_activation_label.grid(row=1, column=7)
        self.layer_activation_input = ttk.Combobox(self.layer_frame, values=["ReLU", "Sigmoid", "Tanh", "LeakyReLU", "Softmax"],
                                        state='readonly')
        self.layer_activation_input.set(activation_function)
        self.layer_activation_input.grid(row=1, column=8)


    def layer_counter(self):
        global count
        count = count+1
        print(count)

    def add_hidden_layer(self,layer_type="Hidden", activation_function="ReLU"):
        global count
        layer_label = ttk.Label(self.layer_frame, text=f"Hidden Layer {count} :")
        layer_label.grid(row=count+1, column=0)


        layer_name_label = ttk.Label(self.layer_frame, text="Layer Name: ")
        layer_name_label.grid(row=count+1, column=3)
        layer_name = ttk.Entry(self.layer_frame)
        layer_name.insert(0, f'layer{count+1}')
        layer_name.grid(row=count+1, column=4)

        layer_neurons_label = ttk.Label(self.layer_frame, text="Number of Neurons: ")
        layer_neurons_label.grid(row=count+1, column=5)
        layer_neurons = ttk.Entry(self.layer_frame)
        layer_neurons.insert(0,12)
        layer_neurons.grid(row=count+1, column=6)

        layer_activation_label = ttk.Label(self.layer_frame, text="Activation Function: ")
        layer_activation_label.grid(row=count+1, column=7)
        layer_activation = ttk.Combobox(self.layer_frame, values=["ReLU", "Sigmoid","Tanh","LeakyReLU","Softmax"], state='readonly')
        layer_activation.set(activation_function)
        layer_activation.grid(row=count+1, column=8)

        self.layers.append((layer_name, layer_neurons, layer_activation))

def set_compiler(frame):
    global count
    count=count+10
    loss_label = ttk.Label(frame, text="Loss Function: ")
    loss_label.grid(row=count + 7, column=0)
    app.loss_label_selection = ttk.Combobox(frame, values=["mean_squared_error", "mean_squared_logarithmic_error", "mean_absolute_error", "binary_crossentropy"],
                                    state='readonly')
    app.loss_label_selection.set("mean_squared_error")
    app.loss_label_selection.grid(row=count + 7, column=1)
    optimizer_label = ttk.Label(frame, text="Optimizer: ")
    optimizer_label.grid(row=count + 6, column=0)
    app.optimizer_label_selection = ttk.Combobox(frame, values=["adam", "adamax", 'sgd','rmsprop','adamw','adadelta'
                                                            ],
                                        state='readonly')
    app.optimizer_label_selection.set("adam")
    app.optimizer_label_selection.grid(row=count + 6, column=1)
    epoch_size_label = ttk.Label(frame, text="Epoch size: ")
    epoch_size_label.grid(row=count + 8, column=0)
    app.epoch_size = ttk.Entry(frame)
    app.epoch_size.insert(0, 100)
    app.epoch_size.grid(row=count + 8, column=1)
    batch_size_label = ttk.Label(frame, text="Batch size: ")
    batch_size_label.grid(row=count + 9, column=0)
    app.batch_size = ttk.Entry(frame)
    app.batch_size.insert(0, 32)
    app.batch_size.grid(row=count + 9, column=1)
    frame.compile_button = customtkinter.CTkButton(frame, text="Compile network",
                                                      command=lambda: [compile_network()])
    frame.compile_button.grid(row=count + 10, column=1)
    count=count-10

def compile_network():
    global model
    model.compile(loss=app.loss_label_selection.get(), optimizer=app.optimizer_label_selection.get(), metrics=['accuracy'])
    model.fit(features, target, epochs=int(app.epoch_size.get()), batch_size=int(app.batch_size.get()))
    _, accuracy = model.evaluate(features, target)
    print('Accuracy: %.2f' % (accuracy * 100))

def set_network():
    global count, features, target, model
    layer_names=[]
    activation_functions=[]
    neuron_numbers=[]# [i[:] for i in self.layers]
    for i in app.layers:
        layer_names.append(i[0].get())
        activation_functions.append(i[2].get())
        neuron_numbers.append(i[1].get())
    layer_names.append(app.layer_name_output.get())
    activation_functions.append(app.layer_activation_output.get())
    neuron_numbers.append(app.layer_neurons_output.get())
    print(layer_names,activation_functions,neuron_numbers)
    model = keras.Sequential()
    model.add(keras.layers.Dense(app.layer_neurons_input.get(),input_shape=(features.shape[1],),activation=app.layer_activation_input.get(),name=app.layer_name_input.get()))
    for index,i in enumerate(layer_names):
        print(i, neuron_numbers[index])
        model.add(keras.layers.Dense(neuron_numbers[index],activation=activation_functions[index],name=layer_names[index]))
    model.build()
    model.summary()

iris = datasets.load_iris()
global features, target
features = iris.data[:, :2]  # we only take the first two features.
target = iris.target
print(features.shape)
app = NeuralNetworkApp()
app.mainloop()