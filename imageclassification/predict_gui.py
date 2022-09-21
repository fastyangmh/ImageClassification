# import
from imageclassification.project_parameters import ProjectParameters
from deeplearningtemplate.predict_gui import BasePredictGUI
from imageclassification.predict import Predict
from PIL import Image, ImageTk
from deeplearningtemplate.data_preparation import parse_transforms
from tkinter import Label, messagebox
import tkinter as tk
import gradio as gr


# class
class PredictGUI(BasePredictGUI):
    def __init__(self, project_parameters) -> None:
        super().__init__(extensions=('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                                     '.pgm', '.tif', '.tiff', '.webp'))
        self.predictor = Predict(project_parameters=project_parameters)
        self.classes = project_parameters.classes
        self.loader = Image.open
        self.transform = parse_transforms(
            transforms_config=project_parameters.transforms_config)['predict']
        self.color_space = project_parameters.color_space
        self.web_interface = project_parameters.web_interface
        self.examples = project_parameters.examples if len(
            project_parameters.examples) else None

        # label
        self.image_label = Label(master=self.window)

    def reset_widget(self):
        super().reset_widget()
        self.image_label.config(image=None)

    def resize_image(self, image):
        width, height = image.size
        ratio = max(self.window.winfo_height(),
                    self.window.winfo_width()) / max(width, height)
        ratio *= 0.25
        return image.resize((int(width * ratio), int(height * ratio)))

    def display(self):
        image = self.loader(self.filepath).convert(self.color_space)
        resized_image = self.resize_image(image=image)
        imageTk = ImageTk.PhotoImage(resized_image)
        self.image_label.config(image=imageTk)
        self.image_label.image = imageTk

    def open_file(self):
        super().open_file()
        self.display()

    def recognize(self):
        if self.filepath is not None:
            predicted = self.predictor.predict(inputs=self.filepath)
            text = ''
            for idx, (c, p) in enumerate(zip(self.classes, predicted)):
                text += '{}: {}, '.format(c, p.round(3))
                if (idx + 1) < len(self.classes) and (idx + 1) % 5 == 0:
                    text += '\n'
            # remove last commas and space
            text = text[:-2]
            self.predicted_label.config(text='probability:\n{}'.format(text))
            self.result_label.config(text=self.classes[predicted.argmax(-1)])
        else:
            messagebox.showerror(title='Error!', message='please open a file!')

    def inference(self, inputs):
        prediction = self.predictor.predict(inputs=inputs)
        result = {c: p for c, p in zip(self.classes, prediction)}
        return result

    def run(self):
        if self.web_interface:
            gr.Interface(fn=self.inference,
                         inputs=gr.inputs.Image(image_mode=self.color_space,
                                                type='filepath'),
                         outputs='label',
                         examples=self.examples,
                         interpretation="default").launch(share=True,
                                                          inbrowser=True)
        else:
            # NW
            self.open_file_button.pack(anchor=tk.NW)
            self.recognize_button.pack(anchor=tk.NW)

            # N
            self.filepath_label.pack(anchor=tk.N)
            self.image_label.pack(anchor=tk.N)
            self.predicted_label.pack(anchor=tk.N)
            self.result_label.pack(anchor=tk.N)

            # run
            super().run()


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # launch prediction gui
    PredictGUI(project_parameters=project_parameters).run()
