#import
from DeepLearningTemplate import ProjectParameters, create_model, SupervisedModel, Trainer, Predictor, ClassificationPredictorGUI, Tuner
from functools import partial
from src.data_preparation import create_datamodule
from PIL import Image
import gradio as gr


#def
def main():
    # project parameters
    project_parameters = ProjectParameters().parse()
    result = None
    if project_parameters.mode == 'train':
        model_function = partial(create_model, model_cls=SupervisedModel)
        trainer = Trainer(project_parameters=project_parameters,
                          datamodule_function=create_datamodule,
                          model_function=model_function)
        result = trainer()
    elif project_parameters.mode == 'predict':
        model = create_model(project_parameters=project_parameters,
                             model_cls=SupervisedModel)
        loader = Image.open
        predictor = Predictor(project_parameters=project_parameters,
                              model=model,
                              loader=loader)
        result = predictor(inputs=project_parameters.root)
    elif project_parameters.mode == 'predict_gui':
        model = create_model(project_parameters=project_parameters,
                             model_cls=SupervisedModel)
        loader = Image.open
        predict_gui = ClassificationPredictorGUI(
            project_parameters=project_parameters,
            model=model,
            loader=loader,
            gradio_inputs=gr.Image(type='filepath'),
            gradio_outputs=gr.Label(),
            examples=project_parameters.examples)
        result = predict_gui()
    elif project_parameters.mode == 'tuning':
        tuner = Tuner(project_parameters=project_parameters)
        result = tuner()
    else:
        assert 0, f'please check the mode argument.\nmode: {project_parameters.mode}'
    return result


if __name__ == '__main__':
    #launch
    result = main()
