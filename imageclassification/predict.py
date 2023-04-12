# import
from imageclassification.project_parameters import ProjectParameters
from deeplearningtemplate.predict import ImagePredictDataset
from imageclassification.model import create_model
import torch
from deeplearningtemplate.data_preparation import parse_transforms
from typing import Any
from os.path import isfile
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


#class
class Predict:
    def __init__(self, project_parameters) -> None:
        self.model = create_model(project_parameters=project_parameters).eval()
        if project_parameters.device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.cuda()
        self.transform = parse_transforms(
            transforms_config=project_parameters.transforms_config)['predict']
        self.device = project_parameters.device
        self.batch_size = project_parameters.batch_size
        self.num_workers = project_parameters.num_workers
        self.classes = project_parameters.classes
        self.loader = Image.open
        self.color_space = project_parameters.color_space

    def predict(self, inputs) -> Any:
        result = []
        if type(inputs) == Image.Image:
            # the transformed sample dimension is (1, in_chans, width, height)
            sample = self.transform(inputs)[None]
            if self.device == 'cuda' and torch.cuda.is_available():
                sample = sample.cuda()
            with torch.no_grad():
                result.append(self.model(sample).tolist()[0])
        elif isfile(path=inputs):
            # predict the file
            sample = self.loader(inputs).convert(self.color_space)
            # the transformed sample dimension is (1, in_chans, width, height)
            sample = self.transform(sample)[None]
            if self.device == 'cuda' and torch.cuda.is_available():
                sample = sample.cuda()
            with torch.no_grad():
                result.append(self.model(sample).tolist()[0])
        else:
            # predict the file from folder
            dataset = ImagePredictDataset(root=inputs,
                                          loader=self.loader,
                                          transform=self.transform,
                                          color_space=self.color_space)
            pin_memory = True if self.device == 'cuda' and torch.cuda.is_available(
            ) else False
            data_loader = DataLoader(dataset=dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=self.num_workers,
                                     pin_memory=pin_memory)
            with torch.no_grad():
                for sample in tqdm(data_loader):
                    if self.device == 'cuda' and torch.cuda.is_available():
                        sample = sample.cuda()
                    result.append(self.model(sample).tolist())
        result = np.concatenate(result, 0)
        return result


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # predict file
    result = Predict(project_parameters=project_parameters).predict(
        inputs=project_parameters.root)
