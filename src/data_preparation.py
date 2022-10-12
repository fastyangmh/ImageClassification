#import
from DeepLearningTemplate import ProjectParameters, MyImageFolder, MyCIFAR10, MyMNIST, ImageLightningDataModule


#def
def create_datamodule(project_parameters):
    if project_parameters.predefined_dataset:
        dataset_cls = eval('My{}'.format(
            project_parameters.predefined_dataset))
    else:
        dataset_cls = MyImageFolder
    return ImageLightningDataModule(
        root=project_parameters.root,
        predefined_dataset=project_parameters.predefined_dataset,
        dataset_cls=dataset_cls,
        transforms_config=project_parameters.transforms_config,
        target_transforms_config=project_parameters.target_transforms_config,
        max_samples=project_parameters.max_samples,
        classes=project_parameters.classes,
        batch_size=project_parameters.batch_size,
        num_workers=project_parameters.num_workers,
        accelerator=project_parameters.accelerator,
        random_seed=project_parameters.random_seed)


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create datamodule
    datamodule = create_datamodule(project_parameters=project_parameters)

    # prepare data
    datamodule.prepare_data()

    # set up data
    datamodule.setup()

    # get train, validation, test dataset
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    test_dataset = datamodule.test_dataset

    # get the first sample and target in the train dataset
    x, y = train_dataset[0]

    # display the dimension of sample and target
    print('the dimension of sample: {}'.format(x.shape))
    print('the dimension of target: {}'.format(y.shape))
