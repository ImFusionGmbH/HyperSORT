import numpy as np
from torch.utils.data import Dataset
import imfusion
import imfusion.machinelearning as ml

from utils import *



class ImFusionIODataset(Dataset):
    """
    Dataset using ImFusion io and pre-processing capabilities

    Parses a data list file which is a tab separated csv file in the form:
    datafield0    datafield1    ...    datafieldN
    file_patient0_0    file_patient0_1    file_patient0_N
    ...
    file_patientM_0    file_patientM_1    file_patientM_N

    Each patient dataset is pre-processed following the pipeline specified in pipeline.yaml

    Documentation for the ImFusion SDK is available here: https://docs.imfusion.com/python/installing.html

    Args:
        data_file (str): path to data list file
        fields (dict): dictionary with 2 entries ('image' and 'label') respectively indicating which datafield should be considered for the network input and label maps
        pipeline (str): path to the pipeline preprocessing file 
    """

    def __init__(self, data_file: str, fields: dict[str, list[str]], pipeline: list[dict], phase: ml.Phase = ml.Phase.TRAIN) -> None:
        with open(data_file) as f:
            data_list = f.read()
            data_list = data_list.split("\n")
        data_list = [x.split("\t") for x in data_list if x != ""]
        data_list = np.array(data_list)
        
        self.fields = fields

        self.files = data_list[1:]

        def parse_operation(operation: dict[str, Any]) -> tuple[str, imfusion.Properties, ml.Phase]:
            """
            parses the operation specification and warps it into an operation spec
            """
            operation_name, configuration = next(iter(operation.items()))
            phase = configuration.pop("phase", ml.Phase.TRAIN)
            return (operation_name, imfusion.Properties(configuration), phase)
        
        self.preprocessing_pipeline = ml.OperationsSequence([parse_operation(operation) for operation in pipeline])

        self.phase = phase

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        
        dataitem = ml.DataItem({
            field: ml.ImageElement(imfusion.load(self.files[idx, i])[j]) for (field, i, j) in self.fields
        })
        self.preprocessing_pipeline.process(dataitem, self.phase)
        return {
            field: element.torch()[0] for (field, element) in dataitem.items()
        } | {"data_identifier": idx}


