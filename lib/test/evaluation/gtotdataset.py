import numpy as np
from lib.test.evaluation.data import Sequence,  BaseDataset, SequenceList
from lib.test.utils.load_text import load_text
import os
class GTOTDataset(BaseDataset):
    # GTOT dataset
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.gtot_path
        self.sequence_list = ['Jogging','Otcbvs','Gathering','occBike','LightOcc','Walking','GarageHover','Torabi','Quarreling','BusScale','Minibus','tunnel','RainyMotor1','WalkingOcc','OccCar-1','Otcbvs1','Pool','Cycling','WalkingNig','crowdNig','FastCarNig','WalkingNig1','carNig','OccCar-2','Exposure4','Motorbike','BlackSwan1','GoTogether','RainyCar2','Crossing','Tricycle','BlackCar','BlueCar','DarkNig','Football','MotorNig','MinibusNigOcc','Exposure2','MinibusNig','Motorbike1','RainyPeople','RainyMotor2','Running','Torabi1','FastMotor','BusScale1','Minibus1','FastMotorNig','RainyCar1','fastCar2']
        
    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/init.txt'.format(self.base_path, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter='\t', dtype=np.float64)

        frames_path_i = '{}/{}/i'.format(self.base_path, sequence_name)
        frames_path_v = '{}/{}/v'.format(self.base_path, sequence_name)

        frame_list_i = sorted([p for p in os.listdir(frames_path_i) if os.path.splitext(p)[1].lower() in ['.jpg', '.png', '.bmp']])
        # print("frame_list_i",frame_list_i)
        frame_list_v = sorted([p for p in os.listdir(frames_path_v) if os.path.splitext(p)[1].lower() in ['.jpg', '.png', '.bmp']])
        frames_list_i = [os.path.join(frames_path_i, frame) for frame in frame_list_i]
        # print("frames_list_i",frames_list_i)
        frames_list_v = [os.path.join(frames_path_v, frame) for frame in frame_list_v]
        frames_list = [frames_list_v, frames_list_i]
        return Sequence(sequence_name, frames_list, 'gtot', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        with open('{}List.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()

        if split == 'ltrval':
            with open('{}/got10k_val_split.txt'.format(self.env_settings.dataspec_path)) as f:
                seq_ids = f.read().splitlines()

            sequence_list = [sequence_list[int(x)] for x in seq_ids]
        return sequence_list
    