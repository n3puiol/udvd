import glob
import os


class KITTIPose:
    def __init__(self, pose: str):
        self.raw_pose = pose.split()
        self.x = float(self.raw_pose[3])
        self.y = float(self.raw_pose[7])
        self.z = float(self.raw_pose[11])


class KITTIDataset:
    def __init__(self, path):
        self.path = path
        self.poses = self.load_poses()
        self.sequences = self.load_sequences()

    def load_poses(self):
        pose_files = sorted(glob.glob(os.path.join(self.path, "poses/*.txt")))
        poses = []
        for file in pose_files:
            with open(file, 'r') as f:
                poses.append([KITTIPose(pose) for pose in f.readlines()])
        return poses

    def load_sequences(self):
        sequences = []
        for seq in sorted(glob.glob(os.path.join(self.path, "sequences/*"))):
            seq_images = sorted(glob.glob(os.path.join(seq, "image_2/*.png")))
            sequences.append(seq_images)
        return sequences

    def __getitem__(self, index):
        if index >= len(self.poses):
            pose = None
        else:
            pose = self.poses[index]
        return self.sequences[index], pose
