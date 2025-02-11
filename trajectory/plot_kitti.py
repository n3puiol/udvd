from trajectory.kitti_dataset import KITTIDataset
import matplotlib.pyplot as plt


def plot_pose(sequence_nb: int, dataset: KITTIDataset):
    _, poses = dataset[sequence_nb]
    if poses is None:
        print(f"Sequence {sequence_nb} does not have poses")
        return
    print(f"Sequence {sequence_nb} has {len(poses)} poses")

    for pose in poses:
        plt.plot(pose.x, pose.z, 'ro', markersize=1)

    plt.xlabel("x")
    plt.ylabel("z")

    plt.title(f"Ground truth trajectory of sequence {sequence_nb}")

    plt.show()


def main():
    dataset = KITTIDataset("../datasets/KITTI")
    plot_pose(0, dataset)


if __name__ == "__main__":
    main()
