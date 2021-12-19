import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
from typing import NamedTuple, Tuple, List, Callable
from tqdm import tqdm
import io
from PIL import Image
from skimage.filters import difference_of_gaussians


def normal(mean, std, alpha=1, beta=0, flip=False):
    def sampler(prev_level, cur_level):
        flip_factor = 1 if np.random.rand() > 0.5 or not flip else -1

        alpha_factor = 1
        add = 0

        if prev_level != cur_level:
            alpha_factor = alpha ** cur_level
            add = beta

        return flip_factor * (np.random.normal(mean, std) * alpha_factor + add)

    return sampler


class GrowthConfig(NamedTuple):
    # samplers
    length_sampler: Callable[[int], float]
    angle_sampler: Callable[[int], float]
    thickness_sampler: Callable[[int], float]
    branch_energy_sampler: Callable[[int], float]

    # parameters
    branchiness: int
    min_thickness: float
    max_level: int
    max_num_branches: int


class BranchSegment(NamedTuple):
    start: np.ndarray
    end: np.ndarray
    thickness: float
    branch_energy: float
    level: int

    @property
    def length(self):
        return np.linalg.norm(self.end - self.start)

    @property
    def angle(self):
        return np.arctan2(self.end[1] - self.start[1], self.end[0] - self.start[0])

    def _grow_one(self, branching: bool, config: GrowthConfig) -> 'BranchSegment':
        new_level = self.level + (0 if branching else 1)
        new_length = self.length * config.length_sampler(self.level, new_level)
        new_angle = self.angle + config.angle_sampler(self.level, new_level)
        new_thickness = self.thickness * config.thickness_sampler(self.level, new_level)

        if branching:
            new_energy = 0
        else:
            new_energy = self.branch_energy + config.branch_energy_sampler(self.level, new_level)

        new_end = np.array([np.cos(new_angle), np.sin(new_angle)]) * new_length + self.end
        return BranchSegment(self.end, new_end, new_thickness, new_energy, new_level)

    def grow(self, config: GrowthConfig) -> List['BranchSegment']:
        new_branches = []

        for _ in range(config.branchiness):
            branching_power = np.random.rand()
            if branching_power > self.branch_energy:
                new_branches.append(self._grow_one(True, config))

        new_branches.append(self._grow_one(False, config))
        return new_branches


def make_tree(config, ax, wood_color, needle_color):
    branch = BranchSegment(
        start=np.array([0, 0]),
        end=np.array([0, 1]),
        thickness=10,
        branch_energy=0,
        level=0
    )

    queue = [branch]

    total = 0
    line_segments = []
    line_widths = []
    line_colors = []

    while len(queue) > 0:
        branch = queue.pop(0)

        if branch.level <= config.max_level and branch.thickness > config.min_thickness and total < config.max_num_branches:
            queue.extend(branch.grow(config))

        line_segments.append([branch.start, branch.end])
        line_widths.append(branch.thickness)
        line_colors.append(wood_color * (0.9 + 0.2 * np.random.rand()))

        if branch.level > 2:
            num_needles = 3
            angles = np.random.rand(num_needles) * 2 * np.pi
            lengths = np.random.rand(num_needles) * 0.2 + 0.5
            start_alphas = np.random.rand(num_needles)
            starts = start_alphas[:, None] * branch.start[None] + (1 - start_alphas[:, None]) * branch.end[None]
            ends = starts + np.stack([np.cos(angles), np.sin(angles)], axis=1) * lengths[:, None]

            line_segments.extend([[s, e] for s, e in zip(starts, ends)])
            line_widths.extend([0.2] * num_needles)
            line_colors.extend(needle_color[None] * (0.9 + 0.2 * np.random.rand(num_needles, 3)))

        total += 1

    lc = mc.LineCollection(line_segments, colors=line_colors, linewidths=line_widths, joinstyle="round",
                           capstyle="round")
    ax.add_collection(lc)
    ax.autoscale()
    ax.axis('equal')
    ax.axis('off')


day_wood_color = np.array([95 / 255, 61 / 255, 45 / 255])
day_needle_color = np.array([39 / 255, 102 / 255, 65 / 255])
day_bg_color = "#DFEDF7"

night_wood_color = np.array([89 / 255, 65 / 255, 48 / 255])
night_needle_color = np.array([55 / 255, 69 / 255, 42 / 255])
night_bg_color = "#262A2C"


def frostify(fig):
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png', facecolor=night_bg_color)
    plt.close(fig)

    im = Image.open(img_buf)
    arr = np.array(im).copy().astype(np.float32)
    img_buf.close()

    frost = difference_of_gaussians(arr[..., 1], 0.1, 5)
    frost[frost < 0] = 0
    frost /= frost.max()

    snow = difference_of_gaussians(arr[..., 1], 4.8, 5)
    snow = np.gradient(snow)[0]
    snow[snow > 0.1] += np.random.rand(*snow[snow > 0.1].shape) * 0.1

    snow[snow < 0] = 0
    snow /= snow.max()
    snow = snow ** 2

    gradient = np.linspace(0, 1, snow.shape[1])[:, None]
    snow += difference_of_gaussians(np.random.rand(*snow.shape) ** 2 * gradient, 5, 10) + np.random.rand(
        *snow.shape) * 0.2 * (0.05 + gradient)

    snow_image = arr[..., :3] / 255 + snow[..., None] * 0.8 + frost[..., None] * 0.7
    snow_image /= snow_image.max(axis=(0, 1))
    return snow_image


pine_config = GrowthConfig(
    length_sampler=normal(mean=0.99, std=0.01, alpha=0.5),
    angle_sampler=normal(mean=0, std=0.08, alpha=2.5, beta=np.pi / 2, flip=True),
    thickness_sampler=normal(mean=0.95, std=0.1, alpha=0.5),
    branch_energy_sampler=normal(mean=0.01, std=0.01, alpha=1.2),
    branchiness=1,
    min_thickness=0.1,
    max_level=2,
    max_num_branches=20_000
)

tree_config = GrowthConfig(
    length_sampler=normal(mean=0.99, std=0.01, alpha=0.5),
    angle_sampler=normal(mean=0, std=0.3, alpha=1.2, flip=True),
    thickness_sampler=normal(mean=0.95, std=0.1, alpha=0.5),
    branch_energy_sampler=normal(mean=0.001, std=0.001, alpha=1.2),
    branchiness=1,
    min_thickness=0.1,
    max_level=2,
    max_num_branches=20_000
)

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--first_index", type=int)
    parser.add_argument("--last_index", type=int)
    parser.add_argument("--pine", action="store_true")
    parser.add_argument("--tree", action="store_true")
    parser.add_argument("--day", action="store_true")
    parser.add_argument("--night", action="store_true")

    args = parser.parse_args()
    assert args.pine != args.tree
    assert args.day != args.night

    name = "pine" if args.pine else "tree"
    name += "_day" if args.day else "_night"
    config = pine_config if args.pine else tree_config

    os.makedirs("output", exist_ok=True)

    with tqdm(total=args.last_index, initial=args.first_index) as pbar:
        for i in range(args.first_index, args.last_index + 1):
            if args.day:
                fig = plt.figure(figsize=(10, 10))
                make_tree(config, plt.gca(), day_wood_color, day_needle_color)
                fig.savefig(f"output/{name}_{i}.jpg", format='jpg', facecolor=day_bg_color)
                plt.close()
            else:
                fig = plt.figure(figsize=(10, 10))
                make_tree(config, plt.gca(), night_wood_color, night_needle_color)
                im = frostify(fig)
                plt.close()
                Image.fromarray((im * 255).astype(np.uint8)).save(f"output/{name}_{i}.jpg")

            pbar.update(1)
