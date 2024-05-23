import os.path

import numpy as np
from tqdm.contrib.concurrent import process_map
import argparse
from loguru import logger


class Evaluator:
    def __init__(self, gt_root: str, pred_root: str, num_workers: int):
        self.gt_root = gt_root
        self.pred_root = pred_root
        self.file_list = os.listdir(self.gt_root)
        self.num_workers = num_workers
        self.dices = {}

    def run(self):
        self.dices = dict(process_map(self._solve, range(len(self.file_list)), max_workers=self.num_workers,
                                      chunksize=1))
        logger.info(f'Overall Mean Dice:{self._eval()}')

    def _solve(self, item):
        gt = np.load(os.path.join(self.gt_root, self.file_list[item]))
        pred = np.load(os.path.join(self.pred_root, self.file_list[item]))
        gt = gt['tumor_mask'][2]
        gt[gt > 0] = 1
        pred = pred['tumor_mask'][2]
        pred[pred > 0] = 1
        x = gt.sum()
        y = pred.sum()
        z = (gt * pred).sum()
        dice = 2 * z / (x + y + 1e-7)
        if x == 0 and y == 0:
            dice = 1
        elif x == 0 or y == 0:
            dice = 0
        return self.file_list[item], dice

    def _eval(self):
        logger.info(self.dices)
        return np.mean(np.stack([v for k, v in self.dices.items()]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="hi_measure")
    parser.add_argument('--gt_dir', type=str, default='/data4/liver_CT4_Z2_ts2.2', help='gt root dir')
    parser.add_argument('--pre_dir', type=str, default='/data4/ycy/experiment/SwinUNETR_1/outputs_npz', help='pred root dir')
    parser.add_argument('--output_dir', type=str, default=None, help='output root dir')
    args = parser.parse_args()
    
    if not args.output_dir:
        args.output_dir = args.pre_dir.rsplit('/', 1)[0]
    logger.add(os.path.join(args.output_dir,  'test_log.txt'))
    
    solver = Evaluator(gt_root=args.gt_dir,
                       pred_root=args.pre_dir,
                       num_workers=10)
    solver.run()
