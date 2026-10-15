"""Stage 1: ensemble detections with MS3D kernel box fusion.

Adapted from darrenjkt/MS3D (Apache-2.0). Paths are configuration driven.
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ps_cfg', type=Path, required=True,
                        help='YAML file containing MS3D parameters')
    parser.add_argument('--len', type=int, default=1,
                        help='Number of detection sets used for debugging')
    parser.add_argument('--save_dir', type=Path,
                        help='Override SAVE_DIR in the YAML file')
    parser.add_argument('--exp_name', type=str,
                        help='Reserved for compatibility with the MS3D interface')
    args = parser.parse_args()

    import yaml
    from pcdet.utils import box_fusion_utils, ms3d_utils

    with args.ps_cfg.open() as stream:
        parameters = yaml.safe_load(stream)
    detection_annotations = box_fusion_utils.load_src_paths_txt(parameters['DETS_TXT'])
    num_detection_sets = len(detection_annotations) - 1
    detection_sets = box_fusion_utils.get_detection_sets(detection_annotations, score_th=0.1)
    print(f'Number of detection sets: {num_detection_sets}')
    if args.len > 1:
        detection_sets = detection_sets[:args.len]

    class_config = {}
    for index, category in enumerate(box_fusion_utils.SUPERCATEGORIES):
        if category in class_config:
            continue
        class_config[category] = {
            'cls_id': index + 1,
            'discard': parameters['ENSEMBLE_KBF']['DISCARD'][index],
            'radius': parameters['ENSEMBLE_KBF']['RADIUS'][index],
            'nms': parameters['ENSEMBLE_KBF']['NMS'][index],
            'neg_th': parameters['PS_SCORE_TH']['NEG_TH'][index],
        }

    pseudo_labels = box_fusion_utils.get_initial_pseudo_labels(detection_sets, class_config)
    save_dir = args.save_dir or Path(parameters['SAVE_DIR'])
    output_name = 'initial_pseudo_labels.pkl'
    ms3d_utils.save_data(pseudo_labels, save_dir, name=output_name)
    print(f'Saved {save_dir / output_name}')


if __name__ == '__main__':
    main()
