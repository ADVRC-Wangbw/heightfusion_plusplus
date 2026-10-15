"""Stage 2: generate MS3D tracks from fused pseudo labels.

Adapted from darrenjkt/MS3D (Apache-2.0). Paths are configuration driven.
"""

import argparse
from pathlib import Path
import pickle
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ps_cfg', type=Path, required=True,
                        help='YAML file containing MS3D parameters')
    parser.add_argument('--cls_id', type=int, required=True,
                        help='1: vehicle, 2: pedestrian, 3: cyclist')
    parser.add_argument('--static_veh', action='store_true',
                        help='Generate static-vehicle tracks')
    parser.add_argument('--save_name', type=str,
                        help='Override the default output filename')
    args = parser.parse_args()

    import yaml
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.utils import ms3d_utils, tracker_utils

    with args.ps_cfg.open() as stream:
        parameters = yaml.safe_load(stream)
    cfg_from_yaml_file(parameters['DATA_CONFIG_PATH'], cfg)
    dataset = ms3d_utils.load_dataset(cfg, split='train')
    save_dir = Path(parameters['SAVE_DIR'])
    with (save_dir / 'initial_pseudo_labels.pkl').open('rb') as stream:
        pseudo_labels = pickle.load(stream)

    if args.cls_id == 1:
        if args.static_veh:
            track_config = tracker_utils.prepare_track_cfg(parameters['TRACKING']['VEH_STATIC'])
            output_name = 'tracks_world_veh_static.pkl'
        else:
            track_config = tracker_utils.prepare_track_cfg(parameters['TRACKING']['VEH_ALL'])
            output_name = 'tracks_world_veh_all.pkl'
    elif args.cls_id == 2:
        track_config = tracker_utils.prepare_track_cfg(parameters['TRACKING']['PEDESTRIAN'])
        output_name = 'tracks_world_ped.pkl'
    else:
        raise NotImplementedError('The upstream MS3D tracker supports vehicle and pedestrian stages')

    output_name = args.save_name or output_name
    tracks = tracker_utils.get_tracklets(dataset, pseudo_labels, track_config, cls_id=args.cls_id)
    ms3d_utils.save_data(tracks, save_dir, name=output_name)
    print(f'Saved {save_dir / output_name}')


if __name__ == '__main__':
    main()
