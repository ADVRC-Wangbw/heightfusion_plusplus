"""Stage 3: apply MS3D temporal refinement to fused pseudo labels.

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
    parser.add_argument('--save_veh_intermediate_tracks', action='store_true',
                        help='Save refined vehicle tracks after each temporal stage')
    args = parser.parse_args()

    import yaml
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.utils import ms3d_utils

    with args.ps_cfg.open() as stream:
        parameters = yaml.safe_load(stream)
    cfg_from_yaml_file(parameters['DATA_CONFIG_PATH'], cfg)
    dataset = ms3d_utils.load_dataset(cfg, split='train')
    save_dir = Path(parameters['SAVE_DIR'])
    pseudo_labels = ms3d_utils.load_pkl(save_dir / 'initial_pseudo_labels.pkl')
    tracks_vehicle = ms3d_utils.load_pkl(save_dir / 'tracks_world_veh_all.pkl')
    tracks_static = ms3d_utils.load_pkl(save_dir / 'tracks_world_veh_static.pkl')
    tracks_pedestrian = ms3d_utils.load_pkl(save_dir / 'tracks_world_ped.pkl')
    frame_ids = list(pseudo_labels)

    print('Refining vehicle labels')
    tracks_vehicle, tracks_static = ms3d_utils.refine_veh_labels(
        dataset, frame_ids, tracks_vehicle, tracks_static,
        static_trk_score_th=parameters['TRACKING']['VEH_STATIC']['RUNNING']['SCORE_TH'],
        veh_pos_th=parameters['PS_SCORE_TH']['POS_TH'][0],
        refine_cfg=parameters['TEMPORAL_REFINEMENT'],
        save_dir=save_dir if args.save_veh_intermediate_tracks else None,
    )
    print('Refining pedestrian labels')
    tracks_pedestrian = ms3d_utils.refine_ped_labels(
        tracks_pedestrian,
        ped_pos_th=parameters['PS_SCORE_TH']['POS_TH'][1],
        track_filtering_cfg=parameters['TEMPORAL_REFINEMENT']['TRACK_FILTERING'],
    )
    print('Combining pseudo-labels for each class')
    final_labels = ms3d_utils.update_ps(
        dataset, pseudo_labels, tracks_vehicle, tracks_static, tracks_pedestrian,
        veh_pos_th=parameters['PS_SCORE_TH']['POS_TH'][0], veh_nms_th=0.05,
        ped_nms_th=0.5, frame2box_key_static='frameid_to_propboxes',
        frame2box_key='frameid_to_box', frame_ids=frame_ids,
    )
    final_labels = ms3d_utils.select_ps_by_th(final_labels, parameters['PS_SCORE_TH']['POS_TH'])
    output_name = 'final_ps_dict.pkl'
    ms3d_utils.save_data(final_labels, save_dir, name=output_name)
    print(f'Saved {save_dir / output_name}')


if __name__ == '__main__':
    main()
