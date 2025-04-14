from uco3d import UCO3DDataset, UCO3DFrameDataBuilder
from uco3d.dataset_utils.utils import get_dataset_root
from uco3d.data_utils import load_whole_sequence
import os
os.environ['UCO3D_DATASET_ROOT'] = '/home/ddinucci/Desktop/uco3d/small'

def load_uco_data(scene_name, batch = 0):
    dataset_root = get_dataset_root(assert_exists=True)
    # Get the "small" subset list containing a small subset
    # of the uCO3D categories. For loading the whole dataset
    # use "set_lists_all-categories.sqlite".
    subset_lists_file = os.path.join(
        dataset_root,
        "set_lists", 
        "set_lists_test_10.sqlite",
    )
    uco_dataset = UCO3DDataset(
    subset_lists_file=subset_lists_file,
    subsets=["train"],
    #n_frames_per_sequence=100,
    #pick_sequences=[scene_name],
    frame_data_builder=UCO3DFrameDataBuilder(
        apply_alignment=False,
        load_images=True,
        load_depths=False,
        load_masks=True,
        load_depth_masks=False,
        load_gaussian_splats=False,
        gaussian_splats_truncate_background=False,
        load_point_clouds=False,
        load_segmented_point_clouds=False,
        load_sparse_point_clouds=False,
        box_crop=False,
        box_crop_context=0.4,
        load_frames_from_videos=True,
        image_height=512,
        image_width=512,
        undistort_loaded_blobs=False,
        use_cache=False,
        )
    )
    
    dataloader = load_whole_sequence(
        uco_dataset,
        scene_name,
        batch=batch,
        max_frames=100,
    )
    gs_splat, ply_path = load_uco_gs(scene_name, subset_lists_file)

    return {'data':dataloader, 'splats': gs_splat, 'ply_path': ply_path}

def load_uco_gs(scene_name, subset_lists_file):
    uco_dataset = UCO3DDataset(
    subset_lists_file=subset_lists_file,
    subsets=["train"],
    n_frames_per_sequence=1,
    pick_sequences=[scene_name],
    frame_data_builder=UCO3DFrameDataBuilder(
        apply_alignment=False,
        load_images=False,
        load_depths=False,
        load_masks=False,
        load_depth_masks=False,
        load_gaussian_splats=True,
        gaussian_splats_truncate_background=False,
        load_point_clouds=False,
        load_segmented_point_clouds=False,
        load_sparse_point_clouds=True,
        box_crop=False,
        box_crop_context=0.4,
        load_frames_from_videos=False,
        image_height=512,
        image_width=512,
        undistort_loaded_blobs=False,
        use_cache=False,
        )
    )

    return uco_dataset[0].sequence_gaussian_splats, uco_dataset[0].sequence_sparse_point_cloud_path

def create_transforms_from_uco():
    uco_data = load_uco_data()
    
    return True
