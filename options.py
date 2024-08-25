# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import configargparse


class MonodepthOptions:

    def __init__(self):
        self.parser = configargparse.ArgumentParser()

        self.parser.add_argument('--config', is_config_file=True,
                                 help='config file path')
        self.parser.add_argument("--debug", action="store_true")
        self.parser.add_argument("--eval_only",
                                 help="if set, only evaluation",
                                 action="store_true")
        self.parser.add_argument("--local_rank", default=0, type=int)

        # paths
        self.parser.add_argument("--dataroot", 
                                 type=str, 
                                 help="the root for the ddad and nuscenes dataset",
                                 default='data/nuscenes')
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="nusc-depth")
        self.parser.add_argument("--log_dir", type=str, help="log directory", default='logs')

        # method options
        self.parser.add_argument("--volume_depth",
                                 action="store_true",
                                 help="if set, using the depth from volume rendering, rather than the depthdecoder")
        self.parser.add_argument("--voxels_size",
                                 type=int, nargs='+',
                                 default=[24, 300, 300],
                                 help='the resolution of the voxel for rendering： Z, Y, X = 24, 300, 300')
        self.parser.add_argument("--real_size",
                                 type=float, nargs='+',
                                 default=[-40, 40, -40, 40, -1, 5.4],
                                 help='the real scale of the voxel: XMIN, XMAX, ZMIN, ZMAX, YMIN, YMAX')

        self.parser.add_argument("--self_supervise", type=str, default='self', help="if set [gt, self, semi]")

        self.parser.add_argument("--loss", type=str, help="activation in the decoder [l1, sml1, silog, rl1]", default='silog')

        self.parser.add_argument("--gtw", type=float, help="the w of gt loss", default='1.0')


        self.parser.add_argument("--eval_occ",
                                 action="store_true",
                                 help="if set, eval the occupancy score")

        self.parser.add_argument("--contracted_ratio",
                                        type=float, default=0.8,
                                        help="the threshold for the contracted coordinate")

        
        self.parser.add_argument("--contracted_coord",
                                 type=lambda x: x.lower() == 'true', default=True,
                                 help="if set, using the contracted coordinate")
        
        self.parser.add_argument("--infinite_range",
                                 type=lambda x: x.lower() == 'true', default=True,
                                 help="sampling strategy for contracted coordinate")
        self.parser.add_argument("--auxiliary_frame",
                                 type=lambda x: x.lower() == 'true', default=True,
                                 help="if set, using auxiliary images")

        self.parser.add_argument("--use_semantic",
                                 help="if set, use semantic segmentation for training",
                                 type=lambda x: x.lower() == 'true', default=True)
        
        
        self.parser.add_argument("--semantic_classes",
                                 type=int, default=17,
                                 help="the output channel of the semantic_head")
        self.parser.add_argument("--class_frequencies",
                                 nargs="+", type=int,
                                 default= [0, 18164955800, 734218842, 2336187448, 10996756106, 395414611,
                                           638889260, 651023279, 117046208, 985341947, 7303776233, 43131984997,
                                           0, 9342674867, 9525824742, 51204832885, 22848065525, 31841090018])
        self.parser.add_argument("--semantic_sample_ratio",
                                 type=float, default=0.25,
                                 help="sample less points for semantic to accelerate the training")
                                 
        self.parser.add_argument("--last_free",
                                 action="store_true",
                                 help="if the last class is free space")

        # DATASET options
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="nusc")

        self.parser.add_argument("--cam_N",
                                 type=int,
                                 help="THE NUM OF CAM",
                                 default=6)

        self.parser.add_argument("--use_fix_mask", type=lambda x: x.lower() == 'true', default=False,
                                   help="if set, eval the occupancy score!")

        # OPTIMIZATION options
        self.parser.add_argument("--use_fp16",
                                 action="store_true",
                                 help="if set, using mixed precision training")
        
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=6)
        self.parser.add_argument("--B",
                                 type=int,
                                 help="real batch size",
                                 default=1)

        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--weight_decay",
                                 type=float,
                                 help="weight decay",
                                 default=0.0)

        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=12)

        self.parser.add_argument("--flow_epochs",
                                 type=int,
                                 help="number of epochs that begin the flow modeling",
                                 default=12)

        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=10)

        # DEPTH ESTIMATION options
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load, currently only support for 3 frames",
                                 default=[0, -1, 1])
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--scales",
                                 type=int, nargs="+",
                                 help="scales used in the loss",
                                 default=[0])

        # self.parser.add_argument("--pose_model_input",
        #                          type=str,
        #                          help="how many images the pose network gets",
        #                          default="pairs",
        #                          choices=["pairs", "all"])
        # self.parser.add_argument("--pose_model_type",
        #                          type=str,
        #                          help="normal or shared",
        #                          default="separate_resnet")

        # SYSTEM options
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=4)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")

        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of steps between each log",
                                 default=25)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="save frequency for visualization",
                                 default=100)
        self.parser.add_argument("--eval_frequency",
                                 type=int,
                                 help="number of steps between each save",
                                 default=1000)

        # RENDERING options
        self.parser.add_argument("--render_type",
                                 type=str,
                                 help="rednering by the density or probability [density, prob, neus, volsdf]",
                                 default='prob')
        self.parser.add_argument("--stepsize",
                                 type=float,
                                 help="stepsize (in voxel) for rendering",
                                 default=0.5)

        # HYPERPARAMETERS
        self.parser.add_argument("--semantic_loss_weight",
                                 type=float, default=0.05,
                                 help="the weight for the semantic loss")
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=0.001)
        
        self.parser.add_argument("--height_ori",
                                 type=int,
                                 help="original input image height",
                                 default=1216)
        self.parser.add_argument("--width_ori",
                                 type=int,
                                 help="original input image width",
                                 default=1936)

        self.parser.add_argument("--height",
                                 type=int, default=384,
                                 help="input image height")
        self.parser.add_argument("--width",
                                 type=int, default=640,
                                 help="input image width")
        self.parser.add_argument("--render_h",
                                 type=int, default=224,
                                 help="input image height")
        self.parser.add_argument("--render_w",
                                 type=int, default=352,
                                 help="input image width")
        
        self.parser.add_argument("--weight_entropy_last",
                                 type=float, default=0.0)
        self.parser.add_argument("--weight_distortion",
                                 type=float, default=0.0)
        self.parser.add_argument("--weight_sparse_reg",
                                 type=float, default=0.0)

        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=80.0)

        self.parser.add_argument("--min_depth_test",
                                 type=float,
                                 help="the min depth for the evaluation",
                                 default=0.1)
        self.parser.add_argument("--max_depth_test",
                                 type=float,
                                 help="the max depth for the evaluation",
                                 default=80.0)

        self.parser.add_argument("--en_lr",
                                 type=float,
                                 help="learning rate for encoder in volume rendering",
                                 default=0.0001)
        self.parser.add_argument("--de_lr",
                                 type=float,
                                 help="learning rate for decoder (3D CNN) in volume rendering",
                                 default=0.001)

        self.parser.add_argument("--aggregation",
                                 type=str,
                                 help="the type of the feature aggregation [mlp 3dcnn 2dcnn]",
                                 default= '3dcnn')

        self.parser.add_argument("--position", type=str,
                                 help="rednering by the density or probability [No, embedding, embedding1]",
                                 default='embedding')

        self.parser.add_argument("--data_type",  type=str,
                                 help=" data size for traing and testing - > [train_all, all, mini, tiny]",
                                 default='all')

        self.parser.add_argument("--input_channel", type=int, help="the final feature channel in the encoder",
                                 default=64)

        self.parser.add_argument("--con_channel", type=int, help="the final feature channel in the encoder",
                                 default=16)

        self.parser.add_argument("--out_channel", type=int, help="the output channel of the voxel",
                                 default=1)

        self.parser.add_argument("--encoder", type=str, help="the method for the comparison [101, 50]", default='101')
        
        # for gs
        self.parser.add_argument("--gs", type=str, default='No', help="if set, using gs rendering!")

        self.parser.add_argument("--gs_scale", type=float, default=0.02, help="the scale of gs")
        
        self.parser.add_argument("--spring", type=int,  default=0, help="if set, using spring offset in gs!")

        self.parser.add_argument("--enl", type=float, 
                                 default=0.0, help="the weight of the entropy loss for the guassian opacity!")
        
        self.parser.add_argument("--mask_val", type=float, default=0, help="the mask of  gs actiation, did not use")

        # flow
        self.parser.add_argument("--flow", type=str, default='no', help="if set, model the motion with flow field")
        self.parser.add_argument("--motion_scale", type=float,  default=3.0, help="the scale for the flow motion")

        # temporal
        self.parser.add_argument("--use_t",  type=str, default='No', help="if [No, 2d, 3d], do the temporal information fusion!")
        self.parser.add_argument("--view_trans", type=str,
                                 help="the manner for image space to 3D volume space [simple, query, bevformer]",
                                 default='simple')

        self.parser.add_argument("--upsample", type=lambda x: x.lower() == 'true', default=False, help="if set, using line space sample")
        self.parser.add_argument("--gt_pose",  type=str, default='yes', help="if no, use the gt pose from sensor, if yes, use the pose from the stage 1")
        self.parser.add_argument("--detach_pose", type=str, default='yes', help="if yes, detach the gradient (not optimize the pose net)!")


        # these are for stage 1, to be update
        self.parser.add_argument("--gs_prior", type=str, default='No', help="if set, ...")
        self.parser.add_argument("--gs_proj", type=str, default='No', help="if set, ...")
        self.parser.add_argument("--o_mask", type=str, default='No', help="if set, ...")
        self.parser.add_argument("--gs_depth", type=str, default='No', help="if set, ...")
    

        self.parser.add_argument("--rayiou", type=str, default='No', help="if set, do the rayiou evaluation in the final epoch!")
        self.parser.add_argument("--gs_sample", type=float, default=0, help="if set, 0 mean the xyz position is from the 2d lift!")
        self.parser.add_argument("--use_sfm_spatial", type=lambda x: x.lower() == 'true', default=False, help="if set, use sfm pseudo label")
        self.parser.add_argument("--evl_score", type=lambda x: x.lower() == 'true', default=False, help="if set, eval the occupancy score!")


    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
