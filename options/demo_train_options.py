from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        self.parser.add_argument('--dataroot', required=True, help='path to meshes (should have subfolders train, test)')
        self.parser.add_argument('--dataset_mode', choices={"classification", "segmentation"}, default='classification')
        self.parser.add_argument('--ninput_edges', type=int, default=33, help='# of input edges (will include dummy edges)')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples per epoch')
        # network params
        self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        self.parser.add_argument('--arch', type=str, default='mconvnet', help='selects network to use') #todo add choices
        self.parser.add_argument('--resblocks', type=int, default=0, help='# of res blocks')
        self.parser.add_argument('--fc_n', type=int, default=100, help='# between fc and nclasses') #todo make generic
        self.parser.add_argument('--ncf', nargs='+', default=[8, 32, 32], type=int, help='conv filters')
        self.parser.add_argument('--pool_res', nargs='+', default=[12, 10, 8], type=int, help='pooling res')
        self.parser.add_argument('--norm', type=str, default='batch',help='instance normalization or batch normalization or group normalization')
        self.parser.add_argument('--num_groups', type=int, default=16, help='# of groups for groupnorm')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        # general params
        self.parser.add_argument('--num_threads', default=3, type=int, help='# threads for loading data')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='debug', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes meshes in order, otherwise takes them randomly')
        self.parser.add_argument('--seed', type=int, help='if specified, uses seed')
        # visualization params
        self.parser.add_argument('--export_folder', type=str, default='', help='exports intermediate collapses to this folder')
        #
        self.initialized = True

        self.parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=250, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--run_test_freq', type=int, default=1, help='frequency of running test in training script')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=500, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        # data augmentation stuff
        self.parser.add_argument('--num_aug', type=int, default=1, help='# of augmentation files') # changed 10 -> 1
        self.parser.add_argument('--scale_verts', action='store_false', help='non-uniformly scale the mesh e.g., in x, y or z') # changed store_true to store_false
        self.parser.add_argument('--slide_verts', type=float, default=0, help='percent vertices which will be shifted along the mesh surface')
        self.parser.add_argument('--flip_edges', type=float, default=0, help='percent of edges to randomly flip')
        # tensorboard visualization
        self.parser.add_argument('--no_vis', action='store_true', help='will not use tensorboard')
        self.parser.add_argument('--verbose_plot', action='store_true', help='plots network weights, etc.')
        self.is_train = True
