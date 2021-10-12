 def __init__(self, run_name='yelp_balanced.default'):
        print('Loading recommendation engine...')

        self.wandb_api = wandb.Api()
        run = self.wandb_api.run("jaxball/faber-ml/{}".format(run_name))
        self.config = AttrDict(run.config)

        self.device = self._setup_device(self.config.gpu)
        self.config.device = self.device
        self.df = pd.read_json('data/generated/review.json')
        self.model = get_model_by_name(self.config.model)(self.config)
        self.model = self.model.to(self.device)

        # loading checkpoint
        log_dir = os.path.join('./logs/', run_name)
        print("checkpoint logdir = {}".format(log_dir))
        ckpt_path, step = get_ckpt_path(log_dir, step=None)
        print('Loading checkpoint from {}...'.format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt['model'])
        print('Loaded checkpoint at step {}.'.format(ckpt['step']))