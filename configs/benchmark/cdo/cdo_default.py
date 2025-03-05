from .cdo_256_100e import cfg as cfg_base


class cfg(cfg_base):
    def __init__(self, model_name = 'mshrnet32', gamma = 0,
                 oom = True, generate_anomaly = True, mode='semi10', loss='Lcdo'):
        cfg_base.__init__(self)

        model_name_list = ['mshrnet18',
                           'mshrnet32',
                           'mshrnet48',
                           'timm_resnet18',
                           'timm_resnet32',
                           'timm_resnet50',
                           'timm_wide_resnet50_2']

        ###### Train DATA
        self.train_data.type = 'DefaultAD'
        self.train_data.name = 'mvtec'  # mvtec, visa
        self.train_data.mode = mode
        self.train_data.cls_names = ['carpet']
        self.train_data.anomaly_generator.name = 'white_noise'
        self.train_data.anomaly_generator.enable = generate_anomaly  ### TODO
        self.train_data.anomaly_generator.kwargs = dict()
        self.train_data.sampler.name = 'naive'
        self.train_data.sampler.kwargs = dict()

        ###### Test DATA
        self.test_data.type = 'DefaultAD'
        self.test_data.name = 'mvtec'  # mvtec, visa
        self.test_data.mode = mode
        self.test_data.cls_names = ['carpet']

        if 'mshrnet' in model_name:
            self.model_t.name = model_name
            self.model_t.kwargs = dict(pretrained=True, checkpoint_path='', strict=False)

            self.model_s.name = model_name
            self.model_s.kwargs = dict(pretrained=False, checkpoint_path='', strict=False)
        elif 'resnet' in model_name:
            self.model_t.name = model_name
            self.model_t.kwargs = dict(pretrained=True, checkpoint_path=None,
                                       strict=False, features_only=True, out_indices=[1, 2, 3])

            self.model_s.name = model_name
            self.model_s.kwargs = dict(pretrained=False, checkpoint_path=None,
                                       strict=False, features_only=True, out_indices=[1, 2, 3])

        if loss == 'L2':
            self.loss.loss_terms = [
                dict(type='L2Loss', name='cdo_loss'),
            ]
        elif loss == 'Lcdo':
            self.loss.loss_terms = [
                dict(type='CDOLoss', name='cdo_loss', gamma=gamma, OOM=oom),
            ]  ### TODO
