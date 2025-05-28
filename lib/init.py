class pathsBib: 
    """
    Class to store paths.
    """ 
    def __init__(self, config):
        """
        Initialize paths.
        """
        self.config_dir = 'configs/'
        self.data_dir = 'data/'
        self.data_path = self.data_dir + config['data_name'] + '.h5'
        if config['latent_type'] == 'dls':
            self.latent_id = 'dls_p'  + str(config['latent_params']['patch_size']) + 'm' + str(config['latent_params']['num_modes'])
        else:
            self.latent_id = 'pod_m' + str(config['latent_params']['num_modes']) 
        self.latent_dir = 'results/' + config['data_name'] + '/' + self.latent_id + '/'
        self.latent_path = self.latent_dir + 'latent_coeff.h5'

        self.model_id = config['model']
        self.model_id += '_t' + str(config['params']['time_lag'])
        self.model_id += '_ta' + str(config['train']['train_ahead'])
        self.model_id += '_l' + str(config['params']['num_layers'])
        self.model_id += '_d' + str(config['params']['d_model']) 
        self.model_id += '_h' + str(config['params']['nhead']) 
        self.model_id += '_train' + str(config['train']['train_split']) 
        self.model_id += '_test' + str(config['train']['test_split']) 
        self.model_id += '_Ntrain' + str(config['train']['sample_train']) 
        self.model_id += '_Ntest' + str(config['train']['sample_test'])
        self.model_id = self.model_id.replace('.', '_')

        self.model_dir = self.latent_dir + self.model_id + '/'
        self.log_path = self.model_dir + config['mode'] + '.log'
        self.model_path = self.model_dir + 'model.pth'
        self.checkpoint_dir = self.model_dir + 'checkpoints/'
        self.checkpoint_path = self.checkpoint_dir + 'checkpoint.tar'
        self.predictions_dir = self.model_dir + 'pred/'
        self.fig_dir = self.model_dir + 'figs/'
        self.anim_dir = self.model_dir + 'anim/'
        self.metrics_dir = self.model_dir + 'saved_metrics/'
        


def init_path(config):
    """
    Initialisation of all the paths 

    Returns:
        pathsBib        :   (class) class containing all the paths
        is_init_path    :   (bool) if initialise success
    """
    import os 
    from pathlib import Path
    
    paths_bib = pathsBib(config)
    is_init_path = False
    try:
        # print(f"{'#'*20}\t{'Init paths...':<20}\t{'#'*20}")
        path_list =[i for key,i in paths_bib.__dict__.items() if type(i)==str and "/" in i and '_dir' in key]
        for pth in path_list:
            Path(pth).mkdir(exist_ok=True, parents=True)
            # print(f"INIT:\t{pth}\tDONE")
            
        is_init_path = True
    except:
        print(f"Error: Failed to create full path list")


    return is_init_path, paths_bib