import hydra

from omegaconf import DictConfig

@hydra.main(version_base=None, config_path='conf', config_name='alrunner')
def main(cfg: DictConfig):
    al_runner = hydra.utils.instantiate(cfg.al_detector)
    if cfg.general.inspect_target == 'real_arch':
        infra_names = cfg.general.realarch_names
    elif cfg.general.inspect_target == 'syn_arch':
        infra_names = cfg.general.synarch_names
    elif cfg.general.inspect_target == 'tunnel':
        infra_names = cfg.general.tunnel_names
    else:
        raise ValueError('The dataset name is not correct.')
    
    s_dict_list, metrics_dict_list = al_runner.anomaly_detect(cfg.general.inspect_target, infra_names)
    return s_dict_list, metrics_dict_list  
      
if __name__ == '__main__':
    main()

   