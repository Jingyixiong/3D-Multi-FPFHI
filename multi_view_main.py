'''
Project point cloud into images. The subsampled points, intrinsic parameters,
projected images and the (h, w) idx of each projected points in point cloud
are all generated.
'''
import hydra

from omegaconf import DictConfig

@hydra.main(version_base=None, config_path='conf', config_name='alrunner')
def render_allarch(cfg: DictConfig):
    renderer = hydra.utils.instantiate(cfg.renderer)
    renderer.render_synarch()
    renderer.render_realarch()
    return True

if __name__ == '__main__':
    render_allarch()