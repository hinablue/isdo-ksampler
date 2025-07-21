from modules import sd_samplers_kdiffusion, sd_samplers_common
from backend.modules import k_diffusion_extra


class AlterSampler(sd_samplers_kdiffusion.KDiffusionSampler):
    def __init__(self, sd_model, sampler_name):
        self.sampler_name = sampler_name
        self.unet = sd_model.forge_objects.unet
        sampler_function = getattr(k_diffusion_extra, "sample_{}".format(sampler_name))
        super().__init__(sampler_function, sd_model, None)


def build_constructor(sampler_name):
    def constructor(m):
        return AlterSampler(m, sampler_name)

    return constructor


# 延遲導入 ISDO 採樣器以避免循環導入
def get_isdo_samplers():
    """延遲獲取 ISDO 採樣器數據"""
    try:
        from .isdo_samplers_integration import samplers_data_isdo
        print("成功導入 ISDO 採樣器")
        return samplers_data_isdo
    except ImportError as e:
        print(f"無法導入 ISDO 採樣器: {e}")
        return []
    except Exception as e:
        print(f"導入 ISDO 採樣器時發生錯誤: {e}")
        return []


# 延遲獲取 ISDO 採樣器
samplers_data_isdo = get_isdo_samplers()

samplers_data_alter = [
    sd_samplers_common.SamplerData('DDPM', build_constructor(sampler_name='ddpm'), ['ddpm'], {}),
    *samplers_data_isdo,  # 添加 ISDO 採樣器
]

# 輸出可用的採樣器信息
if samplers_data_isdo:
    print(f"已添加 {len(samplers_data_isdo)} 個 ISDO 採樣器到 alter_samplers")
    for sampler in samplers_data_isdo:
        print(f"  - {sampler.name}")
else:
    print("ISDO 採樣器不可用，僅使用標準 alter samplers")