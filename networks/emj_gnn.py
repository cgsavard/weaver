import torch
from utils.nn.model.EmjTagger import EmjTagger


def get_model(data_config, **kwargs):
    conv_params = [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256)),
        ]
    fc_params = [(256, 0.1)]
    use_fusion = True

    features_dims = len(data_config.input_dicts['features'])
    num_classes = len(data_config.label_value)
    model = EmjTagger(features_dims, num_classes,
                      conv_params, fc_params,
                      use_fusion=use_fusion,
                      use_fts_bn=kwargs.get('use_fts_bn', False),
                      use_counts=kwargs.get('use_counts', True),
                      input_dropout=kwargs.get('input_dropout', None),
                      for_inference=kwargs.get('for_inference', False)
                              )

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['softmax'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
        }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
