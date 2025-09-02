Composable Score-based Graph Diffusion Model for Multi-Conditional Molecular Generation
================================================================
This is the code for CSGD (**C**omposable **S**core-based **G**raph **D**iffusion model):

<div style="display: flex; justify-content: center;" markdown="1">
      <img src="asset/framework.png" style="width: 100%;" alt="Description of the first image">
 
</div>

## Requirements
All dependencies are specified in the `requirements.txt` file.

This code was developed and tested with Python 3.9.16, PyTorch 2.0.0, and PyG 2.3.0, Pytorch-lightning 2.0.1.

For molecular generation evaluation, we should first install rdkit.

Then `fcd_torch`: `pip install fcd_torch` (https://github.com/insilicomedicine/fcd_torch).

And `mini_moses` package: `pip install git+https://github.com/igor-krawczuk/mini-moses` (https://github.com/igor-krawczuk/mini-moses),

## Usage

We could train the model on an A800 GPU card. Here is an example to running the code for polymer graphs:

```
python main.py --config-name=config.yaml \
                model.ensure_connected=True \
                dataset.task_name='O2-N2-CO2' \
                dataset.guidance_target='O2-N2-CO2'
```
All default configurations can be found in `configs`.

Other examples for small molecule generation:

```
python main.py --config-name=config.yaml \
                dataset.task_name='bace_b' \
                dataset.guidance_target='Class'

python main.py --config-name=config.yaml \
                dataset.task_name='bbbp_b' \
                dataset.guidance_target='p_np'

python main.py --config-name=config.yaml \
                dataset.task_name='hiv_b' \
                dataset.guidance_target='HIV_active'
```

