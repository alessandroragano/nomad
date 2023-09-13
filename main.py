#! /home/alergn/virtualenvs/torchgpu/bin/python3.9
import click
import sys
import yaml

@click.command()
@click.option('--config_file', type=str)
def training(config_file):
    
    # Open config file
    with open(config_file) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # Import script (this changes based on which config file arg is passed by the user)
    module_name = config['training_script']
    __import__(module_name)

    # Make imported module variables available
    imported_module = sys.modules[module_name]

    # Create new training object
    train_obj = imported_module.Training(config_file)

    # *** EXPERIMENT CHOICE ***
    # TRAINING NOMAD
    if config['experiment_name'] == 'Training':
        train_obj.training_loop()
    
    # PERFORMANCE EVALUATION
    # NMR AUDIO QUALITY
    elif config['experiment_name'] == 'quality_nmr':
        train_obj.eval_audio_quality(config['nomad_model_path'])
    
    # NMR RANKING VALIDATION SET CONDITIONS 
    elif config['experiment_name'] == 'valid_rank':
        train_obj.eval_degr_level(config['nomad_model_path'])
    
    # NMR RANKING DEGRADATION INTENSITY
    elif config['experiment_name'] == 'intensity':
        train_obj.eval_degradation_intensity(config['nomad_model_path'])

    # FULL REFERENCE AUDIO QUALITY
    elif config['experiment_name'] == 'quality_fr':
        train_obj.eval_full_reference(config['nomad_model_path'])

    # PLOT PERFORMANCE (to check)
    elif config['experiment_name'] == 'Plot':
        train_obj.plot_performance()

if __name__ == '__main__':
    training()  