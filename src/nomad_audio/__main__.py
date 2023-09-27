import click
from nomad_audio import nomad

@click.command()
@click.option('--nmr_path', type=str, help='Path to non-matching reference files')
@click.option('--test_path', type=str, help='Path to test files')
@click.option('--results_path', type=str, default=None, help='Used to specify a path file where to save both averaged Nomad scores csv and Nomad scores csv for each non-matching reference used. Default uses a current datetime format in results-csv.')
@click.option('--device', type=str, default=None, help='Specify device, cuda or cpu. Automatically set cuda if None and GPU is detected')
def main(nmr_path, test_path, results_path, device):
    
    # Predict nomad scores
    nomad_avg, nomad_scores = nomad.predict(nmr_path, test_path, results_path)
    print('Nomad average scores, printing top 5 test files')
    print(nomad_avg.head())

if __name__ == "__main__":
    main()