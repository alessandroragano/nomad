from nomad import Nomad
nomad = Nomad()

def test_nomad_score():
    # Test dir mode
    nmr_path = 'data/nmr-data'
    test_path = 'data/test-data'

    avg_score, scores = nomad.predict('dir', nmr_path, test_path)
    
    print(avg_score)
    print('\n')
    print(scores)
    
    # Test csv mode
    nmr_csv = 'data/nmr_file.csv'
    test_csv = 'data/test_file.csv'

    avg_score, scores = nomad.predict('csv', nmr_csv, test_csv)
    
    print(avg_score)
    print('\n')
    print(scores)

test_nomad_score()