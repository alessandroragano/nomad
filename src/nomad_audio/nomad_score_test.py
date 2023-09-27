from nomad_audio import nomad

def test_nomad_score():
    nmr_path = 'data/nmr-data'
    test_path = 'data/test-data'

    avg_score, scores = nomad.predict(nmr_path, test_path)
    
    print(avg_score)
    print('\n')
    print(scores)