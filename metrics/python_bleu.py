from nltk.translate.bleu_score import corpus_bleu, closest_ref_length, brevity_penalty

def calculate_bleu(list_of_references, hypotheses):
    '''
    by Qiu
    '''
    hyp_lengths, ref_lengths = 0, 0
    for references, hypothesis in zip(list_of_references, hypotheses):
        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)
    try:
        bp = brevity_penalty(ref_lengths, hyp_lengths)
        bleu1 = corpus_bleu(list_of_references, hypotheses, weights=(1, 0, 0, 0)) / bp * 100
        bleu2 = corpus_bleu(list_of_references, hypotheses, weights=(0, 1, 0, 0)) / bp * 100
        bleu3 = corpus_bleu(list_of_references, hypotheses, weights=(0, 0, 1, 0)) / bp * 100
        bleu4 = corpus_bleu(list_of_references, hypotheses, weights=(0, 0, 0, 1)) / bp * 100
        bleu_all = corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25)) * 100
    except Exception:
        bleu_all, bleu1, bleu2, bleu3, bleu4 = 0.0, 0.0, 0.0, 0.0, 0.0
    
    return bleu_all, bleu1, bleu2, bleu3, bleu4
