from jiwer.measures import cer, wer

def compute_metrics(hypothesis: list[str], references: list[str]) -> dict:
    cer_score = cer(references, hypothesis)
    wer_score = wer(references, hypothesis)

    return {
        "CER": cer_score*100,
        "WER": wer_score*100
    }
