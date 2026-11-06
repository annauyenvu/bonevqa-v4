"""Same predeclared loss evaluation, bound to clean recovery v3_3 only.

Old preparation remains preserved and cannot be mistaken for the recovery.
"""
import argparse
from pathlib import Path
from experiments import evaluate_B_sequence_loss_pilots_v3_1 as evaluation

evaluation.__file__=str(Path(__file__).resolve())
evaluation.FOLDER=evaluation.ROOT/'refactor_v2/reports/B_sequence_loss_pilots_v3_3'
evaluation.OUT=evaluation.ROOT/'refactor_v2/reports/B_sequence_loss_quality_v3_2'

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','verify'])
    args=parser.parse_args()
    {'prepare':evaluation.prepare,'run':evaluation.run,'verify':evaluation.replay}[args.action]()
