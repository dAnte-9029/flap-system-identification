"""Complete the original low-order flight baseline under its retained training protocol for two new seeds."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
prior=HERE.parents[1]/'history-identification-20261125/extended'
sys.path.insert(0,str(prior))
import run_extended

if __name__=='__main__':
    seed=int(sys.argv[1]); gpu=sys.argv[2]
    assert seed in (29,43)
    run_extended.SEED=seed
    sys.argv=[sys.argv[0],'run','low','--gpu',gpu,'--output',str(HERE/'backbones'/f'incumbent{seed}')]
    run_extended.main()
